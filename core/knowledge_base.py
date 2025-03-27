# dsrag_minimal/core/knowledge_base.py
import numpy as np
import os
import time
import warnings # Añadido para warnings
from typing import Optional, Union, Dict, List, Any
import concurrent.futures
from tqdm import tqdm
import json # Para serializar/deserializar metadata

# Desde dsparse (paquete hermano)
from dsparse.main import parse_and_chunk
from dsparse.models.types import FileParsingConfig, SemanticSectioningConfig, ChunkingConfig, Section, Chunk
from dsparse.file_parsing.file_system import FileSystem, LocalFileSystem

# Desde core (mismo paquete)
from core.embedding import Embedding, OllamaEmbedding # Importa las clases base/concretas que podrías necesitar aquí
from core.reranker import Reranker, NoReranker
from core.llm import LLM #, OllamaLLM, OpenAILLM, AnthropicLLM # No necesitas importar las concretas aquí si se inyectan
from core.rse import get_relevance_values, get_best_segments, get_meta_document, RSE_PARAMS_PRESETS
from core.auto_context import (
    get_document_title, get_document_summary, get_section_summary,
    get_chunk_header, get_segment_header, _load_prompt
)

# Desde database (paquete hermano)
from database.vector.qdrant_db import QdrantVectorDB
from database.vector.types import MetadataFilter, Vector, ChunkMetadata, VectorSearchResult
from database.chunk.sqlite_db import SQLiteDB

# Desde metadata (módulo en la raíz)
from metadata import MetadataStorage, LocalMetadataStorage

# Constantes para nombres de archivos de prompts
PROMPT_FILES = {
    "doc_title": "auto_context_doc_title.txt",
    "doc_summary": "auto_context_doc_summary.txt",
    "section_summary": "auto_context_section_summary.txt",
}

class KnowledgeBase:
    """
    Clase principal para gestionar una base de conocimiento en dsRAG (versión mínima).
    Utiliza Qdrant para vectores, SQLite para chunks y FS local por defecto.
    Permite la inyección de modelos de embedding, LLM y reranker.
    """
    def __init__(
        self,
        kb_id: str,
        # --- Modelos Inyectables ---
        embedding_model: Optional[Embedding] = None, # Modelo de embedding (OllamaEmbedding, OpenAIEmbedding, etc.)
        auto_context_model: Optional[LLM] = None,   # LLM para AutoContext (OllamaLLM, OpenAILLM, etc.)
        reranker: Optional[Reranker] = None,         # Reranker (NoReranker por defecto)
        # --- Configuración de Almacenamiento (Fija a local/Qdrant/SQLite) ---
        storage_directory: str = "~/dsrag_minimal_data", # Directorio base para datos locales
        vector_dimension: Optional[int] = None, # Requerido si embedding_model no lo proporciona o es nuevo
        qdrant_location: Optional[str] = ":memory:", # Qdrant en memoria por defecto
        # --- Metadatos del KB ---
        title: str = "",
        description: str = "",
        language: str = "en",
        # --- Control de Inicialización ---
        exists_ok: bool = True, # Permitir cargar KB existente
        # --- Opcional: Almacenamiento de Metadatos Personalizado ---
        metadata_storage: Optional[MetadataStorage] = None,
        # --- Opciones Qdrant adicionales (pasadas a QdrantVectorDB) ---
        **qdrant_kwargs: Any
    ):
        """
        Inicializa la KnowledgeBase mínima.

        Args:
            kb_id (str): ID único para la base de conocimiento.
            embedding_model (Embedding, opcional): Instancia del modelo de embedding. Si es None, se debe proporcionar vector_dimension.
            auto_context_model (LLM, opcional): Instancia del LLM para AutoContext. Si es None, AutoContext fallará.
            reranker (Reranker, opcional): Instancia del reranker. Defaults to NoReranker.
            storage_directory (str): Directorio para SQLite, metadatos locales y FS local.
            vector_dimension (int, opcional): Dimensión de los vectores. Requerido si embedding_model es None o no tiene `dimension`.
            qdrant_location (str, opcional): Ubicación para Qdrant (':memory:', path local, o URL).
            title (str): Título del KB.
            description (str): Descripción del KB.
            language (str): Idioma principal del KB (para prompts).
            exists_ok (bool): Si es True, carga el KB si ya existe. Si es False y existe, lanza error.
            metadata_storage (MetadataStorage, opcional): Instancia para almacenar metadatos. Default: LocalMetadataStorage.
            **qdrant_kwargs: Argumentos adicionales para QdrantClient (e.g., api_key, host, port).
        """
        self.kb_id = kb_id
        self.storage_directory = os.path.expanduser(storage_directory)

        # Configuración de almacenamiento de metadatos
        self.metadata_storage = metadata_storage or LocalMetadataStorage(self.storage_directory)

        # Determina la dimensión del vector
        if vector_dimension is None:
            if embedding_model and hasattr(embedding_model, 'dimension') and embedding_model.dimension:
                vector_dimension = embedding_model.dimension
            else:
                raise ValueError("vector_dimension must be provided if embedding_model is None or doesn't have a 'dimension' attribute.")
        self.vector_dimension = vector_dimension

        # Flag para indicar si es un KB nuevo o cargado
        is_new_kb = not self.metadata_storage.kb_exists(self.kb_id)

        if not is_new_kb and not exists_ok:
            raise ValueError(f"Knowledge Base '{kb_id}' already exists. Use exists_ok=True to load.")

        if is_new_kb:
            # --- Inicialización de KB Nuevo ---
            print(f"Initializing new Knowledge Base: {kb_id}")
            self.kb_metadata = {
                "kb_id": kb_id, # Guardar kb_id en metadatos también
                "title": title or kb_id,
                "description": description,
                "language": language,
                "created_on": int(time.time()),
                "vector_dimension": self.vector_dimension # Guardar dimensión
            }
            # Inicializa componentes por defecto o con los proporcionados
            self.embedding_model = embedding_model # Puede ser None si solo se quiere cargar datos
            self.auto_context_model = auto_context_model # Puede ser None
            self.reranker = reranker or NoReranker()
            # Fija los componentes de almacenamiento
            self.vector_db = QdrantVectorDB(kb_id=self.kb_id, vector_dimension=self.vector_dimension, location=qdrant_location, **qdrant_kwargs)
            self.chunk_db = SQLiteDB(kb_id=self.kb_id, storage_directory=self.storage_directory)
            self.file_system = LocalFileSystem(base_path=os.path.join(self.storage_directory, "kb_files")) # Subdirectorio para archivos

            self._save_metadata() # Guarda la configuración inicial
        else:
            # --- Carga de KB Existente ---
            print(f"Loading existing Knowledge Base: {kb_id}")
            self._load_metadata()
            # Comprueba la dimensión del vector
            if self.kb_metadata.get("vector_dimension") != self.vector_dimension:
                 warnings.warn(f"Provided vector_dimension ({self.vector_dimension}) differs from saved metadata ({self.kb_metadata.get('vector_dimension')}). Using saved value.")
                 self.vector_dimension = self.kb_metadata.get("vector_dimension")

            # Carga componentes de almacenamiento (siempre Qdrant/SQLite/LocalFS)
            # Qdrant: La conexión se establece al instanciar
            self.vector_db = QdrantVectorDB(kb_id=self.kb_id, vector_dimension=self.vector_dimension, location=qdrant_location, **qdrant_kwargs)
            self.chunk_db = SQLiteDB(kb_id=self.kb_id, storage_directory=self.storage_directory)
            self.file_system = LocalFileSystem(base_path=os.path.join(self.storage_directory, "kb_files"))

            # Carga/sobrescribe modelos inyectables
            self.embedding_model = embedding_model or self._load_component_from_metadata('embedding_model', Embedding)
            self.auto_context_model = auto_context_model or self._load_component_from_metadata('auto_context_model', LLM)
            self.reranker = reranker or self._load_component_from_metadata('reranker', Reranker, default_instance=NoReranker())

            # Asegura que los modelos cargados tengan la dimensión correcta si aplica
            if self.embedding_model and hasattr(self.embedding_model, 'dimension') and self.embedding_model.dimension != self.vector_dimension:
                 warnings.warn(f"Loaded embedding model dimension ({self.embedding_model.dimension}) differs from KB vector dimension ({self.vector_dimension}). This might cause issues.")

            # Guarda la configuración actualizada (especialmente si se inyectaron modelos nuevos)
            self._save_metadata()

    def _save_metadata(self):
        """Guarda metadatos y configuración de componentes inyectables."""
        components_config = {}
        if self.embedding_model:
            components_config['embedding_model'] = self.embedding_model.to_dict()
        if self.auto_context_model:
            components_config['auto_context_model'] = self.auto_context_model.to_dict()
        if self.reranker:
            components_config['reranker'] = self.reranker.to_dict()
        # No es necesario guardar config de Qdrant/SQLite/LocalFS, se reconstruyen con kb_id/path

        full_data = {**self.kb_metadata, "components": components_config}
        try:
            self.metadata_storage.save(full_data, self.kb_id)
        except Exception as e:
            print(f"Error saving metadata for KB '{self.kb_id}': {e}")

    def _load_metadata(self):
        """Carga metadatos y configuración de componentes."""
        try:
            data = self.metadata_storage.load(self.kb_id)
            self.kb_metadata = {k: v for k, v in data.items() if k != "components"}
            # Asegura que kb_id y vector_dimension estén en metadata
            self.kb_metadata.setdefault('kb_id', self.kb_id)
            self.kb_metadata.setdefault('vector_dimension', self.vector_dimension)
            # Carga la configuración de componentes para usarla en _load_component_from_metadata
            self._loaded_components_config = data.get("components", {})
        except Exception as e:
            print(f"Error loading metadata for KB '{self.kb_id}': {e}")
            # Si falla la carga, inicializa con valores mínimos
            self.kb_metadata = {"kb_id": self.kb_id, "vector_dimension": self.vector_dimension}
            self._loaded_components_config = {}

    def _load_component_from_metadata(self, component_key: str, base_class: type, default_instance: Optional[Any] = None) -> Optional[Any]:
        """Carga una instancia de componente desde la configuración guardada."""
        config = self._loaded_components_config.get(component_key)
        if config and isinstance(config, dict):
            try:
                # Usa el método from_dict de la clase base correspondiente
                return base_class.from_dict(config.copy()) # Pasa una copia
            except ValueError as e:
                print(f"Warning: Could not load {component_key} from metadata: {e}. Using default.")
                return default_instance
            except Exception as e:
                 print(f"Unexpected error loading {component_key} from metadata: {e}. Using default.")
                 return default_instance
        return default_instance

    # --- Gestión de Documentos ---

    def add_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        file_path: Optional[str] = None,
        document_title: Optional[str] = None,
        # --- Configuraciones de Procesamiento ---
        auto_context_config: Optional[Dict[str, Any]] = None,
        # file_parsing_config: Ya no se necesita VLM
        semantic_sectioning_config: Optional[Dict[str, Any]] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
        # --- Metadatos Adicionales ---
        supp_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Añade un documento al KB. Procesa, secciona, chunkea, genera contexto y embeddings.

        Args:
            doc_id (str): ID único del documento.
            text (str, opcional): Contenido del documento como texto.
            file_path (str, opcional): Ruta al archivo (.txt, .md, .pdf, .docx).
            document_title (str, opcional): Título del documento.
            auto_context_config (dict, opcional): Configuración para AutoContext.
            semantic_sectioning_config (dict, opcional): Configuración para sectioning.
            chunking_config (dict, opcional): Configuración para chunking.
            supp_id (str, opcional): ID suplementario.
            metadata (dict, opcional): Metadatos adicionales a nivel de documento.
        """
        # Validaciones iniciales
        if not text and not file_path:
            raise ValueError("Either 'text' or 'file_path' must be provided.")
        if text and file_path:
            warnings.warn("Both 'text' and 'file_path' provided. Using 'text'.")
        if "/" in doc_id or "\\" in doc_id:
            raise ValueError("doc_id cannot contain path separators ('/' or '\\').")
        if self.chunk_db.get_document(doc_id):
             print(f"Document '{doc_id}' already exists in KB '{self.kb_id}'. Skipping.")
             return

        # Configuración por defecto si no se proporciona
        auto_context_cfg = auto_context_config or {}
        semantic_sectioning_cfg = semantic_sectioning_config or {}
        chunking_cfg = chunking_config or {}
        doc_metadata = metadata or {}

        # Asegura que el modelo de embedding esté disponible si se necesita
        if not self.embedding_model:
            raise RuntimeError("Cannot add document: embedding_model is not set.")

        print(f"Processing document: {doc_id}...")
        start_time_total = time.time()

        try:
            # --- 1. Parseo y Chunking (Usando dsparse refactorizado) ---
            start_time_parse = time.time()
            # Prepara config para dsparse (sin VLM)
            dsparse_file_parsing_cfg: FileParsingConfig = {"use_vlm": False} # VLM deshabilitado

            # Inyecta el modelo LLM para sectioning si está disponible
            semantic_sectioning_cfg['llm'] = self.auto_context_model # Reusa el modelo de auto_context

            sections, chunks = parse_and_chunk(
                kb_id=self.kb_id,
                doc_id=doc_id,
                file_path=file_path,
                text=text,
                file_parsing_config=dsparse_file_parsing_cfg,
                semantic_sectioning_config=semantic_sectioning_cfg,
                chunking_config=chunking_cfg,
                file_system=self.file_system, # Siempre LocalFileSystem
            )
            parse_time = time.time() - start_time_parse
            print(f"  - Parsing & Chunking completed in {parse_time:.2f}s. Found {len(sections)} sections, {len(chunks)} chunks.")

            if not chunks:
                print(f"Warning: No chunks generated for document {doc_id}. Skipping.")
                return

            # --- 2. AutoContext (Títulos y Resúmenes) ---
            start_time_context = time.time()
            document_text_for_context = "\n".join(s['content'] for s in sections) # Reconstruye texto
            chunks_with_context = self._run_auto_context(
                chunks=chunks,
                sections=sections,
                doc_id=doc_id,
                document_text=document_text_for_context,
                provided_title=document_title,
                config=auto_context_cfg
            )
            context_time = time.time() - start_time_context
            print(f"  - AutoContext completed in {context_time:.2f}s.")

            # --- 3. Preparación para Embedding ---
            # Prepend chunk headers (ahora parte de _run_auto_context)
            texts_to_embed = [
                 f"{c.get('chunk_header', '')}\n\n{c['content']}".strip()
                 for c in chunks_with_context
            ]

            # --- 4. Generación de Embeddings ---
            start_time_embed = time.time()
            # Procesa en batches para modelos API o manejo eficiente de memoria local
            batch_size = 100 # Ajustable
            all_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i+batch_size]
                batch_embeddings = self.embedding_model.get_embeddings(batch_texts, input_type="document")
                # Valida dimensiones
                for emb in batch_embeddings:
                    if len(emb) != self.vector_dimension:
                         raise ValueError(f"Embedding dimension mismatch for doc {doc_id}. Expected {self.vector_dimension}, got {len(emb)}. Check model config.")
                all_embeddings.extend(batch_embeddings)

            embed_time = time.time() - start_time_embed
            print(f"  - Embedding completed in {embed_time:.2f}s.")

            # --- 5. Almacenamiento en DBs ---
            start_time_db = time.time()
            # 5a. Almacenar Chunks en SQLiteDB
            self.chunk_db.add_document(
                doc_id=doc_id,
                # Pasa solo los datos necesarios para la tabla SQLite
                chunks={
                    i: {
                        "chunk_text": c["content"],
                        "document_title": c.get("document_title"),
                        "document_summary": c.get("document_summary"),
                        "section_title": c.get("section_title"),
                        "section_summary": c.get("section_summary"),
                        "chunk_page_start": c.get("page_start"),
                        "chunk_page_end": c.get("page_end"),
                        "is_visual": c.get("is_visual", False) # Asegura default
                    } for i, c in enumerate(chunks_with_context)
                },
                supp_id=supp_id,
                metadata=doc_metadata # Guarda metadata a nivel de documento aquí
            )

            # 5b. Almacenar Vectores en QdrantVectorDB
            # Prepara metadatos para Qdrant (incluye metadata a nivel de chunk)
            qdrant_metadata = []
            for i, chunk in enumerate(chunks_with_context):
                 chunk_meta: ChunkMetadata = {
                      "doc_id": doc_id,
                      "chunk_index": i,
                      "chunk_header": chunk.get("chunk_header", ""),
                      "chunk_text": chunk["content"], # Puede ser útil para Qdrant
                      # Incluye metadatos a nivel de documento y otros relevantes
                      **(doc_metadata or {}),
                      "page_start": chunk.get("page_start"),
                      "page_end": chunk.get("page_end"),
                      "is_visual": chunk.get("is_visual", False),
                      "document_title": chunk.get("document_title"),
                      # No incluir resúmenes largos en metadata vectorial usualmente
                      # "document_summary": chunk.get("document_summary"),
                      "section_title": chunk.get("section_title"),
                      # "section_summary": chunk.get("section_summary"),
                 }
                 # Limpia Nones para Qdrant
                 qdrant_metadata.append({k: v for k, v in chunk_meta.items() if v is not None})

            self.vector_db.add_vectors(vectors=all_embeddings, metadata=qdrant_metadata)
            db_time = time.time() - start_time_db
            print(f"  - Database storage completed in {db_time:.2f}s.")

            # --- 6. Actualizar Metadatos del KB ---
            # Podríamos actualizar estadísticas como número de documentos, etc.
            # self._update_kb_stats(...)
            self._save_metadata() # Guarda metadatos actualizados

            total_time = time.time() - start_time_total
            print(f"Document '{doc_id}' added successfully in {total_time:.2f}s.")

        except Exception as e:
            import traceback
            print(f"Error adding document '{doc_id}': {e}")
            print(traceback.format_exc())
            # Considera si eliminar datos parcialmente añadidos en caso de error
            # self.delete_document(doc_id)
            raise # Relanza la excepción para que el llamador la maneje

    def _run_auto_context(self, chunks: List[Chunk], sections: List[Section], doc_id: str, document_text: str, provided_title: Optional[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ejecuta la lógica de AutoContext (títulos, resúmenes, headers)."""
        if not self.auto_context_model:
            print("Warning: auto_context_model not provided. Skipping AutoContext.")
            # Añade stubs vacíos para que el resto del código funcione
            for chunk in chunks:
                 chunk["document_title"] = provided_title or doc_id
                 chunk["document_summary"] = ""
                 chunk["section_title"] = ""
                 chunk["section_summary"] = ""
                 chunk["chunk_header"] = get_chunk_header(chunk["document_title"])
            return chunks

        kb_lang = self.kb_metadata.get('language', 'en')

        # 1. Título del Documento
        doc_title = provided_title
        if not doc_title and config.get("use_generated_title", True):
            prompt_guidance = config.get("document_title_guidance", "")
            prompt = _load_prompt(PROMPT_FILES["doc_title"]).format(
                document_title_guidance=prompt_guidance,
                non_english_addendum=_load_prompt("language_addendum.txt") if kb_lang != 'en' else "",
                document_text=document_text[:8000], # Limita texto para el prompt
                truncation_message="Note: Document text provided is truncated." # Mensaje simple
            )
            doc_title = get_response_via_instance(self.auto_context_model, prompt=prompt).strip()
        doc_title = doc_title or doc_id # Fallback a doc_id

        # 2. Resumen del Documento
        doc_summary = ""
        if config.get("get_document_summary", True):
            prompt_guidance = config.get("document_summarization_guidance", "")
            prompt = _load_prompt(PROMPT_FILES["doc_summary"]).format(
                document_summarization_guidance=prompt_guidance,
                non_english_addendum=_load_prompt("language_addendum.txt") if kb_lang != 'en' else "",
                document_title=doc_title,
                document_text=document_text[:10000], # Límite mayor para resumen
                truncation_message="Note: Document text provided is truncated."
            )
            doc_summary = get_response_via_instance(self.auto_context_model, prompt=prompt).strip()

        # 3. Resúmenes de Sección (Opcional)
        # Precalcula resúmenes de sección para evitar llamadas LLM repetidas
        section_summaries = {} # {section_index: summary}
        if config.get("get_section_summaries", False):
             prompt_guidance = config.get("section_summarization_guidance", "")
             # Procesar en batch o paralelo si es posible y el LLM lo soporta
             for i, section in enumerate(sections):
                  prompt = _load_prompt(PROMPT_FILES["section_summary"]).format(
                       section_summarization_guidance=prompt_guidance,
                       non_english_addendum=_load_prompt("language_addendum.txt") if kb_lang != 'en' else "",
                       document_title=doc_title,
                       section_title=section['title'],
                       section_text=section['content'][:8000] # Limita texto de sección
                  )
                  section_summaries[i] = get_response_via_instance(self.auto_context_model, prompt=prompt).strip()

        # 4. Aplica contexto a cada chunk y genera header
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            section_idx = chunk.get('section_index')
            section_title = ""
            section_summary = ""
            if section_idx is not None and section_idx < len(sections):
                 section_title = sections[section_idx]['title']
                 section_summary = section_summaries.get(section_idx, "") # Obtiene resumen precalculado

            # Construye el header del chunk
            chunk_header = get_chunk_header(
                document_title=doc_title,
                document_summary=doc_summary,
                section_title=section_title,
                section_summary=section_summary,
            )

            # Crea un nuevo dict para el chunk enriquecido
            enriched_chunk = chunk.copy()
            enriched_chunk["document_title"] = doc_title
            enriched_chunk["document_summary"] = doc_summary
            enriched_chunk["section_title"] = section_title
            enriched_chunk["section_summary"] = section_summary
            enriched_chunk["chunk_header"] = chunk_header
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks

    # --- Funciones de utilidad (simplificadas o movidas) ---

    # _get_chunk_text, _get_is_visual, etc. ahora son parte de SQLiteDB

    def _get_segment_header(self, doc_id: str, chunk_index: int) -> str:
        """Genera header para un segmento (solo contexto de documento)."""
        # Obtiene info del primer chunk del segmento desde SQLiteDB
        doc_title = self.chunk_db.get_document_title(doc_id, chunk_index) or ""
        doc_summary = self.chunk_db.get_document_summary(doc_id, chunk_index) or ""
        return get_segment_header(document_title=doc_title, document_summary=doc_summary)

    def _get_segment_content_from_database(self, doc_id: str, chunk_start: int, chunk_end: int, return_mode: str) -> Union[str, List[str]]:
        """Obtiene contenido del segmento (texto o imágenes) desde SQLiteDB y LocalFileSystem."""
        # return_mode: "text", "page_images", "dynamic"

        # Determina el modo si es dinámico
        effective_return_mode = return_mode
        if return_mode == "dynamic":
            segment_is_visual = False
            for idx in range(chunk_start, chunk_end):
                if self.chunk_db.get_is_visual(doc_id, idx):
                    segment_is_visual = True
                    break
            effective_return_mode = "page_images" if segment_is_visual else "text"

        # Obtiene contenido según el modo
        if effective_return_mode == "text":
            segment_header = self._get_segment_header(doc_id, chunk_start)
            chunk_texts = [
                self.chunk_db.get_chunk_text(doc_id, idx) or ""
                for idx in range(chunk_start, chunk_end)
            ]
            full_text = "\n".join(chunk_texts).strip()
            # Prepend header solo si hay contenido
            return f"{segment_header}\n\n{full_text}".strip() if full_text else ""
        else: # effective_return_mode == "page_images"
            # Obtiene números de página de inicio/fin del segmento
            page_start_num, _ = self.chunk_db.get_chunk_page_numbers(doc_id, chunk_start)
            _, page_end_num = self.chunk_db.get_chunk_page_numbers(doc_id, chunk_end - 1)

            if page_start_num is None or page_end_num is None:
                 print(f"Warning: Page numbers not found for segment {doc_id}:{chunk_start}-{chunk_end}. Falling back to text.")
                 # Fallback a texto si faltan números de página
                 return self._get_segment_content_from_database(doc_id, chunk_start, chunk_end, "text")

            # Obtiene rutas de archivo de imagen desde LocalFileSystem
            image_paths = self.file_system.get_files(self.kb_id, doc_id, page_start_num, page_end_num)

            if not image_paths:
                 print(f"Warning: Page images not found for segment {doc_id}:{chunk_start}-{chunk_end} (pages {page_start_num}-{page_end_num}). Falling back to text.")
                 # Fallback a texto si no se encuentran imágenes
                 return self._get_segment_content_from_database(doc_id, chunk_start, chunk_end, "text")

            return image_paths # Devuelve lista de rutas locales

    # --- Lógica de Consulta (Refactorizada para Qdrant y Reranker inyectado) ---

    def _search(self, query: str, top_k: int, metadata_filter: Optional[MetadataFilter] = None) -> List[VectorSearchResult]:
        """Busca en Qdrant, luego aplica reranker."""
        if not self.embedding_model:
             raise RuntimeError("Cannot search: embedding_model is not set.")

        # 1. Obtiene embedding de la consulta
        query_vector = self.embedding_model.get_embeddings(query, input_type="query")
        if query_vector is None or len(query_vector) == 0:
             print(f"Warning: Could not generate query vector for query: {query}")
             return []

        # 2. Busca en Qdrant
        # Nota: El filtro de metadatos necesita ser compatible con Qdrant aquí
        try:
            search_results = self.vector_db.search(query_vector, top_k, metadata_filter)
        except Exception as e:
            print(f"Error during vector search for query '{query}': {e}")
            return []

        if not search_results:
            return []

        # 3. Aplica Reranker (podría ser NoReranker)
        # Prepara documentos para el reranker (si es necesario)
        # El formato puede variar según el reranker, aquí asumimos un formato genérico
        docs_for_rerank = [{
             'content': f"{r['metadata'].get('chunk_header', '')}\n\n{r['metadata'].get('chunk_text', '')}".strip(),
             'metadata': r['metadata'], # Pasa la metadata original
             'similarity': r.get('similarity', 0.0) # Pasa el score original
             } for r in search_results
        ]

        try:
            # Llama al método rerank del reranker inyectado
            reranked_docs = self.reranker.rerank(query, docs_for_rerank)
        except Exception as e:
            print(f"Error during reranking for query '{query}': {e}. Returning results from vector search.")
            # Fallback a los resultados originales de Qdrant si el reranker falla
            return search_results # Devuelve los resultados originales

        # 4. Formatea la salida del reranker al formato VectorSearchResult esperado
        # Asegura que el score de similaridad exista y sea un float
        final_results = []
        for doc in reranked_docs:
            similarity_score = doc.get('similarity') # El reranker debería añadir/actualizar esto
            if similarity_score is None:
                # Si el reranker no dio score, intenta usar el original o pon 0
                original_meta = doc.get('metadata', {})
                # Busca el score original (puede no estar si el VectorDB no lo dio)
                original_result = next((r for r in search_results if r.get('metadata', {}).get('doc_id') == original_meta.get('doc_id') and r.get('metadata', {}).get('chunk_index') == original_meta.get('chunk_index')), None)
                similarity_score = original_result.get('similarity', 0.0) if original_result else 0.0
            elif not isinstance(similarity_score, (float, int)):
                 try:
                     similarity_score = float(similarity_score)
                 except (ValueError, TypeError):
                     similarity_score = 0.0 # Default a 0 si no se puede convertir

            final_results.append(VectorSearchResult(
                doc_id=doc['metadata'].get('doc_id'),
                metadata=doc['metadata'],
                similarity=float(similarity_score), # Asegura float
                vector=None # Vector no es necesario después del reranking
            ))

        return final_results

    def _get_all_ranked_results(self, search_queries: List[str], metadata_filter: Optional[MetadataFilter] = None) -> List[List[VectorSearchResult]]:
        """Ejecuta múltiples búsquedas en paralelo."""
        # Limita la concurrencia para evitar sobrecargar APIs/DB local
        max_workers = min(len(search_queries), 4) # Ajustable
        all_ranked_results = []
        # Usa ThreadPoolExecutor para I/O (llamadas API embedding/rerank, DB query)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Mapea cada query a la función _search
            future_to_query = {executor.submit(self._search, query, 200, metadata_filter): query for query in search_queries} # Aumenta top_k para RSE
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    all_ranked_results.append(results)
                except Exception as e:
                    print(f"Error executing search for query '{query}': {e}")
                    all_ranked_results.append([]) # Añade lista vacía en caso de error

        # Asegura que el orden corresponda al de search_queries si es necesario
        # (as_completed no garantiza orden, pero para RSE el orden de las listas externas no importa tanto)
        return all_ranked_results

    def query(
        self,
        search_queries: List[str],
        rse_params: Union[Dict[str, Any], str] = "balanced", # Permite presets o dict
        metadata_filter: Optional[MetadataFilter] = None,
        return_mode: str = "text", # "text", "page_images", "dynamic"
    ) -> List[Dict[str, Any]]:
        """Realiza una consulta al KB usando RSE."""

        if not search_queries: return [] # Devuelve vacío si no hay queries

        print(f"Querying KB '{self.kb_id}' with {len(search_queries)} queries...")
        start_time_query = time.time()

        # --- 1. Parámetros RSE ---
        if isinstance(rse_params, str):
            params = RSE_PARAMS_PRESETS.get(rse_params)
            if params is None:
                raise ValueError(f"Invalid RSE preset name: {rse_params}. Use 'balanced', 'precision', or 'find_all'.")
        elif isinstance(rse_params, dict):
            params = rse_params
        else: # Default a balanced si no es string ni dict
            params = RSE_PARAMS_PRESETS["balanced"]

        # Usa defaults para parámetros RSE faltantes
        rse_config = {**RSE_PARAMS_PRESETS["balanced"], **params}
        # Ajusta overall_max_length según el número de queries
        rse_config["overall_max_length"] += (len(search_queries) - 1) * rse_config["overall_max_length_extension"]

        # --- 2. Búsqueda y Reranking ---
        start_time_search = time.time()
        all_ranked_results = self._get_all_ranked_results(search_queries, metadata_filter)
        search_time = time.time() - start_time_search
        print(f"  - Vector search & reranking took {search_time:.2f}s")

        # Verifica si hubo resultados
        if not any(all_ranked_results):
             print("  - No relevant chunks found.")
             return []

        # --- 3. Preparación para RSE (Meta-documento) ---
        start_time_rse_prep = time.time()
        document_splits, document_start_points, unique_doc_ids = get_meta_document(
            all_ranked_results=all_ranked_results,
            top_k_for_document_selection=rse_config["top_k_for_document_selection"],
        )
        if not document_splits:
             print("  - No documents selected for RSE.")
             return []
        meta_doc_len = document_splits[-1]
        rse_prep_time = time.time() - start_time_rse_prep
        print(f"  - RSE meta-document prep took {rse_prep_time:.2f}s ({len(unique_doc_ids)} docs)")

        # --- 4. Cálculo de Relevancia RSE ---
        start_time_relevance = time.time()
        all_relevance_values = get_relevance_values(
            all_ranked_results=all_ranked_results,
            meta_document_length=meta_doc_len,
            document_start_points=document_start_points,
            unique_document_ids=unique_doc_ids,
            irrelevant_chunk_penalty=rse_config["irrelevant_chunk_penalty"],
            decay_rate=rse_config["decay_rate"],
            chunk_length_adjustment=rse_config["chunk_length_adjustment"],
        )
        relevance_time = time.time() - start_time_relevance
        print(f"  - RSE relevance calculation took {relevance_time:.2f}s")

        # --- 5. Obtención de Mejores Segmentos RSE ---
        start_time_segments = time.time()
        best_segments_indices, scores = get_best_segments(
            all_relevance_values=all_relevance_values,
            document_splits=document_splits,
            max_length=rse_config["max_length"],
            overall_max_length=rse_config["overall_max_length"],
            minimum_value=rse_config["minimum_value"],
        )
        segments_time = time.time() - start_time_segments
        print(f"  - RSE segment optimization took {segments_time:.2f}s. Found {len(best_segments_indices)} segments.")

        if not best_segments_indices:
            return []

        # --- 6. Formateo y Recuperación de Contenido del Segmento ---
        start_time_content = time.time()
        relevant_segment_info = []
        # Mapea índices del meta-documento a doc_id y chunk_index relativos
        for i, (start, end) in enumerate(best_segments_indices):
             # Encuentra el doc_id correspondiente
             doc_idx = -1
             for idx, split_point in enumerate(document_splits):
                  if start < split_point:
                       doc_idx = idx
                       break
             if doc_idx == -1: continue # Debería encontrar un doc

             doc_id = unique_doc_ids[doc_idx]
             doc_start_in_meta = document_start_points[doc_id]
             chunk_start_relative = start - doc_start_in_meta
             chunk_end_relative = end - doc_start_in_meta

             # Obtiene números de página
             page_start, page_end = self.chunk_db.get_chunk_page_numbers(doc_id, chunk_start_relative)
             _, last_chunk_page_end = self.chunk_db.get_chunk_page_numbers(doc_id, chunk_end_relative - 1)
             final_page_end = last_chunk_page_end if last_chunk_page_end is not None else page_start

             segment_data = {
                  "doc_id": doc_id,
                  "chunk_start": chunk_start_relative,
                  "chunk_end": chunk_end_relative, # Exclusivo
                  "score": scores[i],
                  "segment_page_start": page_start,
                  "segment_page_end": final_page_end,
                  # --- Campos deprecados por compatibilidad ---
                  "chunk_page_start": page_start,
                  "chunk_page_end": final_page_end,
                  "text": "" # Placeholder, se llenará después
             }
             relevant_segment_info.append(segment_data)

        # Recupera contenido para cada segmento (podría paralelizarse)
        # Nota: La recuperación de contenido (texto o imágenes) puede ser la parte más lenta aquí
        # Si se requiere < 3s, esta parte debe ser muy rápida.
        # SQLite debería ser rápido para texto. FS local también.
        segment_contents = {} # { (doc_id, start, end): content }
        for segment in relevant_segment_info:
             key = (segment["doc_id"], segment["chunk_start"], segment["chunk_end"])
             segment_contents[key] = self._get_segment_content_from_database(
                  segment["doc_id"], segment["chunk_start"], segment["chunk_end"], return_mode
             )

        # Añade contenido a los resultados
        for segment in relevant_segment_info:
            key = (segment["doc_id"], segment["chunk_start"], segment["chunk_end"])
            content = segment_contents.get(key, "") # Obtiene contenido recuperado
            segment["content"] = content
            # Llena 'text' por compatibilidad si el contenido es string
            if isinstance(content, str):
                segment["text"] = content

        content_time = time.time() - start_time_content
        print(f"  - Segment content retrieval took {content_time:.2f}s")

        total_query_time = time.time() - start_time_query
        print(f"Total query time: {total_query_time:.2f}s")

        return relevant_segment_info

    # --- Otros métodos ---
    def delete_document(self, doc_id: str):
        """Elimina un documento del KB."""
        try:
             print(f"Deleting document '{doc_id}' from KB '{self.kb_id}'...")
             self.chunk_db.remove_document(doc_id)
             self.vector_db.remove_document(doc_id)
             # Asume que file_system tiene un método para eliminar archivos del doc
             if hasattr(self.file_system, 'delete_directory'):
                  self.file_system.delete_directory(self.kb_id, doc_id)
             print(f"Document '{doc_id}' deleted.")
             self._save_metadata() # Actualiza metadatos si es necesario
        except Exception as e:
             print(f"Error deleting document '{doc_id}': {e}")

    def delete(self):
        """Elimina completamente el KB."""
        print(f"Deleting Knowledge Base '{self.kb_id}'...")
        try:
             self.chunk_db.delete()
             print("  - Chunk DB deleted.")
        except Exception as e: print(f"  - Error deleting Chunk DB: {e}")
        try:
             self.vector_db.delete()
             print("  - Vector DB deleted.")
        except Exception as e: print(f"  - Error deleting Vector DB: {e}")
        try:
             if hasattr(self.file_system, 'delete_kb'):
                   self.file_system.delete_kb(self.kb_id)
                   print("  - Associated files deleted.")
        except Exception as e: print(f"  - Error deleting associated files: {e}")
        try:
             self.metadata_storage.delete(self.kb_id)
             print("  - Metadata deleted.")
        except Exception as e: print(f"  - Error deleting metadata: {e}")
        print(f"Knowledge Base '{self.kb_id}' deletion process completed.")

# Helper para cargar prompts (colocar dentro de KnowledgeBase o en un módulo util)
def _load_prompt(filename: str) -> str:
    """Carga un prompt desde el directorio 'prompts'."""
    # Determina la ruta relativa al archivo actual
    prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
    filepath = os.path.join(prompt_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}")
        # Devuelve un string vacío o lanza un error más específico
        return ""
    except Exception as e:
        print(f"Error loading prompt from {filepath}: {e}")
        return ""
