# dsrag_minimal/create_kb.py
# (Versión simplificada enfocada en `create_kb_from_file`)
import os
import time
import warnings # Añadido
from typing import Optional, Dict, Any

# Importaciones absolutas
from core.knowledge_base import KnowledgeBase
from core.embedding import Embedding # Solo base
from core.llm import LLM # Solo base
from core.reranker import Reranker # Solo base
from database.vector.qdrant_db import QdrantVectorDB
from database.chunk.sqlite_db import SQLiteDB
from dsparse.file_parsing.file_system import FileSystem, LocalFileSystem
from metadata import MetadataStorage, LocalMetadataStorage

# Importa parsers directamente
from dsparse.file_parsing.non_vlm_file_parsing import (
    extract_text_from_pdf, extract_text_from_docx, parse_file_no_vlm # Añadido parse_file_no_vlm
)

def create_kb_from_file(
    kb_id: str,
    file_path: str,
    # --- Modelos Inyectables (Opcional, si no se quiere usar defaults de KB) ---
    embedding_model: Optional[Embedding] = None,
    auto_context_model: Optional[LLM] = None,
    reranker: Optional[Reranker] = None,
    # --- Configuración KB ---
    title: Optional[str] = None,
    description: str = "",
    language: str = "en",
    storage_directory: str = "~/dsrag_minimal_data",
    vector_dimension: Optional[int] = None, # Requerido si embedding_model es None
    qdrant_location: Optional[str] = ":memory:",
    # --- Configuración Add Document ---
    auto_context_config: Optional[Dict[str, Any]] = None,
    semantic_sectioning_config: Optional[Dict[str, Any]] = None,
    chunking_config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    # --- Control de Creación ---
    exists_ok: bool = False, # Default a False para helper de creación
    **qdrant_kwargs: Any # Para opciones Qdrant
) -> KnowledgeBase:
    """
    Crea una nueva KnowledgeBase a partir de un único archivo.

    Simplificado para manejar solo la creación desde archivo, no directorios.
    Permite inyectar modelos o usa los defaults de KnowledgeBase.

    Args:
        kb_id: ID único del KB.
        file_path: Ruta al archivo (.pdf, .docx, .txt, .md).
        embedding_model, auto_context_model, reranker: Instancias de modelos (opcional).
        title, description, language: Metadatos del KB.
        storage_directory: Directorio base para almacenamiento.
        vector_dimension: Dimensión de vectores (necesario si embedding_model es None).
        qdrant_location: Ubicación de Qdrant.
        auto_context_config, semantic_sectioning_config, chunking_config, metadata:
            Configuraciones/metadatos para add_document.
        exists_ok: Si es True, permite cargar un KB existente con el mismo ID.
                   Si es False (default), lanza error si el KB ya existe.
        **qdrant_kwargs: Opciones adicionales para Qdrant.

    Returns:
        La instancia KnowledgeBase creada y con el documento añadido.

    Raises:
        ValueError: Si el KB ya existe y exists_ok es False, o si falta vector_dimension.
        FileNotFoundError: Si file_path no existe.
        Exception: Si ocurre un error al parsear o añadir el documento.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Creating Knowledge Base '{kb_id}' from file: {os.path.basename(file_path)}...")

    # 1. Inicializa el KnowledgeBase
    # El constructor de KB manejará la lógica de exists_ok y la inicialización de componentes
    kb = KnowledgeBase(
        kb_id=kb_id,
        embedding_model=embedding_model,
        auto_context_model=auto_context_model,
        reranker=reranker,
        storage_directory=storage_directory,
        vector_dimension=vector_dimension,
        qdrant_location=qdrant_location,
        title=title or kb_id, # Usa kb_id como título por defecto
        description=description,
        language=language,
        exists_ok=exists_ok,
        **qdrant_kwargs
    )

    # 2. Extrae texto del archivo (ya no se hace dentro de KB.add_document)
    #    Esto mantiene la extracción de texto separada de la lógica del KB.
    file_name = os.path.basename(file_path)
    doc_text: Optional[str] = None

    try:
        print(f"  - Parsing text from {file_name}...")
        if file_path.lower().endswith('.pdf'):
            # Nota: parse_file_no_vlm devuelve (texto, lista_paginas)
            doc_text, _ = parse_file_no_vlm(file_path)
        elif file_path.lower().endswith('.docx'):
            doc_text = extract_text_from_docx(file_path)
        elif file_path.lower().endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_name}. Supported: .pdf, .docx, .txt, .md")

        if not doc_text:
            print(f"Warning: No text could be extracted from {file_name}. Document will not be added.")
            return kb # Devuelve el KB vacío

    except Exception as e:
        print(f"Error extracting text from {file_name}: {e}")
        # Decide si relanzar el error o devolver el KB vacío
        raise # Relanza por defecto

    # 3. Añade el documento al KB usando el texto extraído
    try:
        # Usa file_name como doc_id por defecto si no se proporciona un título específico
        effective_doc_id = file_name
        effective_doc_title = title or file_name

        # Asegura que semantic_sectioning_config tenga el LLM si se va a usar
        if semantic_sectioning_config and semantic_sectioning_config.get("use_semantic_sectioning", True):
             if 'llm' not in semantic_sectioning_config and kb.auto_context_model:
                  semantic_sectioning_config['llm'] = kb.auto_context_model
             elif 'llm' not in semantic_sectioning_config:
                  print("Warning: Semantic sectioning enabled but no LLM provided. Disabling sectioning.")
                  semantic_sectioning_config["use_semantic_sectioning"] = False


        kb.add_document(
            doc_id=effective_doc_id,
            text=doc_text, # Pasa el texto extraído
            # file_path=file_path, # Opcional: KB podría querer guardarlo, pero no es necesario para el procesamiento
            document_title=effective_doc_title,
            auto_context_config=auto_context_config or {}, # Pasa diccionarios vacíos si son None
            semantic_sectioning_config=semantic_sectioning_config or {},
            chunking_config=chunking_config or {},
            metadata=metadata or {}
        )
    except Exception as e:
        print(f"Error adding document '{file_name}' to Knowledge Base '{kb_id}': {e}")
        # Considera si eliminar el KB si la adición falla en un helper de creación
        # kb.delete()
        raise # Relanza el error

    print(f"Knowledge Base '{kb_id}' created successfully with document '{file_name}'.")
    return kb

# Se elimina create_kb_from_directory para mantener la versión mínima
