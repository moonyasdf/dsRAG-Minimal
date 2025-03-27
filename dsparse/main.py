# dsrag_minimal/dsparse/main.py
import os
from typing import List, Tuple, Optional, Dict, Any

# Importaciones internas refactorizadas
from dsparse.file_parsing.non_vlm_file_parsing import parse_file_no_vlm, extract_text_from_docx, extract_text_from_pdf # Añadido extract*
from dsparse.sectioning_and_chunking.semantic_sectioning import get_sections_from_str, get_sections_from_pages, str_to_lines # Añadido str_to_lines
from dsparse.sectioning_and_chunking.chunking import chunk_document
from dsparse.models.types import FileParsingConfig, SemanticSectioningConfig, ChunkingConfig, Section, Chunk, Line
from dsparse.file_parsing.file_system import FileSystem, LocalFileSystem

# Importa LLM para pasarlo a semantic_sectioning
from core.llm import LLM

def parse_and_chunk(
    kb_id: str,
    doc_id: str,
    semantic_sectioning_config: SemanticSectioningConfig, # Incluye ahora instancia LLM
    chunking_config: ChunkingConfig,
    file_system: FileSystem, # Siempre LocalFileSystem en esta versión
    file_path: Optional[str] = None,
    text: Optional[str] = None,
    # file_parsing_config ya no es necesario (sin VLM)
    # always_save_page_images: bool = False, # Simplificado, FS local maneja esto
) -> Tuple[List[Section], List[Chunk]]:
    """
    Función principal de dsParse (versión mínima sin VLM).
    Parsea, secciona (semánticamente o no) y chunkea texto o archivos soportados.

    Args:
        kb_id: ID de la base de conocimiento.
        doc_id: ID del documento.
        semantic_sectioning_config: Configuración para sectioning, debe incluir 'llm': LLM instance.
        chunking_config: Configuración para chunking.
        file_system: Instancia de FileSystem (LocalFileSystem).
        file_path: Ruta al archivo (.pdf, .docx, .txt, .md).
        text: Contenido del documento como string.

    Returns:
        Tupla (sections, chunks).
    """
    if not text and not file_path:
        raise ValueError("Either text or file_path must be provided")

    # --- 1. Parseo de Archivo (si se proporciona path) ---
    document_text: str
    pdf_pages: Optional[List[str]] = None # Para guardar páginas PDF si se usa get_sections_from_pages

    if file_path:
        print(f"  - Parsing file: {file_path}...")
        try:
            document_text, pdf_pages = parse_file_no_vlm(file_path)
            if not document_text and not pdf_pages: # Si el parseo falla
                 raise ValueError(f"Could not extract text or pages from {file_path}")
            elif not document_text and pdf_pages: # Si solo hay páginas (raro pero posible)
                 document_text = "\n\n".join(pdf_pages) # Reconstruye texto base
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            raise
    elif text:
        document_text = text
    else:
         # Esto no debería ocurrir debido a la validación inicial
         raise ValueError("Internal error: No text or file_path available.")

    # --- 2. Sectioning (Semántico o Básico) ---
    print("  - Performing sectioning...")
    use_semantic = semantic_sectioning_config.get("use_semantic_sectioning", True)
    sectioning_llm: Optional[LLM] = semantic_sectioning_config.get('llm')

    sections: List[Section]
    document_lines: List[Line] # Líneas generadas internamente por sectioning

    try:
        if use_semantic and sectioning_llm:
            # Elige la función según si tenemos páginas PDF originales
            if pdf_pages:
                sections, document_lines = get_sections_from_pages(
                    pages=pdf_pages,
                    max_characters=semantic_sectioning_config.get('max_characters', 20000),
                    semantic_sectioning_config=semantic_sectioning_config # Pasa toda la config
                )
            else:
                sections, document_lines = get_sections_from_str(
                    document=document_text,
                    max_characters=semantic_sectioning_config.get('max_characters', 20000),
                    semantic_sectioning_config=semantic_sectioning_config # Pasa toda la config
                )
        else:
            # Fallback a no sectioning (o sectioning básico si se implementara)
            # Reusa str_to_lines para obtener una estructura de 'lines'
            print("  - Semantic sectioning disabled or LLM not provided. Using single section.")
            document_lines = str_to_lines(document_text) # Necesario para chunk_document
            sections = [{ # Crea una única sección
                "title": "Full Document", # Título genérico
                "start": 0,
                "end": len(document_lines) - 1 if document_lines else -1,
                "content": document_text
            }]
            # Ajusta si document_lines está vacío
            if not document_lines: sections = []

    except Exception as e:
        print(f"Error during sectioning: {e}")
        # Fallback a una única sección en caso de error
        document_lines = str_to_lines(document_text)
        sections = [{
             "title": "Full Document (Sectioning Failed)",
             "start": 0,
             "end": len(document_lines) - 1 if document_lines else -1,
             "content": document_text
        }]
        if not document_lines: sections = []
        print("  - Falling back to single section due to error.")

    # --- 3. Chunking ---
    print("  - Performing chunking...")
    if not sections or not document_lines:
         print("  - Skipping chunking as no sections/lines were generated.")
         return [], [] # Devuelve listas vacías

    # Obtiene parámetros de chunking o usa defaults
    chunk_size = chunking_config.get('chunk_size', 800)
    min_length = chunking_config.get('min_length_for_chunking', 1600)

    try:
        chunks = chunk_document(
            sections=sections,
            document_lines=document_lines,
            chunk_size=chunk_size,
            min_length_for_chunking=min_length
        )
    except Exception as e:
        print(f"Error during chunking: {e}")
        raise # Relanza error de chunking

    return sections, chunks
