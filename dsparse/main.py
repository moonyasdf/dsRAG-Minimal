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
    semantic_sectioning_config: SemanticSectioningConfig,
    chunking_config: ChunkingConfig,
    file_system: FileSystem,
    file_path: Optional[str] = None,
    text: Optional[str] = None,
    # Elimina file_parsing_config como parámetro
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
    pdf_pages: Optional[List[str]] = None

    if file_path:
        print(f"  - Parsing file: {file_path}...")
        try:
            # parse_file_no_vlm maneja diferentes tipos de archivo
            document_text, pdf_pages = parse_file_no_vlm(file_path)
            if not document_text and not pdf_pages:
                 raise ValueError(f"Could not extract text or pages from {file_path}")
            elif not document_text and pdf_pages:
                 document_text = "\n\n".join(pdf_pages)
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            raise
    elif text:
        document_text = text
    else:
         raise ValueError("Internal error: No text or file_path available.")

    # --- 2. Sectioning (Semántico o Básico) ---
    print("  - Performing sectioning...")
    use_semantic = semantic_sectioning_config.get("use_semantic_sectioning", True)
    sectioning_llm: Optional[LLM] = semantic_sectioning_config.get('llm')

    sections: List[Section]
    document_lines: List[Line]

    try:
        if use_semantic and sectioning_llm:
            if pdf_pages:
                sections, document_lines = get_sections_from_pages(
                    pages=pdf_pages,
                    semantic_sectioning_config=semantic_sectioning_config
                    # max_characters se puede pasar dentro de config
                )
            else:
                sections, document_lines = get_sections_from_str(
                    document=document_text,
                    semantic_sectioning_config=semantic_sectioning_config
                    # max_characters se puede pasar dentro de config
                )
        else:
            print("  - Semantic sectioning disabled or LLM not provided. Using single section.")
            document_lines = str_to_lines(document_text)
            sections = [{
                "title": "Full Document",
                "start": 0,
                "end": len(document_lines) - 1 if document_lines else -1,
                "content": document_text
            }]
            if not document_lines: sections = []

    except Exception as e:
        print(f"Error during sectioning: {e}")
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
         return [], []

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
        raise

    return sections, chunks
