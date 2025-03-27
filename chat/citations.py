# dsrag_minimal/chat/citations.py
# (Sin cambios funcionales respecto al original, solo verificación de tipos/imports)
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import instructor
import os
import json

# Usa ruta relativa correcta
from ..dsparse.file_parsing.file_system import FileSystem

class Citation(BaseModel):
    """Representa una única citación a una fuente."""
    doc_id: str = Field(..., description="ID del documento fuente.")
    page_number: Optional[int] = Field(None, description="Número de página (si está disponible).")
    cited_text: str = Field(..., description="Texto exacto citado.")
    # Añadido kb_id para saber de qué KB viene
    kb_id: Optional[str] = Field(None, description="ID de la Knowledge Base que contiene el documento.")

class ResponseWithCitations(BaseModel):
    """Modelo para la respuesta del LLM que incluye citaciones."""
    response: str = Field(..., description="La respuesta generada al usuario.")
    citations: List[Citation] = Field(..., description="Lista de citaciones usadas.")

# Modelo parcial para streaming (usando instructor)
PartialResponseWithCitations = instructor.Partial[ResponseWithCitations]

def format_page_content(page_number: int, content: str) -> str:
    """Formatea contenido de una página con etiquetas de número de página."""
    # Formato simple y claro para el LLM
    return f"<page number=\"{page_number}\">\n{content}\n</page>"

def get_source_text(kb_id: str, doc_id: str, page_start: Optional[int], page_end: Optional[int], file_system: FileSystem) -> Optional[str]:
    """
    Obtiene el texto fuente para un rango de páginas, formateado con etiquetas.
    Devuelve None si el contenido de página no está disponible.
    """
    if page_start is None or page_end is None:
        return None # No se pueden obtener páginas sin números

    try:
        # Carga el contenido del rango de páginas usando el file_system
        page_contents = file_system.load_page_content_range(kb_id, doc_id, page_start, page_end)

        if not page_contents:
            # print(f"Debug: No page contents found for {kb_id}/{doc_id} pages {page_start}-{page_end}")
            return None # No se encontró contenido

        # Construye el texto fuente con etiquetas
        source_text = f"<document id=\"{doc_id}\">\n"
        for i, content in enumerate(page_contents):
            page_number = page_start + i
            source_text += format_page_content(page_number, content) + "\n"
        source_text += f"</document>" # Cierra la etiqueta del documento

        return source_text.strip()

    except Exception as e:
        print(f"Error loading page content range for {kb_id}/{doc_id} ({page_start}-{page_end}): {e}")
        return None # Devuelve None en caso de error

def format_sources_for_context(search_results: List[Dict[str, Any]], file_system: FileSystem) -> Tuple[str, Dict[str, str]]:
    """
    Formatea los resultados de búsqueda en un string de contexto para el LLM.

    Args:
        search_results: Lista de segmentos relevantes (cada uno con kb_id, doc_id, etc.).
        file_system: Instancia del FileSystem para cargar contenido de página si es necesario.

    Returns:
        Una tupla: (context_string, all_doc_ids_map), donde all_doc_ids_map mapea doc_id a kb_id.
    """
    context_parts = []
    all_doc_ids_map: Dict[str, str] = {} # Mapea doc_id -> kb_id

    # Agrupa resultados por documento para intentar obtener rangos de páginas contiguos
    docs_to_process: Dict[Tuple[str, str], List[Dict[str, Any]]] = {} # (kb_id, doc_id) -> [results]
    for result in search_results:
        kb_id = result.get("kb_id")
        doc_id = result.get("doc_id")
        if kb_id and doc_id:
             key = (kb_id, doc_id)
             if key not in docs_to_process:
                  docs_to_process[key] = []
             docs_to_process[key].append(result)
             all_doc_ids_map[doc_id] = kb_id # Guarda el mapeo

    # Procesa cada documento
    for (kb_id, doc_id), results in docs_to_process.items():
        # Intenta obtener texto basado en páginas si están disponibles
        page_start = results[0].get("segment_page_start")
        page_end = results[-1].get("segment_page_end") # Asume que los resultados están ordenados por chunk

        full_page_source_text = None
        if page_start is not None and page_end is not None:
            # Intenta cargar el contenido de página formateado
            full_page_source_text = get_source_text(kb_id, doc_id, page_start, page_end, file_system)

        if full_page_source_text:
            # Usa el contenido de página si se encontró
            context_parts.append(full_page_source_text)
        else:
            # Fallback a usar el contenido directo del segmento (menos preciso para citaciones de página)
            doc_context = f"<document id=\"{doc_id}\">\n"
            # Concatena el contenido de los segmentos de este documento
            doc_content = "\n...\n".join([r.get("content", "") for r in results if r.get("content")])
            doc_context += doc_content + "\n</document>"
            context_parts.append(doc_context.strip())

    # Une todas las partes del contexto
    final_context = "\n\n".join(context_parts).strip()
    return final_context, all_doc_ids_map

# (convert_elements_to_page_content se puede mantener si aún se usa en add_document,
# aunque la versión mínima de add_document no la llama explícitamente.
# La dejamos comentada por si se reactiva el parseo con page numbers)
# def convert_elements_to_page_content(...)
