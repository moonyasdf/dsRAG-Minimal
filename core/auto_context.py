# dsrag_minimal/core/auto_context.py
import os
from typing import Optional, List, Dict, Any

# Asume que LLM y get_response_via_instance están en core.llm
from .llm import LLM, get_response_via_instance
# Asume que _load_prompt está definido aquí o importado
# (Se define aquí por simplicidad, podría estar en utils)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')

def _load_prompt(filename: str) -> str:
    """Carga un prompt desde el directorio 'prompts'."""
    filepath = os.path.join(PROMPT_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}")
        return "" # Devuelve string vacío en error
    except Exception as e:
        print(f"Error loading prompt from {filepath}: {e}")
        return ""

# Carga los prompts al inicio del módulo
PROMPT_STRINGS = {
    "doc_title": _load_prompt("auto_context_doc_title.txt"),
    "doc_summary": _load_prompt("auto_context_doc_summary.txt"),
    "section_summary": _load_prompt("auto_context_section_summary.txt"),
    "language_addendum": _load_prompt("language_addendum.txt"), # Para soporte no inglés
}
# Mensaje de truncamiento (simplificado, sin cálculo de palabras)
TRUNCATION_MESSAGE = "Note: The provided text may be truncated."

def get_document_title(
    auto_context_model: LLM,
    document_text: str,
    document_title_guidance: str = "",
    language: str = "en",
    max_context_chars: int = 8000 # Límite de caracteres para el prompt
) -> str:
    """Genera un título para el documento usando el LLM."""
    if not PROMPT_STRINGS["doc_title"]: return "[Error: Title Prompt Missing]"
    if not auto_context_model: return "[AutoContext Model Not Provided]"

    # Trunca el texto si es necesario
    truncated_text = document_text[:max_context_chars]
    trunc_msg = TRUNCATION_MESSAGE if len(document_text) > max_context_chars else ""
    lang_addendum = PROMPT_STRINGS["language_addendum"] if language != 'en' else ""

    prompt = PROMPT_STRINGS["doc_title"].format(
        document_title_guidance=document_title_guidance,
        non_english_addendum=lang_addendum,
        document_text=truncated_text,
        truncation_message=trunc_msg
    )

    try:
        # Usa la función inyectable
        title = get_response_via_instance(auto_context_model, prompt=prompt).strip()
        # Limpieza simple del título (quita comillas, etc.)
        return title.strip('"\' ')
    except Exception as e:
        print(f"Error generating document title: {e}")
        return "[Error Generating Title]"

def get_document_summary(
    auto_context_model: LLM,
    document_text: str,
    document_title: str,
    document_summarization_guidance: str = "",
    language: str = "en",
    max_context_chars: int = 10000 # Límite mayor para resumen
) -> str:
    """Genera un resumen para el documento usando el LLM."""
    if not PROMPT_STRINGS["doc_summary"]: return "[Error: Summary Prompt Missing]"
    if not auto_context_model: return "[AutoContext Model Not Provided]"

    truncated_text = document_text[:max_context_chars]
    trunc_msg = TRUNCATION_MESSAGE if len(document_text) > max_context_chars else ""
    lang_addendum = PROMPT_STRINGS["language_addendum"] if language != 'en' else ""

    prompt = PROMPT_STRINGS["doc_summary"].format(
        document_summarization_guidance=document_summarization_guidance,
        non_english_addendum=lang_addendum,
        document_title=document_title,
        document_text=truncated_text,
        truncation_message=trunc_msg
    )

    try:
        summary = get_response_via_instance(auto_context_model, prompt=prompt).strip()
        # Quita prefijo común si existe
        prefix = "This document is about: "
        if summary.lower().startswith(prefix.lower()):
             summary = summary[len(prefix):].strip()
        # Similar para otros idiomas si es necesario
        # prefix_fr = "Ce document concerne : "
        # if summary.lower().startswith(prefix_fr.lower()):
        #      summary = summary[len(prefix_fr):].strip()
        return summary
    except Exception as e:
        print(f"Error generating document summary: {e}")
        return "[Error Generating Summary]"

def get_section_summary(
    auto_context_model: LLM,
    section_text: str,
    document_title: str,
    section_title: str,
    section_summarization_guidance: str = "",
    language: str = "en",
    max_context_chars: int = 8000 # Límite para texto de sección
) -> str:
    """Genera un resumen para una sección usando el LLM."""
    if not PROMPT_STRINGS["section_summary"]: return "[Error: Section Summary Prompt Missing]"
    if not auto_context_model: return "[AutoContext Model Not Provided]"

    truncated_text = section_text[:max_context_chars]
    # No se añade mensaje de truncamiento aquí, asume que la sección cabe
    lang_addendum = PROMPT_STRINGS["language_addendum"] if language != 'en' else ""

    prompt = PROMPT_STRINGS["section_summary"].format(
        section_summarization_guidance=section_summarization_guidance,
        non_english_addendum=lang_addendum,
        document_title=document_title,
        section_title=section_title,
        section_text=truncated_text
    )

    try:
        summary = get_response_via_instance(auto_context_model, prompt=prompt).strip()
        # Quita prefijo común
        prefix = "This section is about: "
        if summary.lower().startswith(prefix.lower()):
             summary = summary[len(prefix):].strip()
        # Añadir manejo para otros idiomas si es necesario
        return summary
    except Exception as e:
        print(f"Error generating section summary for '{section_title}': {e}")
        return "[Error Generating Section Summary]"

def get_chunk_header(
    document_title: str = "",
    document_summary: str = "",
    section_title: str = "",
    section_summary: str = ""
) -> str:
    """
    Construye el header contextual para un chunk.
    Se antepone al texto del chunk antes del embedding.
    """
    header_parts = []
    # Document Context (siempre incluir título si está disponible)
    doc_context = f"Document: '{document_title}'." if document_title else "Document context:"
    if document_summary:
        doc_context += f" Summary: {document_summary}"
    header_parts.append(doc_context.strip())

    # Section Context (si está disponible)
    if section_title:
        sec_context = f"Section: '{section_title}'."
        if section_summary:
            sec_context += f" Summary: {section_summary}"
        header_parts.append(sec_context.strip())

    # Une las partes con una línea vacía
    return "\n\n".join(part for part in header_parts if part) # Filtra partes vacías

def get_segment_header(
    document_title: str = "",
    document_summary: str = ""
) -> str:
    """
    Construye el header para un segmento de resultado (solo contexto de documento).
    Se antepone al contenido del segmento antes de pasarlo al LLM generador de respuesta.
    """
    if not document_title:
        return "" # No añadir header si no hay título

    header = f"Excerpt from document '{document_title}'."
    if document_summary:
        header += f" Document summary: {document_summary}"
    return header.strip()