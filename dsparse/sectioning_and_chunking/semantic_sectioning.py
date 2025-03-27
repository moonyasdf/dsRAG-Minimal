# dsrag_minimal/dsparse/sectioning_and_chunking/semantic_sectioning.py
# (Adaptado para aceptar instancia LLM y cargar prompt)

import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import instructor
import warnings

# Importa tipos y LLM
from core.llm import LLM, get_response_via_instance
from dsparse.models.types import Line, Section

# --- Carga de Prompt ---
PROMPT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts') # Ajusta ruta

def _load_prompt(filename: str) -> str:
    """Carga un prompt desde el directorio 'prompts'."""
    filepath = os.path.join(PROMPT_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}")
        return ""
    except Exception as e:
        print(f"Error loading prompt from {filepath}: {e}")
        return ""

SYSTEM_PROMPT_TEXT = _load_prompt("semantic_sectioning_system.txt")
LANGUAGE_ADDENDUM_TEXT = _load_prompt("language_addendum.txt")

# --- Modelos Pydantic (sin cambios) ---
class DocumentSection(BaseModel):
    title: str = Field(description="main topic of this section of the document (very descriptive)")
    start_index: int = Field(description="line number where the section begins (inclusive)")

class StructuredDocument(BaseModel):
    sections: List[DocumentSection] = Field(description="an ordered list of sections of the document")

# --- Funciones de Sectioning ---

def get_document_with_lines(document_lines: List[Line], start_line: int, max_characters: int) -> tuple[str, int]:
    """Prepara el texto del documento con números de línea para el LLM."""
    # (Implementación sin cambios respecto a la original)
    # ... (copiar de la versión original) ...
    document_with_line_numbers = ""
    character_count = 0
    end_line = start_line # Inicializa end_line
    for i in range(start_line, len(document_lines)):
        line_content = document_lines[i]["content"]
        # Añade número de línea y contenido
        line_str = f"[{i}] {line_content}\n"
        # Verifica si añadir la línea excede el límite de caracteres
        if character_count + len(line_str) > max_characters and document_with_line_numbers:
            # Si excede y ya tenemos contenido, paramos *antes* de añadir esta línea
            end_line = i - 1
            break
        document_with_line_numbers += line_str
        character_count += len(line_str)
        end_line = i # Actualiza end_line a la última línea añadida
        # Si es la última línea del documento, termina el bucle
        if i == len(document_lines) - 1:
            break
    # Asegura que end_line no sea menor que start_line si solo se procesó una línea o ninguna
    if end_line < start_line and len(document_lines) > start_line:
         end_line = start_line

    return document_with_line_numbers.strip(), end_line # strip() para quitar último newline

def get_structured_document_with_llm(
    document_with_line_numbers: str,
    start_line: int,
    llm_instance: LLM, # Acepta instancia LLM
    language: str
) -> Optional[StructuredDocument]:
    """Llama al LLM inyectado para obtener la estructura del documento."""
    if not SYSTEM_PROMPT_TEXT:
        print("Error: Semantic sectioning system prompt is missing.")
        return None

    # Formatea el prompt del sistema
    formatted_system_prompt = SYSTEM_PROMPT_TEXT.format(start_line=start_line)
    if language != "en" and LANGUAGE_ADDENDUM_TEXT:
        formatted_system_prompt += "\n\n" + LANGUAGE_ADDENDUM_TEXT

    # Prepara los mensajes para el LLM
    messages = [
        {"role": "system", "content": formatted_system_prompt},
        {"role": "user", "content": document_with_line_numbers},
    ]

    try:
        # Llama al LLM usando la función inyectable y el response_model
        response = get_response_via_instance(
            llm_instance=llm_instance,
            messages=messages,
            response_model=StructuredDocument, # Pydantic model para Instructor
            # Pasa parámetros comunes (ajustar si es necesario)
            temperature=0.0,
            max_tokens=4000
        )
        if isinstance(response, StructuredDocument):
             return response
        else:
             # Si la respuesta no es del tipo esperado (podría pasar si el LLM falla en estructurar)
             print(f"Warning: LLM did not return the expected StructuredDocument format. Response: {response}")
             return None
    except Exception as e:
        print(f"Error calling LLM for semantic sectioning: {e}")
        # Considera reintentos o manejo de errores más específico
        return None

def validate_and_fix_sections(sections: List[DocumentSection], document_length: int, min_start_line: int = 0) -> List[DocumentSection]:
    """Valida y corrige índices de sección (0-based)."""
    if not sections:
        # Si no hay secciones, crea una por defecto cubriendo desde min_start_line
        if document_length > min_start_line:
             return [DocumentSection(title="Full Content", start_index=min_start_line)]
        else:
             return []

    # 1. Elimina duplicados y secciones fuera de rango inicial
    seen_indices = set()
    valid_sections = []
    for s in sections:
        # Ignora secciones que empiezan antes del mínimo o igual/después del final del documento
        if s.start_index not in seen_indices and min_start_line <= s.start_index < document_length:
            seen_indices.add(s.start_index)
            valid_sections.append(s)
        else:
             print(f"Warning: Invalid or duplicate section start_index {s.start_index} for '{s.title}'. Skipping.")

    if not valid_sections:
         # Si todas fueron inválidas, crea una por defecto
         if document_length > min_start_line:
              return [DocumentSection(title="Full Content", start_index=min_start_line)]
         else:
              return []

    # 2. Ordena por índice de inicio
    valid_sections.sort(key=lambda x: x.start_index)

    # 3. Asegura que la primera sección empiece en min_start_line
    if valid_sections[0].start_index > min_start_line:
        print(f"Warning: First section started at {valid_sections[0].start_index}, adjusting to {min_start_line}.")
        # Opción 1: Ajusta la primera sección existente
        # valid_sections[0].start_index = min_start_line
        # Opción 2: Añade una sección inicial genérica (más seguro si el LLM omitió el inicio)
        valid_sections.insert(0, DocumentSection(title="Initial Content", start_index=min_start_line))
        # Re-elimina duplicados si la inserción creó uno
        if len(valid_sections) > 1 and valid_sections[0].start_index == valid_sections[1].start_index:
            valid_sections.pop(1)


    # 4. Elimina secciones que empiezan igual que la anterior (redundantes después de ordenar)
    final_sections = []
    last_start = -1
    for s in valid_sections:
        if s.start_index > last_start:
            final_sections.append(s)
            last_start = s.start_index
        else:
            # Esto no debería ocurrir después de la limpieza inicial y ordenación, pero es una salvaguarda
            print(f"Warning: Skipping section '{s.title}' with redundant start_index {s.start_index}.")


    return final_sections


def get_sections_text(sections_info: List[DocumentSection], document_lines: List[Line]) -> List[Section]:
    """Convierte DocumentSection (solo inicio) a Section (con contenido y fin)."""
    # (Implementación sin cambios respecto a la original)
    # ... (copiar de la versión original) ...
    section_dicts = []
    doc_length = len(document_lines)

    if not sections_info: return [] # Maneja lista vacía

    for i, s in enumerate(sections_info):
        start_index = s.start_index
        # Calcula el índice final
        if i == len(sections_info) - 1:
            # La última sección va hasta el final del documento
            end_index = doc_length - 1
        else:
            # Termina justo antes de que empiece la siguiente sección
            end_index = sections_info[i+1].start_index - 1

        # Valida los índices contra la longitud del documento
        start_index = max(0, min(start_index, doc_length - 1))
        end_index = max(start_index, min(end_index, doc_length - 1)) # Asegura end >= start

        # Extrae el contenido de las líneas correspondientes
        if start_index <= end_index: # Solo si el rango es válido
             try:
                 # Obtiene el contenido de las líneas en el rango
                 content_lines = [document_lines[j]["content"] for j in range(start_index, end_index + 1)]
                 section_content = "\n".join(content_lines)

                 section_dicts.append(Section(
                     title=s.title,
                     content=section_content,
                     start=start_index,
                     end=end_index
                 ))
             except IndexError:
                  print(f"Error: Index out of range when getting content for section '{s.title}' (lines {start_index}-{end_index}, doc length {doc_length})")
             except Exception as e:
                  print(f"Error processing content for section '{s.title}': {e}")
        else:
             # Si start > end después de la validación (debería ser raro), omite la sección
             print(f"Warning: Skipping section '{s.title}' due to invalid range after validation ({start_index}-{end_index}).")

    return section_dicts


def get_sections(
    document_lines: List[Line],
    llm_instance: LLM, # Acepta instancia LLM
    language: str,
    max_iterations: int = 10, # Límite de seguridad
    max_characters: int = 20000 # Caracteres por llamada LLM
    ) -> List[Section]:
    """Obtiene secciones semánticas iterativamente."""
    all_sections_info: List[DocumentSection] = []
    start_line = 0
    doc_len = len(document_lines)
    iterations = 0

    while start_line < doc_len and iterations < max_iterations:
        iterations += 1
        print(f"  - Sectioning iteration {iterations}, starting from line {start_line}...")

        # 1. Prepara texto con números de línea para esta iteración
        doc_excerpt, end_line_processed = get_document_with_lines(
            document_lines, start_line, max_characters
        )

        if not doc_excerpt: # Si no hay más texto
            break

        # 2. Llama al LLM para obtener secciones del fragmento
        structured_doc = get_structured_document_with_llm(
            doc_excerpt, start_line, llm_instance, language
        )

        if not structured_doc or not structured_doc.sections:
            print(f"Warning: No sections returned by LLM for lines {start_line}-{end_line_processed}. Moving to next block.")
            # Avanza al final del bloque procesado para evitar bucle infinito
            start_line = end_line_processed + 1
            continue # Salta al siguiente ciclo

        # 3. Valida y corrige las secciones *relativas a esta iteración*
        # El document_length aquí es hasta dónde hemos procesado + 1
        # min_start_line es donde empezamos esta iteración
        current_block_doc_length = end_line_processed + 1
        validated_batch_sections = validate_and_fix_sections(
            structured_doc.sections, current_block_doc_length, min_start_line=start_line
        )

        if not validated_batch_sections:
             print(f"Warning: No valid sections after validation for lines {start_line}-{end_line_processed}. Moving to next block.")
             start_line = end_line_processed + 1
             continue

        # 4. Decide cuántas secciones añadir y dónde empezar la siguiente iteración
        if end_line_processed >= doc_len - 1:
            # Si procesamos hasta el final del documento, añadimos todas las secciones válidas
            all_sections_info.extend(validated_batch_sections)
            start_line = doc_len # Termina el bucle
        elif len(validated_batch_sections) > 1:
            # Si hay múltiples secciones, añade todas menos la última.
            # La siguiente iteración empezará desde el inicio de esa última sección
            # para asegurar que el LLM tenga contexto superpuesto.
            sections_to_add = validated_batch_sections[:-1]
            next_start_section = validated_batch_sections[-1]

            all_sections_info.extend(sections_to_add)
            start_line = next_start_section.start_index
            # Asegura que start_line avance para evitar bucles infinitos si el LLM se atasca
            if start_line <= (sections_to_add[-1].start_index if sections_to_add else -1):
                 print(f"Warning: Potential stall in sectioning at line {start_line}. Forcing advance.")
                 start_line = end_line_processed + 1

        else: # len(validated_batch_sections) == 1
            # Si solo hay una sección en este bloque grande, puede ser demasiado larga.
            # Añádela por ahora, pero empieza la siguiente iteración *después* de este bloque.
            # Esto es menos ideal para superposición, pero evita procesar lo mismo.
            all_sections_info.extend(validated_batch_sections)
            start_line = end_line_processed + 1

        # Limpia duplicados que puedan surgir por la superposición
        temp_sections_info = []
        seen_starts = set()
        for s in all_sections_info:
             if s.start_index not in seen_starts:
                  temp_sections_info.append(s)
                  seen_starts.add(s.start_index)
        all_sections_info = temp_sections_info
        # Reordena por si acaso (aunque la lógica debería mantener el orden)
        all_sections_info.sort(key=lambda x: x.start_index)


    if iterations == max_iterations:
        print(f"Warning: Reached max iterations ({max_iterations}) for semantic sectioning.")

    # 5. Convierte la lista final de DocumentSection a Section (con contenido y fin)
    final_sections = get_sections_text(all_sections_info, document_lines)

    return final_sections

# --- Funciones de conversión a Líneas (simplificadas) ---

def split_long_line(line: str, max_line_length: int = 200) -> List[str]:
    """Divide líneas largas."""
    # (Implementación sin cambios)
    # ... (copiar de la versión original) ...
    if len(line) <= max_line_length:
        return [line]

    words = line.split(' ') # Divide por espacio
    lines = []
    current_line = ""
    for word in words:
        # Si la línea actual está vacía, añade la palabra directamente
        if not current_line:
            # Si la palabra en sí es más larga que el máximo, la añade sola (caso extremo)
            if len(word) > max_line_length:
                 # Podríamos dividir la palabra, pero por simplicidad la añadimos tal cual
                 lines.append(word)
                 current_line = "" # Resetea para la siguiente palabra
            else:
                 current_line = word
        # Si añadir la palabra (y un espacio) no excede el límite
        elif len(current_line) + 1 + len(word) <= max_line_length:
            current_line += " " + word
        # Si excede el límite
        else:
            lines.append(current_line) # Guarda la línea actual
            # Empieza la nueva línea con la palabra actual
            if len(word) <= max_line_length:
                 current_line = word
            else: # Si la palabra es demasiado larga
                 lines.append(word) # Añádela como línea separada
                 current_line = "" # Resetea
    # Añade la última línea si no está vacía
    if current_line:
        lines.append(current_line)
    return lines

def str_to_lines(document: str, max_line_length: int = 200) -> List[Line]:
    """Convierte un string a una lista de objetos Line."""
    document_lines: List[Line] = []
    original_lines = document.splitlines() # Usa splitlines para manejar diferentes finales de línea
    for line in original_lines:
        # Divide líneas largas si es necesario
        split_lines_content = split_long_line(line, max_line_length)
        for content in split_lines_content:
            document_lines.append({
                "content": content,
                "element_type": "NarrativeText", # Tipo por defecto para texto plano
                "page_number": None,
                "is_visual": False,
            })
    return document_lines

def pages_to_lines(pages: List[str], max_line_length: int = 200) -> List[Line]:
    """Convierte una lista de strings (páginas) a objetos Line."""
    document_lines: List[Line] = []
    for i, page_text in enumerate(pages):
        page_num = i + 1 # Páginas son 1-based
        original_lines = page_text.splitlines()
        for line in original_lines:
            split_lines_content = split_long_line(line, max_line_length)
            for content in split_lines_content:
                document_lines.append({
                    "content": content,
                    "element_type": "NarrativeText",
                    "page_number": page_num,
                    "is_visual": False,
                })
    return document_lines

# --- Funciones de Alto Nivel ---

def get_sections_from_str(
    document: str,
    semantic_sectioning_config: Dict[str, Any],
    max_characters: int = 20000,
    ) -> Tuple[List[Section], List[Line]]:
    """Obtiene secciones semánticas a partir de un string."""
    llm_instance = semantic_sectioning_config.get('llm')
    language = semantic_sectioning_config.get('language', 'en')
    if not llm_instance:
         raise ValueError("LLM instance ('llm') is required in semantic_sectioning_config.")

    document_lines = str_to_lines(document)
    if not document_lines: return [], []

    max_iterations = max(1, (len(document) // max_characters) * 2 + 1) # Estima iteraciones

    sections = get_sections(document_lines, llm_instance, language, max_iterations, max_characters)
    return sections, document_lines

def get_sections_from_pages(
    pages: List[str],
    semantic_sectioning_config: Dict[str, Any],
    max_characters: int = 20000,
    ) -> Tuple[List[Section], List[Line]]:
    """Obtiene secciones semánticas a partir de una lista de páginas."""
    llm_instance = semantic_sectioning_config.get('llm')
    language = semantic_sectioning_config.get('language', 'en')
    if not llm_instance:
         raise ValueError("LLM instance ('llm') is required in semantic_sectioning_config.")

    document_lines = pages_to_lines(pages)
    if not document_lines: return [], []

    # Calcula longitud total aproximada para max_iterations
    total_chars = sum(len(line['content']) for line in document_lines)
    max_iterations = max(1, (total_chars // max_characters) * 2 + 1)

    sections = get_sections(document_lines, llm_instance, language, max_iterations, max_characters)
    return sections, document_lines

# Nota: get_sections_from_elements se elimina ya que VLM no se usa.
