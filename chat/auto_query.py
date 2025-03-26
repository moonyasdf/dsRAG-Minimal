# dsrag_minimal/chat/auto_query.py
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Importa LLM y función de respuesta inyectable
from ..core.llm import LLM, get_response_via_instance

# --- Carga de Prompt ---
PROMPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')

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

SYSTEM_PROMPT_TEXT = _load_prompt("chat_auto_query_system.txt")

# --- Modelos Pydantic (sin cambios) ---
class Query(BaseModel):
    """Representa una única consulta de búsqueda generada."""
    query: str = Field(..., description="La consulta de búsqueda específica.")
    knowledge_base_id: str = Field(..., description="El ID de la base de conocimiento a la que se dirige la consulta.")

class Queries(BaseModel):
    """Lista de consultas generadas."""
    queries: List[Query] = Field(..., description="Una lista de objetos Query.")

# --- Funciones de AutoQuery ---

def _get_knowledge_base_descriptions_str(kb_info: List[Dict[str, str]]) -> str:
    """Formatea la información del KB para el prompt del LLM."""
    if not kb_info:
        return "No knowledge bases available."
    descriptions = []
    for kb in kb_info:
        # Usa kb['id'] y kb['description'] o kb['title']
        desc = f"kb_id: {kb.get('id', 'N/A')}\ndescription: {kb.get('title', '')} - {kb.get('description', 'No description')}"
        descriptions.append(desc)
    return "\n\n".join(descriptions)

def _validate_queries(generated_queries: List[Query], available_kb_info: List[Dict[str, str]], max_queries: int) -> List[Dict[str, str]]:
    """Valida las queries generadas contra los KBs disponibles."""
    valid_kb_ids = {kb["id"] for kb in available_kb_info}
    validated_queries_list = []

    for query in generated_queries:
        if len(validated_queries_list) >= max_queries:
             break # Detente si ya alcanzamos el máximo

        # Verifica si el kb_id es válido
        if query.knowledge_base_id in valid_kb_ids:
            validated_queries_list.append({"query": query.query, "kb_id": query.knowledge_base_id})
        else:
            # Si el ID no es válido:
            print(f"Warning: LLM generated query for invalid kb_id '{query.knowledge_base_id}'.", end=" ")
            if len(available_kb_info) == 1:
                # Si solo hay un KB, redirige la query a ese KB
                target_kb_id = available_kb_info[0]["id"]
                print(f"Redirecting to the only available KB: '{target_kb_id}'.")
                validated_queries_list.append({"query": query.query, "kb_id": target_kb_id})
            elif len(available_kb_info) > 1:
                # Si hay múltiples KBs, replica la query para todos (puede ser ruidoso)
                print(f"Replicating query for all available KBs ({len(available_kb_info)}).")
                for kb in available_kb_info:
                    if len(validated_queries_list) < max_queries:
                        validated_queries_list.append({"query": query.query, "kb_id": kb["id"]})
                    else:
                         break # No exceder max_queries
            else:
                 print("No available KBs to redirect to. Skipping query.")
                 # No se añade la query si no hay KBs válidos

    return validated_queries_list # Ya está limitado por max_queries

def get_search_queries(
    llm_instance: LLM, # Acepta instancia LLM
    chat_messages: List[Dict[str, str]],
    kb_info: List[Dict[str, str]],
    auto_query_guidance: str = "",
    max_queries: int = 3, # Default más conservador
    # auto_query_model ya no es necesario, se usa llm_instance
) -> List[Dict[str, str]]:
    """
    Genera consultas de búsqueda basadas en el historial de chat y los KBs disponibles.

    Args:
        llm_instance: Instancia LLM para usar en la generación.
        chat_messages: Historial de chat (formato OpenAI).
        kb_info: Información sobre los KBs disponibles ({'id':.., 'title':.., 'description':..}).
        auto_query_guidance: Guía adicional para el LLM.
        max_queries: Número máximo de queries a generar.

    Returns:
        Lista de diccionarios de queries validadas ({'query':.., 'kb_id':..}).
    """
    if not kb_info:
        print("Warning: No knowledge base info provided to get_search_queries. Cannot generate queries.")
        return [] # No se pueden generar queries sin KBs
    if not SYSTEM_PROMPT_TEXT:
        print("Error: Auto-query system prompt is missing. Cannot generate queries.")
        return []

    # Formatea la descripción de los KBs para el prompt
    knowledge_base_descriptions = _get_knowledge_base_descriptions_str(kb_info)

    # Construye el prompt del sistema
    system_message = SYSTEM_PROMPT_TEXT.format(
        max_queries=max_queries,
        auto_query_guidance=auto_query_guidance,
        knowledge_base_descriptions=knowledge_base_descriptions
    )

    # Prepara los mensajes para el LLM (solo historial user/assistant)
    messages_for_llm = [{"role": "system", "content": system_message}]
    # Solo incluye el último mensaje de usuario para la generación de queries
    # o considera una ventana corta del historial si es necesario
    if chat_messages:
        messages_for_llm.append(chat_messages[-1]) # Usa solo el último mensaje

    try:
        # Llama al LLM para obtener la respuesta estructurada (Queries)
        response = get_response_via_instance(
            llm_instance=llm_instance,
            messages=messages_for_llm,
            response_model=Queries,
            # Pasa parámetros comunes (ajustar si es necesario)
            temperature=0.0, # Baja temperatura para queries deterministas
            max_tokens=600 # Suficiente para unas pocas queries
        )

        if isinstance(response, Queries) and response.queries:
            # Valida las queries generadas contra los KBs disponibles
            validated_queries = _validate_queries(response.queries, kb_info, max_queries)
            return validated_queries
        else:
            print("Warning: Auto-query LLM did not return valid Queries object or no queries were generated.")
            return []

    except Exception as e:
        print(f"Error during auto-query generation: {e}")
        return [] # Devuelve lista vacía en caso de error