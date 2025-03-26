# dsrag_minimal/chat/chat.py
import time
import uuid
import json # Para historial y resultados
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterator, Union # Añade Iterator y Union

# Importaciones internas refactorizadas
from .db import ChatThreadDB
from .chat_types import ChatThreadParams, MetadataFilter, ChatResponseInput
from .auto_query import get_search_queries
from .citations import format_sources_for_context, ResponseWithCitations, Citation, PartialResponseWithCitations
from ..core.llm import LLM, get_response_via_instance
from ..core.knowledge_base import KnowledgeBase # Importa KB para type hint
from ..utils.model_names import OPENAI_MODEL_NAMES, ANTHROPIC_MODEL_NAMES, GEMINI_MODEL_NAMES # Para detección de proveedor en get_response

# --- Carga de Prompts ---
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

MAIN_SYSTEM_PROMPT_TEXT = _load_prompt("chat_main_system.txt")
SHORT_OUTPUT_GUIDANCE = _load_prompt("chat_short_output.txt")
LONG_OUTPUT_GUIDANCE = _load_prompt("chat_long_output.txt")

# --- Funciones de Utilidad de Chat ---

def count_tokens(text: str) -> int:
    """Cuenta tokens usando tiktoken (aproximación para OpenAI/Anthropic)."""
    # Importación diferida
    try:
        import tiktoken
        # Usa un encoder común, gpt-4o es una buena aproximación general
        encoding = tiktoken.get_encoding("o200k_base") # Encoder para gpt-4o/gpt-3.5
        return len(encoding.encode(text, disallowed_special=()))
    except ImportError:
        # Fallback simple si tiktoken no está instalado
        return len(text) // 4 # Aproximación muy burda
    except Exception as e:
         print(f"Warning: Error using tiktoken: {e}. Falling back to character count.")
         return len(text) // 4

def limit_chat_messages(chat_messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """Limita el historial de chat basado en el conteo de tokens (de fin a inicio)."""
    total_tokens = 0
    limited_messages = []
    # Itera desde el mensaje más reciente hacia atrás
    for message in reversed(chat_messages):
        content = message.get("content", "")
        if not isinstance(content, str): # Maneja posible contenido no-string
             content = str(content)
        message_tokens = count_tokens(content)
        # Añade el mensaje si aún cabe en el límite
        if total_tokens + message_tokens <= max_tokens:
            limited_messages.insert(0, message) # Añade al principio para mantener orden
            total_tokens += message_tokens
        else:
            # Si añadir este mensaje excede el límite, para
            break
    # print(f"Debug: Limited chat history to {len(limited_messages)} messages, ~{total_tokens} tokens (max: {max_tokens})")
    return limited_messages

def _get_kb_info_for_prompt(kb_ids: List[str], knowledge_bases: Dict[str, KnowledgeBase]) -> List[Dict[str, str]]:
    """Obtiene título y descripción de los KBs especificados."""
    kb_info = []
    for kb_id in kb_ids:
        kb = knowledge_bases.get(kb_id)
        if kb and kb.kb_metadata:
            kb_info.append({
                "id": kb_id,
                "title": kb.kb_metadata.get("title", kb_id), # Usa ID si no hay título
                "description": kb.kb_metadata.get("description", "No description available.")
            })
        else:
            print(f"Warning: KnowledgeBase with id '{kb_id}' not found or has no metadata.")
    return kb_info

# --- Funciones Principales de Chat ---

def create_new_chat_thread(chat_thread_params: Dict[str, Any], chat_thread_db: ChatThreadDB) -> str:
    """
    Crea un nuevo hilo de chat en la base de datos especificada.

    Args:
        chat_thread_params: Diccionario con parámetros iniciales (ver ChatThreadParams).
        chat_thread_db: Instancia de ChatThreadDB donde se creará el hilo.

    Returns:
        El ID del hilo de chat creado.
    """
    # Asigna un ID si no existe
    thread_id = chat_thread_params.get("thread_id") or str(uuid.uuid4())
    chat_thread_params["thread_id"] = thread_id # Asegura que esté en los params

    # Valida/Completa parámetros usando Pydantic (si se define como BaseModel)
    # o manualmente
    params_to_save = ChatThreadParams(**chat_thread_params).model_dump(exclude_unset=True)

    try:
        # Llama al método de la DB para crear el hilo
        created_id = chat_thread_db.create_chat_thread(params_to_save)
        # Normalmente debería ser el mismo thread_id, pero usamos el devuelto por la DB
        print(f"Chat thread created with ID: {created_id}")
        return created_id
    except Exception as e:
        print(f"Error creating chat thread: {e}")
        raise # Relanza la excepción

def get_chat_thread_response(
    thread_id: str,
    get_response_input: ChatResponseInput,
    chat_thread_db: ChatThreadDB,
    knowledge_bases: Dict[str, KnowledgeBase], # KBs activos
    llm_instance: LLM, # Instancia LLM para generar respuesta
    auto_query_llm_instance: Optional[LLM] = None, # Opcional: LLM distinto para auto-query
    stream: bool = False,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """
    Obtiene una respuesta para un hilo de chat, usando búsqueda en KB y LLMs inyectados.

    Args:
        thread_id: ID del hilo.
        get_response_input: Objeto con user_input y overrides opcionales.
        chat_thread_db: Instancia de ChatThreadDB para historial.
        knowledge_bases: Dict {kb_id: KnowledgeBase instance} de KBs disponibles.
        llm_instance: Instancia LLM para generar la respuesta final.
        auto_query_llm_instance: Instancia LLM para generar queries (usa llm_instance si es None).
        stream: Si es True, devuelve un iterador con respuestas parciales.

    Returns:
        - Si stream=False: Diccionario representando la interacción completa.
        - Si stream=True: Iterador de diccionarios de interacción parcial.
        - Dict con 'error' si falla.
    """
    request_timestamp = datetime.now().isoformat() # ISO 8601 format

    # --- 1. Cargar Hilo y Parámetros ---
    try:
        thread_data = chat_thread_db.get_chat_thread(thread_id)
        if not thread_data:
            return {"error": f"Chat thread with ID {thread_id} not found."}
        # Parámetros base del hilo
        current_params = ChatThreadParams(**thread_data.get("params", {}))
        # Historial de interacciones
        interactions_history = thread_data.get("interactions", [])
    except Exception as e:
        return {"error": f"Failed to load chat thread {thread_id}: {e}"}

    # Aplica overrides si se proporcionan en la input
    if get_response_input.chat_thread_params:
        # Actualiza los parámetros para *esta* llamada (no guarda los overrides en DB)
        current_params = ChatThreadParams(
            **current_params.model_dump(),
             # Usa model_dump() si ChatThreadParams es Pydantic
            **get_response_input.chat_thread_params.model_dump(exclude_unset=True)
        )

    # Selecciona el LLM para auto-query
    query_llm = auto_query_llm_instance or llm_instance
    if not query_llm:
         return {"error": "No LLM instance provided for auto-query generation."}

    # --- 2. Preparar Historial y Generar Queries ---
    # Construye historial en formato OpenAI
    chat_messages_history = []
    for interaction in interactions_history:
        if interaction.get("user_input", {}).get("content"):
             chat_messages_history.append({"role": "user", "content": interaction["user_input"]["content"]})
        if interaction.get("model_response", {}).get("content"):
             chat_messages_history.append({"role": "assistant", "content": interaction["model_response"]["content"]})

    # Añade el input actual del usuario al final para generar queries
    messages_for_query_gen = chat_messages_history + [{"role": "user", "content": get_response_input.user_input}]
    # Limita el historial para la generación de queries (puede ser un límite distinto al de la respuesta final)
    limited_messages_for_query = limit_chat_messages(messages_for_query_gen, current_params.max_chat_history_tokens // 2) # Usa la mitad para queries?

    # Obtiene información de los KBs seleccionados
    selected_kb_ids = current_params.kb_ids
    kb_info = _get_kb_info_for_prompt(selected_kb_ids, knowledge_bases)

    search_queries: List[Dict[str, str]] = []
    relevant_segments: List[Dict[str, Any]] = []
    all_doc_ids_map: Dict[str, str] = {} # doc_id -> kb_id

    if kb_info: # Solo genera queries si hay KBs
        print("  - Generating search queries...")
        search_queries = get_search_queries(
            llm_instance=query_llm,
            chat_messages=limited_messages_for_query, # Usa historial limitado
            kb_info=kb_info,
            auto_query_guidance=current_params.auto_query_guidance,
            max_queries=5 # Límite de queries
        )
        print(f"  - Generated {len(search_queries)} search queries.")

        # --- 3. Ejecutar Búsquedas en KB ---
        if search_queries:
            print("  - Performing knowledge base search...")
            # Agrupa queries por KB
            queries_by_kb: Dict[str, List[str]] = {}
            for sq in search_queries:
                kb_id = sq["kb_id"]
                if kb_id not in queries_by_kb: queries_by_kb[kb_id] = []
                queries_by_kb[kb_id].append(sq["query"])

            # Ejecuta queries (podría paralelizarse si KB.query es thread-safe)
            all_kb_results = []
            for kb_id, queries in queries_by_kb.items():
                kb = knowledge_bases.get(kb_id)
                if kb:
                    try:
                        # Usa filtro si se proporcionó
                        filter_param = get_response_input.metadata_filter.model_dump() if get_response_input.metadata_filter else None
                        # Obtiene RSE params del hilo
                        rse_params_val = current_params.rse_params or "balanced"

                        results = kb.query(
                            search_queries=queries,
                            rse_params=rse_params_val,
                            metadata_filter=filter_param,
                            # return_mode="text" # O podría ser dinámico si se manejan imágenes
                        )
                        # Añade kb_id a cada resultado
                        for r in results: r["kb_id"] = kb_id
                        all_kb_results.extend(results)
                    except Exception as e:
                        print(f"Error querying KB '{kb_id}': {e}")
                else:
                    print(f"Warning: KB '{kb_id}' specified in query not found in provided knowledge_bases.")

            # Ordena los segmentos combinados por score (opcional, pero puede ser útil)
            relevant_segments = sorted(all_kb_results, key=lambda x: x.get('score', 0.0), reverse=True)
            print(f"  - Retrieved {len(relevant_segments)} relevant segments.")

    # --- 4. Preparar Contexto para LLM de Respuesta ---
    print("  - Formatting context for response generation...")
    # Usa el FileSystem del *primer* KB encontrado (asume que es el mismo para todos,
    # una limitación de esta simplificación si se usan FS distintos)
    # O mejor: Pasa el dict de KBs a format_sources_for_context si es necesario
    # Por ahora, asumimos LocalFileSystem o que get_source_text no necesita el objeto FS
    # Corrección: Pasamos el FS del primer KB relevante
    fs_to_use = None
    if relevant_segments:
         first_kb_id = relevant_segments[0].get('kb_id')
         if first_kb_id and first_kb_id in knowledge_bases:
              fs_to_use = knowledge_bases[first_kb_id].file_system

    if relevant_segments and fs_to_use:
        relevant_knowledge_str, all_doc_ids_map = format_sources_for_context(relevant_segments, fs_to_use)
    else:
        relevant_knowledge_str = "No relevant information found in knowledge bases." if kb_info else "No knowledge bases were searched."
        all_doc_ids_map = {}

    # --- 5. Construir Mensajes Finales para LLM de Respuesta ---
    # Combina historial y mensaje actual
    messages_for_llm_response = chat_messages_history + [{"role": "user", "content": get_response_input.user_input}]
    # Limita el historial final
    limited_messages_for_response = limit_chat_messages(messages_for_llm_response, current_params.max_chat_history_tokens)

    # Guía de longitud de respuesta
    response_length_guidance = ""
    if current_params.target_output_length == "short":
        response_length_guidance = SHORT_OUTPUT_GUIDANCE
    elif current_params.target_output_length == "long":
        response_length_guidance = LONG_OUTPUT_GUIDANCE

    # Formatea el prompt del sistema final
    formatted_system_message = MAIN_SYSTEM_PROMPT_TEXT.format(
        user_configurable_message=current_params.system_message,
        knowledge_base_descriptions=_get_knowledge_base_descriptions_str(kb_info), # Pasa descripciones
        relevant_knowledge_str=relevant_knowledge_str,
        response_length_guidance=response_length_guidance
    )

    final_llm_messages = [{"role": "system", "content": formatted_system_message}] + limited_messages_for_response

    # --- 6. Generar Respuesta (Streaming o No) ---
    print("  - Generating response...")
    if not llm_instance:
         return {"error": "No LLM instance provided for response generation."}

    # Define la interacción base que se irá actualizando/guardando
    base_interaction = {
        "user_input": {"content": get_response_input.user_input, "timestamp": request_timestamp},
        "search_queries": search_queries,
        "relevant_segments": relevant_segments # Guarda los segmentos originales recuperados
    }

    try:
        # Llama al LLM con el response_model ResponseWithCitations
        llm_response_or_stream = get_response_via_instance(
            llm_instance=llm_instance,
            messages=final_llm_messages,
            response_model=ResponseWithCitations, # Espera este formato
            stream=stream,
            temperature=current_params.temperature,
            max_tokens=4000 # Límite para la respuesta
        )

        # --- 7. Procesar Respuesta/Stream y Guardar ---
        if stream:
            # Devuelve un generador que procesa el stream y guarda al final
            return _process_and_save_stream(
                thread_id,
                base_interaction,
                llm_response_or_stream, # El stream del LLM
                chat_thread_db,
                all_doc_ids_map # Pasa el mapeo doc->kb
            )
        else:
            # Procesa la respuesta completa no streameada
            if not isinstance(llm_response_or_stream, ResponseWithCitations):
                # Si el LLM no devolvió el formato esperado
                print(f"Warning: LLM response was not in expected ResponseWithCitations format. Response: {llm_response_or_stream}")
                # Intenta crear una respuesta básica
                final_response_content = str(llm_response_or_stream) if llm_response_or_stream else "[LLM Error]"
                final_citations = []
            else:
                final_response_content = llm_response_or_stream.response
                # Añade kb_id a las citaciones
                final_citations = []
                for cit in llm_response_or_stream.citations:
                     cit_dict = cit.model_dump()
                     doc_id = cit_dict.get("doc_id")
                     if doc_id in all_doc_ids_map:
                          cit_dict["kb_id"] = all_doc_ids_map[doc_id]
                          final_citations.append(cit_dict)
                     else:
                          print(f"Warning: Citation for unknown doc_id '{doc_id}' discarded.")


            # Construye la interacción final
            final_interaction = {
                **base_interaction,
                "model_response": {
                    "content": final_response_content,
                    "citations": final_citations,
                    "timestamp": datetime.now().isoformat()
                }
            }

            # Guarda la interacción en la DB
            saved_interaction = chat_thread_db.add_interaction(thread_id, final_interaction)
            if not saved_interaction:
                 # Error al guardar, pero devuelve la interacción generada
                 final_interaction["error"] = "Failed to save interaction to database."
                 return final_interaction
            else:
                 # Devuelve la interacción guardada (puede tener message_id añadido)
                 return saved_interaction

    except Exception as e:
        print(f"Error during response generation or saving: {e}")
        import traceback
        traceback.print_exc()
        # Devuelve un diccionario de error
        return {
            **base_interaction,
            "model_response": {
                "content": "[An error occurred during response generation]",
                "citations": [],
                "timestamp": datetime.now().isoformat()
            },
            "error": str(e)
        }

# --- Helper para procesar Stream ---
def _process_and_save_stream(
    thread_id: str,
    base_interaction: Dict[str, Any],
    llm_stream: Iterator[Any], # Stream de PartialResponseWithCitations
    chat_thread_db: ChatThreadDB,
    all_doc_ids_map: Dict[str, str]
) -> Iterator[Dict[str, Any]]:
    """
    Procesa el stream del LLM, yield partes y guarda la interacción final.
    """
    last_partial_response = None
    final_citations_list = [] # Lista final de citaciones en formato dict

    # Itera a través del stream de respuestas parciales
    for partial_response in llm_stream:
        last_partial_response = partial_response
        current_content = ""
        current_citations = [] # Citaciones en *esta* parte del stream

        # Extrae contenido y citaciones del objeto parcial
        # (Puede ser PartialResponseWithCitations o un dict, dependiendo de instructor)
        if isinstance(partial_response, BaseModel): # Si es un objeto Pydantic (parcial)
             current_content = getattr(partial_response, 'response', "") or ""
             raw_citations = getattr(partial_response, 'citations', []) or []
             # Convierte citaciones parciales a dicts
             for cit in raw_citations:
                  if isinstance(cit, BaseModel):
                       current_citations.append(cit.model_dump(exclude_unset=True))
                  elif isinstance(cit, dict):
                       current_citations.append(cit)
        elif isinstance(partial_response, dict): # Si es un dict
            current_content = partial_response.get('response', "") or ""
            current_citations = partial_response.get('citations', []) or []
        else:
            # Maneja caso inesperado
            print(f"Warning: Unexpected type in stream: {type(partial_response)}")
            continue

        # Añade kb_id a las citaciones actuales y actualiza la lista final
        processed_citations = []
        if current_citations:
            temp_final_citations = []
            for cit_dict in current_citations:
                 doc_id = cit_dict.get("doc_id")
                 # Solo procesa citaciones completas (con doc_id y cited_text)
                 if doc_id and cit_dict.get("cited_text"):
                      if doc_id in all_doc_ids_map:
                           cit_dict_copy = cit_dict.copy() # Evita modificar el dict original
                           cit_dict_copy["kb_id"] = all_doc_ids_map[doc_id]
                           processed_citations.append(cit_dict_copy)
                           temp_final_citations.append(cit_dict_copy) # Añade a la lista final temporal
                      else:
                           print(f"Warning: Stream citation for unknown doc_id '{doc_id}' discarded.")
            # Actualiza la lista final solo si tenemos nuevas citaciones completas
            if temp_final_citations:
                 final_citations_list = temp_final_citations


        # Construye el estado actual de la interacción para yield
        yield {
            **base_interaction,
            "model_response": {
                "content": current_content,
                "citations": processed_citations, # Citaciones *actuales* con kb_id
                "timestamp": datetime.now().isoformat() # Timestamp actual
            }
        }

    # --- Después del Bucle (Stream finalizado) ---
    # Guarda la interacción completa final en la DB
    if last_partial_response:
        final_content = ""
        # Extrae contenido final
        if isinstance(last_partial_response, BaseModel):
             final_content = getattr(last_partial_response, 'response', "") or ""
        elif isinstance(last_partial_response, dict):
             final_content = last_partial_response.get('response', "") or ""

        final_interaction_to_save = {
            **base_interaction,
            "model_response": {
                "content": final_content,
                "citations": final_citations_list, # Usa la lista final acumulada
                "timestamp": datetime.now().isoformat() # Timestamp final
            }
        }
        # Intenta guardar en DB
        saved_interaction = chat_thread_db.add_interaction(thread_id, final_interaction_to_save)
        if not saved_interaction:
             print(f"Error: Failed to save final interaction for thread {thread_id} after streaming.")
             # Podría hacer yield de un mensaje de error final
             # yield { "error": "Failed to save interaction"}
    else:
         print(f"Warning: Stream finished without receiving any response parts for thread {thread_id}.")
         # Considera guardar una interacción con respuesta vacía o de error
         error_interaction = {
             **base_interaction,
             "model_response": { "content": "[No response generated]", "citations": [], "timestamp": datetime.now().isoformat() }
         }
         chat_thread_db.add_interaction(thread_id, error_interaction)