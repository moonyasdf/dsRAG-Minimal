# dsrag_minimal/examples/04_chat_basic.py
import sys
import os
import time
# Añadir ruta al directorio raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.knowledge_base import KnowledgeBase
# Importa los modelos usados para el KB (o carga desde metadata)
from core.embedding import SentenceTransformerEmbedding
from core.reranker import JinaReranker
# Importa LLMs a usar
from core.llm import OllamaLLM, OpenAILLM # Importa OpenAILLM para APIs externas
from chat import create_new_chat_thread, get_chat_thread_response, ChatResponseInput
from chat.database.sqlite_db import SQLiteChatThreadDB # Usa SQLite para historial
from openai import OpenAI # Para clientes API personalizados

# --- Configuración ---
KB_ID = "local_multilingual_kb" # KB creado en ejemplo 01
STORAGE_DIR = "~/dsrag_minimal_data"
# Modelos KB (deben coincidir con la creación)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
RERANKER_MODEL_NAME = "jina-reranker-v2-base-multilingual"
VECTOR_DIM = 1024

# --- Modelos para Chat (Selecciona UNA opción o cámbiala dinámicamente) ---

# Opción 1: Ollama Local para Chat y AutoQuery
CHAT_LLM_PROVIDER = "ollama"
CHAT_LLM_MODEL = "gemma:9b" # Modelo para chat y auto-query
AUTO_QUERY_LLM_MODEL = "gemma:9b" # Puede ser el mismo o diferente

# Opción 2: API compatible con OpenAI (LM Studio) para Chat y AutoQuery
# CHAT_LLM_PROVIDER = "lm_studio"
# LM_STUDIO_BASE_URL = "http://localhost:1234/v1" # Ajusta si es diferente
# LM_STUDIO_MODEL_NAME = "loaded-model-name-in-lm-studio" # El nombre que LM Studio usa
# LM_STUDIO_API_KEY = "lm-studio" # Clave placeholder
# CHAT_LLM_MODEL = LM_STUDIO_MODEL_NAME
# AUTO_QUERY_LLM_MODEL = LM_STUDIO_MODEL_NAME

# Opción 3: API compatible con OpenAI (OpenRouter) para Chat y AutoQuery
# CHAT_LLM_PROVIDER = "open_router"
# OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") # Usa secrets
# # Elige un modelo de OpenRouter, e.g., "mistralai/mistral-7b-instruct"
# OPENROUTER_MODEL_NAME = "mistralai/mistral-7b-instruct"
# CHAT_LLM_MODEL = OPENROUTER_MODEL_NAME
# AUTO_QUERY_LLM_MODEL = OPENROUTER_MODEL_NAME

# --- Inicializar DB de Historial de Chat ---
print("Initializing Chat History DB...")
chat_db = SQLiteChatThreadDB(storage_directory=os.path.join(os.path.expanduser(STORAGE_DIR), "chat_db"))
print("Chat History DB initialized.")

# --- Instanciar Modelos (Embedding y Reranker del KB) ---
print("Initializing KB models (Embedding, Reranker)...")
try:
    embedding = SentenceTransformerEmbedding(model_name=EMBEDDING_MODEL_NAME, dimension=VECTOR_DIM)
    reranker = JinaReranker(model_name=RERANKER_MODEL_NAME)
except Exception as e:
    print(f"Failed to initialize KB models: {e}")
    sys.exit(1)

# --- Instanciar LLMs para Chat y AutoQuery (según la opción elegida) ---
print(f"Initializing Chat/AutoQuery LLM (Provider: {CHAT_LLM_PROVIDER})...")
try:
    chat_llm_instance: LLM
    auto_query_llm_instance: LLM

    if CHAT_LLM_PROVIDER == "ollama":
        chat_llm_instance = OllamaLLM(model=CHAT_LLM_MODEL)
        auto_query_llm_instance = OllamaLLM(model=AUTO_QUERY_LLM_MODEL) # Podría ser el mismo objeto
    elif CHAT_LLM_PROVIDER == "lm_studio":
        if not LM_STUDIO_BASE_URL: raise ValueError("LM_STUDIO_BASE_URL not set")
        # Usa OpenAILLM con base_url y api_key adecuados
        chat_llm_instance = OpenAILLM(
            model=CHAT_LLM_MODEL,
            api_key=LM_STUDIO_API_KEY,
            base_url=LM_STUDIO_BASE_URL
        )
        auto_query_llm_instance = OpenAILLM(
            model=AUTO_QUERY_LLM_MODEL,
            api_key=LM_STUDIO_API_KEY,
            base_url=LM_STUDIO_BASE_URL
        )
    elif CHAT_LLM_PROVIDER == "open_router":
        if not OPENROUTER_API_KEY: raise ValueError("OPENROUTER_API_KEY environment variable not set")
        # Usa OpenAILLM con base_url y api_key adecuados
        chat_llm_instance = OpenAILLM(
            model=CHAT_LLM_MODEL, # OpenRouter usa el nombre completo del modelo
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )
        auto_query_llm_instance = OpenAILLM(
            model=AUTO_QUERY_LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )
        # OpenRouter puede necesitar headers adicionales (Referer)
        # Esto requeriría modificar OpenAILLM o crear un wrapper.
        # Por simplicidad, omitimos headers adicionales aquí.
    else:
        raise ValueError(f"Unsupported CHAT_LLM_PROVIDER: {CHAT_LLM_PROVIDER}")

    print("Chat/AutoQuery LLM initialized.")

except Exception as e:
    print(f"Failed to initialize chat/auto-query LLM: {e}")
    sys.exit(1)


# --- Cargar Knowledge Base ---
print(f"Loading Knowledge Base '{KB_ID}'...")
qdrant_persistent_path = os.path.join(os.path.expanduser(STORAGE_DIR), "qdrant_data")
try:
    kb = KnowledgeBase(
        kb_id=KB_ID,
        storage_directory=STORAGE_DIR,
        vector_dimension=VECTOR_DIM,
        embedding_model=embedding,
        reranker=reranker,
        # Pasa el LLM para AutoContext (usado al construir headers en query)
        auto_context_model=auto_query_llm_instance, # Usa el mismo de auto-query aquí
        qdrant_location=None,
        path=qdrant_persistent_path,
        exists_ok=True
    )
    print("Knowledge Base loaded.")
    if kb.vector_db.get_num_vectors() == 0:
         print(f"Warning: Knowledge Base '{KB_ID}' is empty.")

except FileNotFoundError:
     print(f"Error: Metadata for KB '{KB_ID}' not found.")
     sys.exit(1)
except Exception as e:
    print(f"Error loading Knowledge Base '{KB_ID}': {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- Crear Hilo de Chat ---
print("\nCreating new chat thread...")
thread_params = {
    "kb_ids": [KB_ID],
    "model": CHAT_LLM_MODEL, # Modelo para respuesta final (coincide con chat_llm_instance)
    "temperature": 0.1,
    "system_message": "You are an AI assistant answering questions based on the provided AGI paper, using local models.",
    "auto_query_model": AUTO_QUERY_LLM_MODEL, # Modelo para generar queries
    "target_output_length": "medium",
}
try:
    thread_id = create_new_chat_thread(thread_params, chat_db)
    print(f"Chat thread created with ID: {thread_id}")
except Exception as e:
    print(f"Error creating chat thread: {e}")
    sys.exit(1)

# --- Simular Conversación ---
questions = [
    "What are the different levels of AGI discussed in the paper?",
    "Can you elaborate on Level 5?",
    "What was the authors' methodology?"
]

active_kbs = {KB_ID: kb}

for question in questions:
    print(f"\nUser: {question}")
    print("AI: ", end="", flush=True)

    response_input = ChatResponseInput(user_input=question)
    full_response_content = ""
    final_citations = []

    try:
        start_time = time.time()
        # Llama a get_chat_thread_response INYECTANDO las instancias LLM
        stream = get_chat_thread_response(
            thread_id=thread_id,
            get_response_input=response_input,
            chat_thread_db=chat_db,
            knowledge_bases=active_kbs,
            llm_instance=chat_llm_instance, # LLM para la respuesta final
            auto_query_llm_instance=auto_query_llm_instance, # LLM para auto-query
            stream=True
        )

        # Procesa el stream (igual que antes)
        last_chunk_time = start_time
        first_chunk = True
        for partial_interaction in stream:
             # ... (código para imprimir stream igual que antes) ...
            if first_chunk:
                 first_chunk_time = time.time() - start_time
                 print(f"(First chunk: {first_chunk_time:.2f}s) ", end="", flush=True)
                 first_chunk = False

            current_content = partial_interaction.get("model_response", {}).get("content", "")
            if current_content: # Solo procesa si hay contenido
                new_content = current_content[len(full_response_content):]
                print(new_content, end="", flush=True)
                full_response_content = current_content
            # Guarda las últimas citaciones
            final_citations = partial_interaction.get("model_response", {}).get("citations", [])


        end_time = time.time()
        print(f"\n(Total response time: {end_time - start_time:.2f}s)")

        if final_citations:
            print("\n  Sources:")
            for cit in final_citations:
                page_str = f", Page: {cit.get('page_number')}" if cit.get('page_number') else ""
                kb_str = f" (KB: {cit.get('kb_id')})" if cit.get('kb_id') else ""
                print(f"  - Doc: {cit.get('doc_id')}{page_str}{kb_str}")


    except Exception as e:
        print(f"\nError getting response for '{question}': {e}")
        import traceback
        traceback.print_exc()


print("\nChat finished.")