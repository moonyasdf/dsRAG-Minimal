# dsrag_minimal/examples/04_chat_basic.py
import sys
import os
import time
# Añadir ruta al directorio raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.knowledge_base import KnowledgeBase
from core.embedding import OllamaEmbedding # O cambia a OpenAIEmbedding si usas API
from core.llm import OllamaLLM # O cambia a OpenAILLM, AnthropicLLM
from chat import create_new_chat_thread, get_chat_thread_response, ChatResponseInput
from chat.database.sqlite_db import SQLiteChatThreadDB # Usa SQLite para historial

# --- Configuración ---
KB_ID = "local_agi_kb" # Debe coincidir con el KB creado en el ejemplo 01 o 02
STORAGE_DIR = "~/dsrag_minimal_data" # Donde se guardó el KB y donde se guardará el historial
CHAT_LLM_MODEL = "llama3" # Modelo para generar respuestas de chat
# Asegúrate que la dimensión coincide con el KB que cargas
VECTOR_DIM = 768 # Para nomic-embed-text

# --- Inicializar DB de Historial de Chat ---
print("Initializing Chat History DB...")
# Usa un archivo distinto o subdirectorio para el historial de chat
chat_db = SQLiteChatThreadDB(storage_directory=os.path.join(os.path.expanduser(STORAGE_DIR), "chat_db"))
print("Chat History DB initialized.")

# --- Instanciar/Cargar Modelos ---
print("Initializing models for chat...")
try:
    # Necesitas el embedding model para la función de búsqueda dentro de get_chat_thread_response
    # Usa el mismo que al crear el KB
    embedding = OllamaEmbedding(model="nomic-embed-text", dimension=VECTOR_DIM)
    # LLM para generar respuestas y auto-query
    llm = OllamaLLM(model=CHAT_LLM_MODEL)
    # O usa modelos API si prefieres:
    # from core.embedding import OpenAIEmbedding
    # from core.llm import OpenAILLM
    # embedding = OpenAIEmbedding()
    # llm = OpenAILLM()
    # VECTOR_DIM = embedding.dimension # Obtén dimensión del modelo API
except Exception as e:
    print(f"Failed to initialize models: {e}")
    sys.exit(1)

# --- Cargar Knowledge Base ---
print(f"Loading Knowledge Base '{KB_ID}'...")
try:
    # Qdrant necesita el path si es persistente local
    qdrant_path = os.path.join(os.path.expanduser(STORAGE_DIR), "qdrant_data")
    kb = KnowledgeBase(
        kb_id=KB_ID,
        storage_directory=STORAGE_DIR,
        vector_dimension=VECTOR_DIM, # Pasa la dimensión
        embedding_model=embedding,   # Pasa el modelo embedding (necesario para query)
        auto_context_model=llm,      # Pasa LLM (necesario para headers en query)
        # Reranker default es NoReranker
        qdrant_location=None,        # Indica path local
        path=qdrant_path,
        exists_ok=True               # Debe existir
    )
    print("Knowledge Base loaded.")
    # Verifica que el KB no esté vacío
    if kb.vector_db.get_num_vectors() == 0:
         print(f"Warning: Knowledge Base '{KB_ID}' is empty. Queries will not find results.")

except FileNotFoundError:
     print(f"Error: Metadata for KB '{KB_ID}' not found in {STORAGE_DIR}. Did you run example 01 or 02 first?")
     sys.exit(1)
except Exception as e:
    print(f"Error loading Knowledge Base '{KB_ID}': {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# --- Crear Hilo de Chat ---
print("\nCreating new chat thread...")
thread_params = {
    "kb_ids": [KB_ID], # Usa el KB cargado
    "model": CHAT_LLM_MODEL, # Modelo para la respuesta final
    "temperature": 0.1,
    "system_message": "You are an AI assistant answering questions based on the provided AGI paper.",
    "auto_query_model": CHAT_LLM_MODEL, # Modelo para generar queries
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

# Diccionario para pasar los KBs activos a la función de respuesta
active_kbs = {KB_ID: kb}

for question in questions:
    print(f"\nUser: {question}")
    print("AI: ", end="", flush=True) # Imprime prefijo para respuesta

    response_input = ChatResponseInput(user_input=question)
    full_response_content = ""
    final_citations = []

    try:
        start_time = time.time()
        # Llama a get_chat_thread_response con los LLMs inyectados
        # Puedes usar streaming=True para respuesta incremental
        stream = get_chat_thread_response(
            thread_id=thread_id,
            get_response_input=response_input,
            chat_thread_db=chat_db,
            knowledge_bases=active_kbs,
            llm_instance=llm,             # LLM para respuesta
            auto_query_llm_instance=llm, # LLM para auto-query (puede ser diferente)
            stream=True                   # Habilita streaming
        )

        # Procesa el stream
        last_chunk_time = start_time
        first_chunk = True
        for partial_interaction in stream:
            if first_chunk:
                 first_chunk_time = time.time() - start_time
                 print(f"(First chunk: {first_chunk_time:.2f}s) ", end="", flush=True)
                 first_chunk = False

            current_content = partial_interaction.get("model_response", {}).get("content", "")
            new_content = current_content[len(full_response_content):]
            print(new_content, end="", flush=True)
            full_response_content = current_content
            # Guarda las últimas citaciones (pueden cambiar durante el stream)
            final_citations = partial_interaction.get("model_response", {}).get("citations", [])
            # Opcional: Muestra progreso de tiempo entre chunks
            # current_time = time.time()
            # print(f" [Chunk delta: {current_time - last_chunk_time:.2f}s]", end="")
            # last_chunk_time = current_time

        end_time = time.time()
        print(f"\n(Total response time: {end_time - start_time:.2f}s)") # Nueva línea al final

        # Imprime citaciones finales
        if final_citations:
            print("\n  Sources:")
            for cit in final_citations:
                page_str = f", Page: {cit.get('page_number')}" if cit.get('page_number') else ""
                print(f"  - Doc: {cit.get('doc_id')}{page_str}")
                # print(f"    Text: {cit.get('cited_text', '')[:80]}...") # Opcional: preview de texto citado

    except Exception as e:
        print(f"\nError getting response for '{question}': {e}")
        import traceback
        traceback.print_exc()
        # Continúa con la siguiente pregunta si falla una

print("\nChat finished.")