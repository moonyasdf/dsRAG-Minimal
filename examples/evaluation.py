# dsrag_minimal/examples/evaluation.py
import sys
import os
import time
import numpy as np

# Añadir ruta al directorio raíz del proyecto si es necesario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.knowledge_base import KnowledgeBase
from core.embedding import OllamaEmbedding
from core.llm import OllamaLLM
from core.reranker import NoReranker
from database.vector.qdrant_db import QdrantVectorDB # Asegúrate que está accesible
from database.chunk.sqlite_db import SQLiteDB
from dsparse.file_parsing.file_system import LocalFileSystem

# --- Configuración ---
KB_ID = "eval_kb_minimal"
STORAGE_DIR = "~/dsrag_minimal_eval_data"
DOC_PATH = "../tests/data/levels_of_agi.pdf" # Ajusta la ruta a tu documento de prueba
EVAL_QUERIES = [
    "What are the levels of AGI?",
    "What is the highest level of AGI?",
    "Methodology for determining levels of AGI"
]

# --- Modelos Locales (Ollama) ---
try:
    # Define la dimensión correcta para tu modelo local
    # nomic-embed-text -> 768, all-minilm -> 384, llama3 -> 4096 etc.
    embedding_dim = 768
    embedding_model = OllamaEmbedding(model="nomic-embed-text", dimension=embedding_dim)
    llm_model = OllamaLLM(model="llama3") # Usado para AutoContext y Semantic Sectioning
    reranker_model = NoReranker()
except Exception as e:
    print(f"Error initializing local models (Ollama): {e}")
    print("Please ensure Ollama is running and the models are available.")
    sys.exit(1)

# --- Inicialización del KB ---
print(f"--- Initializing Knowledge Base: {KB_ID} ---")
# Limpia datos previos si existen
eval_storage_path = os.path.expanduser(STORAGE_DIR)
if os.path.exists(eval_storage_path):
     import shutil
     print(f"Removing previous data at {eval_storage_path}...")
     # Intenta eliminar la colección Qdrant primero si existe localmente
     try:
          qdrant_db_temp = QdrantVectorDB(kb_id=KB_ID, vector_dimension=embedding_dim, location=os.path.join(eval_storage_path, "qdrant_data"))
          qdrant_db_temp.delete()
     except Exception:
          pass # Ignora si no existe o falla
     shutil.rmtree(eval_storage_path, ignore_errors=True)
     time.sleep(1) # Pausa breve

os.makedirs(eval_storage_path, exist_ok=True)

# Configura Qdrant para almacenamiento local persistente
qdrant_db = QdrantVectorDB(
    kb_id=KB_ID,
    vector_dimension=embedding_dim,
    location=None, # Indica uso local con path
    path=os.path.join(eval_storage_path, "qdrant_data"), # Path para persistencia
    # O usa :memory: para pruebas rápidas no persistentes
    # location=":memory:"
)

# Configura FileSystem local
local_fs = LocalFileSystem(base_path=os.path.join(eval_storage_path, "kb_files"))

kb = KnowledgeBase(
    kb_id=KB_ID,
    embedding_model=embedding_model,
    auto_context_model=llm_model,
    reranker=reranker_model,
    storage_directory=eval_storage_path, # Directorio para SQLite y metadatos
    vector_dimension=embedding_dim,      # Pasa la dimensión explícitamente
    vector_db=qdrant_db,                 # Pasa la instancia Qdrant configurada
    file_system=local_fs,                # Pasa la instancia LocalFileSystem
    exists_ok=False                      # Forzar creación nueva
)
print("Knowledge Base initialized.")

# --- Evaluación de add_document ---
print(f"\n--- Evaluating add_document for {DOC_PATH} ---")
start_add = time.time()
try:
    kb.add_document(
        doc_id="levels_of_agi",
        file_path=DOC_PATH,
        document_title="Levels of AGI Paper",
        # Config específica si es necesaria (ej: deshabilitar sectioning para prueba rápida)
        # semantic_sectioning_config={"use_semantic_sectioning": False},
        # auto_context_config={"get_document_summary": False}
    )
    end_add = time.time()
    add_duration = end_add - start_add
    print(f"add_document completed in {add_duration:.2f} seconds.")
    num_vectors = kb.vector_db.get_num_vectors()
    print(f"Number of vectors added: {num_vectors}")
    if num_vectors == 0:
         print("Error: No vectors were added.")
         sys.exit(1)
except Exception as e:
    print(f"Error during add_document: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Pausa breve para asegurar que Qdrant indexe (especialmente si wait=False en upsert)
print("Waiting for indexing...")
time.sleep(5)

# --- Evaluación de query ---
print(f"\n--- Evaluating query ---")
query_durations = []
all_results = {}

for i, query_text in enumerate(EVAL_QUERIES):
    print(f"\nRunning query {i+1}: '{query_text}'")
    start_query = time.time()
    try:
        results = kb.query(
            search_queries=[query_text],
            rse_params="balanced", # o "precision" / dict personalizado
            return_mode="text"
        )
        end_query = time.time()
        query_duration = end_query - start_query
        query_durations.append(query_duration)
        all_results[query_text] = results
        print(f"Query {i+1} completed in {query_duration:.2f} seconds.")
        print(f"Found {len(results)} segments.")
        # Imprime info básica del primer resultado (si existe)
        if results:
            print(f"  Top segment (Score: {results[0]['score']:.3f}, Chunks: {results[0]['chunk_start']}-{results[0]['chunk_end']}):")
            # Imprime los primeros N caracteres del contenido
            content_preview = results[0]['content']
            # Quita el header del preview para más claridad
            if content_preview.startswith("Document context:"):
                 content_preview = content_preview.split("\n\n", 1)[-1]
            print(f"    Content preview: {content_preview[:200]}...")
        else:
            print("  No segments found for this query.")

    except Exception as e:
        print(f"Error during query '{query_text}': {e}")
        import traceback
        traceback.print_exc()
        query_durations.append(-1) # Indica error

# --- Resultados de la Evaluación ---
print("\n--- Evaluation Summary ---")
print(f"Document processing time (add_document): {add_duration:.2f}s")
if query_durations:
    valid_query_times = [d for d in query_durations if d >= 0]
    if valid_query_times:
         avg_query_time = sum(valid_query_times) / len(valid_query_times)
         max_query_time = max(valid_query_times)
         print(f"Average query time: {avg_query_time:.2f}s")
         print(f"Max query time: {max_query_time:.2f}s")
    else:
         print("No queries completed successfully.")
else:
    print("No queries were run.")

# --- Limpieza (Opcional) ---
# print("\nCleaning up evaluation data...")
# try:
#     kb.delete()
# except Exception as e:
#     print(f"Error during cleanup: {e}")
# if os.path.exists(eval_storage_path):
#      shutil.rmtree(eval_storage_path, ignore_errors=True)
# print("Cleanup complete.")