# dsrag_minimal/examples/01_create_kb_local.py
import sys
import os
# Añadir ruta al directorio raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.embedding import OllamaEmbedding
from core.llm import OllamaLLM
from create_kb import create_kb_from_file
import time

# --- Configuración Local ---
KB_ID = "local_agi_kb"
STORAGE_DIR = "~/dsrag_minimal_data" # Usará este directorio
DOC_PATH = "../tests/data/levels_of_agi.pdf" # Ajusta según tu estructura
# Asegúrate que estos modelos existen en tu Ollama local
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3" # Para AutoContext / Semantic Sectioning
VECTOR_DIM = 768 # Dimensión para nomic-embed-text

# --- Limpieza Previa (Opcional) ---
# (Añadir código de limpieza similar al de evaluation.py si se desea)
eval_storage_path = os.path.expanduser(STORAGE_DIR)
if os.path.exists(os.path.join(eval_storage_path, "metadata", f"{KB_ID}.json")):
     print(f"KB '{KB_ID}' seems to exist. Deleting for fresh start.")
     # Deberíamos instanciar KB temporalmente para borrar, o borrar manualmente
     # Borrado manual simple:
     import shutil
     from database.vector.qdrant_db import QdrantVectorDB
     try:
        q_path = os.path.join(eval_storage_path, "qdrant_data")
        if os.path.exists(q_path):
            qdrant_db_temp = QdrantVectorDB(kb_id=KB_ID, vector_dimension=VECTOR_DIM, path=q_path)
            qdrant_db_temp.delete()
     except Exception as e: print(f"Ignoring Qdrant delete error: {e}")
     db_path = os.path.join(eval_storage_path, "chunk_storage", f"{KB_ID}.db")
     if os.path.exists(db_path): os.remove(db_path)
     meta_path = os.path.join(eval_storage_path, "metadata", f"{KB_ID}.json")
     if os.path.exists(meta_path): os.remove(meta_path)
     fs_path = os.path.join(eval_storage_path, "kb_files", KB_ID)
     if os.path.exists(fs_path): shutil.rmtree(fs_path, ignore_errors=True)
     print("Previous KB data cleaned.")
     time.sleep(1)


# --- Instanciar Modelos Locales ---
print("Initializing local models...")
try:
    embedding = OllamaEmbedding(model=EMBEDDING_MODEL, dimension=VECTOR_DIM)
    llm = OllamaLLM(model=LLM_MODEL)
except Exception as e:
    print(f"Failed to initialize Ollama models: {e}")
    print("Ensure Ollama is running and models are downloaded (`ollama pull nomic-embed-text`, `ollama pull llama3`)")
    sys.exit(1)

# --- Crear KB ---
print(f"\nCreating KB '{KB_ID}' with local models...")
start_time = time.time()
try:
    kb = create_kb_from_file(
        kb_id=KB_ID,
        file_path=DOC_PATH,
        embedding_model=embedding,      # Inyecta instancia embedding
        auto_context_model=llm,         # Inyecta instancia LLM
        # reranker=None,                # Usa NoReranker por defecto
        storage_directory=STORAGE_DIR,
        vector_dimension=VECTOR_DIM,    # Necesario para Qdrant
        qdrant_location=None,           # Usa path local
        path=os.path.join(os.path.expanduser(STORAGE_DIR), "qdrant_data"), # Path persistencia Qdrant
        title="AGI Levels (Local)",
        exists_ok=False, # Fuerza la creación
        # Configuración específica para añadir documento (opcional)
        semantic_sectioning_config={"use_semantic_sectioning": True, "llm": llm}, # Pasa el LLM
        auto_context_config={"get_document_summary": True, "get_section_summaries": False},
    )
    end_time = time.time()
    print(f"KB '{KB_ID}' created and document added in {end_time - start_time:.2f} seconds.")
    print(f"Vector count: {kb.vector_db.get_num_vectors()}")
    print(f"Document count: {kb.chunk_db.get_document_count()}")

except ValueError as ve:
     if "already exists" in str(ve):
          print(f"Error: {ve}. Set exists_ok=True in create_kb_from_file or clean up previous data.")
     else:
          print(f"ValueError during KB creation: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()