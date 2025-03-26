# dsrag_minimal/examples/01_create_kb_local.py
import sys
import os
# Añadir ruta al directorio raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa modelos específicos
from core.embedding import SentenceTransformerEmbedding # Usa E5
from core.llm import OllamaLLM # Usa Ollama para AutoContext/Sectioning
from core.reranker import JinaReranker # Usa Jina
from create_kb import create_kb_from_file
import time

# --- Configuración Local ---
KB_ID = "local_multilingual_kb" # Nuevo ID para reflejar modelos
STORAGE_DIR = "~/dsrag_minimal_data"
DOC_PATH = "../tests/data/levels_of_agi.pdf" # Asegúrate que existe

# --- Modelos Específicos Solicitados ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct" # HF E5 Multilingual
RERANKER_MODEL_NAME = "jina-reranker-v2-base-multilingual" # HF Jina Multilingual
LLM_MODEL_NAME = "gemma:9b" # O 'llama3', etc. - Modelo Ollama para AutoContext/Sectioning
# Determina dimensiones (pueden inferirse, pero es mejor especificarlas si se conocen)
VECTOR_DIM = 1024 # Para E5 large instruct
# Device para modelos locales (auto-detecta GPU o usa CPU)
DEVICE = None # None para auto-detección por SentenceTransformer/Jina

# --- Limpieza Previa (Opcional pero recomendado) ---
eval_storage_path = os.path.expanduser(STORAGE_DIR)
if os.path.exists(os.path.join(eval_storage_path, "metadata", f"{KB_ID}.json")):
    print(f"KB '{KB_ID}' seems to exist. Deleting for fresh start.")
    import shutil
    from database.vector.qdrant_db import QdrantVectorDB
    try:
        qdrant_path = os.path.join(eval_storage_path, "qdrant_data")
        if os.path.exists(qdrant_path):
             qdrant_db_temp = QdrantVectorDB(kb_id=KB_ID, vector_dimension=VECTOR_DIM, path=qdrant_path)
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

# --- Instanciar Modelos ---
print("Initializing local models (SentenceTransformer, Jina, Ollama)...")
try:
    # 1. Embedding: SentenceTransformer E5
    embedding = SentenceTransformerEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        dimension=VECTOR_DIM, # Pasa la dimensión conocida
        device=DEVICE
    )
    # 2. Reranker: Jina Multilingual
    reranker = JinaReranker(
        model_name=RERANKER_MODEL_NAME,
        # api_key=None, # No necesario para modelo local
        # device=DEVICE # Jina puede inferir o puedes pasarlo si es necesario/soportado
    )
    # 3. LLM: Ollama (para AutoContext y Sectioning)
    llm = OllamaLLM(model=LLM_MODEL_NAME)
except ImportError as ie:
     print(f"Import Error: {ie}. Make sure all dependencies are installed ('pip install -r requirements.txt')")
     sys.exit(1)
except Exception as e:
    print(f"Failed to initialize models: {e}")
    print("Ensure Ollama is running and models are downloaded/available.")
    print(f"Required: Ollama ({LLM_MODEL_NAME}), SentenceTransformer ({EMBEDDING_MODEL_NAME}), Jina ({RERANKER_MODEL_NAME})")
    sys.exit(1)

# --- Crear KB ---
print(f"\nCreating KB '{KB_ID}' with specified local models...")
start_time = time.time()
qdrant_persistent_path = os.path.join(os.path.expanduser(STORAGE_DIR), "qdrant_data")
try:
    kb = create_kb_from_file(
        kb_id=KB_ID,
        file_path=DOC_PATH,
        embedding_model=embedding,      # Inyecta E5
        auto_context_model=llm,         # Inyecta Ollama LLM
        reranker=reranker,              # Inyecta Jina
        storage_directory=STORAGE_DIR,
        vector_dimension=embedding.dimension, # Usa dimensión del modelo cargado
        qdrant_location=None,           # Usa path local
        path=qdrant_persistent_path,    # Path persistencia Qdrant
        title="AGI Levels (Multilingual Local)",
        exists_ok=False,                # Fuerza la creación
        semantic_sectioning_config={"use_semantic_sectioning": True, "llm": llm}, # Pasa LLM para sectioning
        auto_context_config={"get_document_summary": True},
    )
    end_time = time.time()
    print(f"KB '{KB_ID}' created and document added in {end_time - start_time:.2f} seconds.")
    print(f"Vector count: {kb.vector_db.get_num_vectors()}")
    print(f"Document count: {kb.chunk_db.get_document_count()}")

except ValueError as ve:
     if "already exists" in str(ve):
          print(f"Error: {ve}. Clean up previous data or set exists_ok=True.")
     else:
          print(f"ValueError during KB creation: {ve}")
except Exception as e:
    print(f"An unexpected error occurred during KB creation: {e}")
    import traceback
    traceback.print_exc()