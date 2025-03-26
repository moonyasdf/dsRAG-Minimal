# dsrag_minimal/examples/03_query_kb.py
import sys
import os
import time
# Añadir ruta al directorio raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.knowledge_base import KnowledgeBase
# Importa los modelos usados para CREAR el KB si necesitas recargarlos
# O confía en que se carguen desde los metadatos
from core.embedding import SentenceTransformerEmbedding
from core.reranker import JinaReranker
from core.llm import OllamaLLM # Necesario si AutoContext genera headers durante query (raro)

# --- Configuración ---
KB_ID = "local_multilingual_kb" # Debe coincidir con el KB creado
STORAGE_DIR = "~/dsrag_minimal_data"
# Modelo embedding y dimensión deben coincidir con los usados en la creación
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
VECTOR_DIM = 1024
# Reranker usado al crear (o NoReranker si no se usó)
RERANKER_MODEL_NAME = "jina-reranker-v2-base-multilingual"

# Queries de ejemplo
QUERIES = [
    "What are the levels of AGI?",
    "What is the highest level of AGI?",
    "Methodology for determining levels of AGI",
    "Principles for defining levels of AGI",
    "What is Autonomy Level 3",
    "Self-driving cars", # Query menos específica
]

# --- Instanciar Modelos (Necesarios para Query) ---
print("Initializing models needed for querying...")
try:
    # 1. Embedding (Necesario para vectorizar la query)
    embedding = SentenceTransformerEmbedding(model_name=EMBEDDING_MODEL_NAME, dimension=VECTOR_DIM)
    # 2. Reranker (Necesario si se usó uno al crear el KB, aquí Jina)
    reranker = JinaReranker(model_name=RERANKER_MODEL_NAME)
    # 3. LLM (Solo si se generan headers dinámicos en query, usualmente no)
    # auto_context_llm = OllamaLLM(model="gemma:9b") # Probablemente no necesario aquí
except Exception as e:
    print(f"Failed to initialize models: {e}")
    sys.exit(1)

# --- Cargar Knowledge Base ---
print(f"Loading Knowledge Base '{KB_ID}'...")
qdrant_persistent_path = os.path.join(os.path.expanduser(STORAGE_DIR), "qdrant_data")
try:
    kb = KnowledgeBase(
        kb_id=KB_ID,
        storage_directory=STORAGE_DIR,
        vector_dimension=VECTOR_DIM, # Crucial pasar la dimensión correcta
        embedding_model=embedding,   # Pasa el modelo embedding (necesario para query)
        reranker=reranker,           # Pasa el reranker (necesario para query)
        # auto_context_model=auto_context_llm, # Pasa si es necesario
        qdrant_location=None,        # Indica path local
        path=qdrant_persistent_path, # Path persistencia Qdrant
        exists_ok=True               # Debe existir
    )
    print("Knowledge Base loaded.")
    if kb.vector_db.get_num_vectors() == 0:
         print(f"Warning: Knowledge Base '{KB_ID}' is empty.")

except FileNotFoundError:
     print(f"Error: Metadata for KB '{KB_ID}' not found in {STORAGE_DIR}. Did you run example 01 first?")
     sys.exit(1)
except Exception as e:
    print(f"Error loading Knowledge Base '{KB_ID}': {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- Ejecutar Queries ---
print(f"\n--- Running Queries ---")
for i, query_text in enumerate(QUERIES):
    print(f"\n[{i+1}/{len(QUERIES)}] Query: '{query_text}'")
    start_time = time.time()
    try:
        # Usa parámetros RSE deseados
        results = kb.query(
            search_queries=[query_text],
            rse_params="balanced", # O "precision"
            # metadata_filter=..., # Añadir filtro si es necesario
            return_mode="text"
        )
        end_time = time.time()
        print(f"  Query took {end_time - start_time:.2f} seconds.")
        print(f"  Found {len(results)} segments.")

        # Imprimir detalles de los resultados
        for j, seg in enumerate(results[:3]): # Imprime top 3
            page_info = f" (Pages: {seg.get('segment_page_start')}-{seg.get('segment_page_end')})" if seg.get('segment_page_start') else ""
            print(f"  Result {j+1} (Score: {seg['score']:.3f}, Doc: {seg['doc_id']}{page_info}):")
            # Preview del contenido (quita header)
            content_preview = seg['content']
            if content_preview.startswith("Excerpt from document"):
                 content_preview = content_preview.split("\n\n", 1)[-1]
            print(f"    Preview: {content_preview[:250]}...")

    except Exception as e:
        print(f"  Error executing query: {e}")

print("\n--- Queries Finished ---")