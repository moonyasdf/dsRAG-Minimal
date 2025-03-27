# dsrag_minimal/database/vector/qdrant_db.py
# (Manteniendo la implementación original de Qdrant, pero asegurándose
# de que sea la única opción importada/utilizada por defecto en knowledge_base.py)

# Asegúrate de que las importaciones y dependencias sean solo para Qdrant
from typing import Sequence, cast, List, Dict, Any, Optional
import uuid
import numpy as np
import warnings # Añadido para warnings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from database.vector.db import VectorDB # Importa la clase base simplificada
from database.vector.types import ChunkMetadata, Vector, VectorSearchResult, MetadataFilter # Importa tipos

def convert_id(_id: str) -> str:
    """Convierte string a UUID para Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, _id))

# Simplificación de format_metadata_filter (Asumiendo que el filtro de entrada es simple)
def format_qdrant_filter(metadata_filter: Optional[MetadataFilter]) -> Optional[models.Filter]:
    """Convierte el filtro simple a formato Qdrant."""
    if not metadata_filter:
        return None

    field = metadata_filter["field"]
    operator = metadata_filter["operator"]
    value = metadata_filter["value"]

    # Mapeo simple (podría expandirse si se necesitan más operadores)
    if operator == "equals":
        return models.Filter(must=[models.FieldCondition(key=f"metadata.{field}", match=models.MatchValue(value=value))])
    elif operator == "in":
        if not isinstance(value, list): raise ValueError("'in' operator requires a list value")
        return models.Filter(must=[models.FieldCondition(key=f"metadata.{field}", match=models.MatchAny(any=value))])
    # Añadir más operadores según sea necesario (lt, gt, etc.)
    # elif operator == "greater_than":
    #     return models.Filter(must=[models.FieldCondition(key=f"metadata.{field}", range=models.Range(gt=value))])

    warnings.warn(f"Operator '{operator}' not fully supported in this minimal Qdrant filter conversion. Returning no filter.")
    return None

class QdrantVectorDB(VectorDB):
    """Implementación de VectorDB usando Qdrant."""
    def __init__(
        self,
        kb_id: str,
        vector_dimension: int, # Dimensión es ahora requerida
        location: Optional[str] = ":memory:", # Default a en memoria para simplicidad local
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs # Ignora otros parámetros
    ):
        self.kb_id = kb_id.lower().replace("_", "-") # Nombre de colección válido para Qdrant
        self.vector_dimension = vector_dimension
        # Guarda opciones para to_dict
        self.client_options = {
            "location": location, "url": url, "port": port, "grpc_port": grpc_port,
            "prefer_grpc": prefer_grpc, "https": https, "api_key": api_key,
            "prefix": prefix, "timeout": timeout, "host": host, "path": path,
        }
        try:
            self.client = QdrantClient(**self.client_options)
            # Intenta crear la colección si no existe
            self._create_collection_if_not_exists()
        except Exception as e:
            print(f"Error initializing Qdrant client for kb_id '{self.kb_id}': {e}")
            print("Ensure Qdrant server is running or location path is valid.")
            raise

    def _create_collection_if_not_exists(self):
        """Crea la colección en Qdrant si es necesario."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.kb_id not in collection_names:
                print(f"Creating Qdrant collection: {self.kb_id} with dimension {self.vector_dimension}")
                self.client.create_collection(
                    collection_name=self.kb_id,
                    vectors_config=VectorParams(size=self.vector_dimension, distance=Distance.COSINE),
                )
                print(f"Collection {self.kb_id} created.")
        except Exception as e:
            # Podría fallar si se ejecuta en paralelo, intenta obtenerla de nuevo
            try:
                 self.client.get_collection(collection_name=self.kb_id)
                 print(f"Collection {self.kb_id} already exists.")
            except Exception as e2:
                 print(f"Error creating or verifying Qdrant collection '{self.kb_id}': {e} / {e2}")
                 raise

    def add_vectors(self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]) -> None:
        """Añade vectores y metadatos a Qdrant."""
        if not vectors: return # No hacer nada si no hay vectores
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors and metadata must match.")

        points = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            # Genera ID único basado en doc_id y chunk_index
            point_id = convert_id(f"{meta.get('doc_id', '')}_{meta.get('chunk_index', i)}")
            # Prepara el payload (metadata)
            payload = {"metadata": meta} # Almacena toda la metadata original en un campo
            # Puedes añadir campos de nivel superior para filtrado si es necesario
            # payload['doc_id'] = meta.get('doc_id')
            # payload['chunk_index'] = meta.get('chunk_index')
            points.append(models.PointStruct(id=point_id, vector=vector.tolist() if isinstance(vector, np.ndarray) else vector, payload=payload))

        try:
            # Usa wait=True para asegurar que la operación se complete (puede impactar rendimiento)
            # Considera wait=False para mayor velocidad si la consistencia inmediata no es crítica
            self.client.upsert(collection_name=self.kb_id, points=points, wait=False)
        except Exception as e:
            print(f"Error upserting points to Qdrant collection '{self.kb_id}': {e}")
            # Considera reintentos o manejo de errores más robusto aquí
            raise

    def remove_document(self, doc_id: str) -> None:
        """Elimina vectores asociados a un doc_id."""
        try:
            # Elimina por filtro usando el campo doc_id dentro de metadata
            self.client.delete(
                collection_name=self.kb_id,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.doc_id", # Accede al campo dentro de metadata
                                match=models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                ),
                # wait=True # Opcional: esperar confirmación
            )
        except Exception as e:
            print(f"Error deleting document '{doc_id}' from Qdrant collection '{self.kb_id}': {e}")
            # Considera logging o manejo de errores

    def search(self, query_vector: Vector, top_k: int = 10, metadata_filter: Optional[MetadataFilter] = None) -> List[VectorSearchResult]:
        """Busca vectores similares en Qdrant."""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        qdrant_filter = format_qdrant_filter(metadata_filter)

        try:
            search_result = self.client.search(
                collection_name=self.kb_id,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True, # Asegura que se devuelva la metadata
                # with_vectors=False # No necesitamos el vector de vuelta usualmente
            )
        except Exception as e:
            # Maneja el caso donde la colección podría no existir o estar vacía
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                print(f"Qdrant collection '{self.kb_id}' not found or empty during search.")
                return []
            print(f"Error searching Qdrant collection '{self.kb_id}': {e}")
            return [] # Devuelve lista vacía en caso de error

        # Formatea los resultados
        results: List[VectorSearchResult] = []
        for hit in search_result:
            # Asume que la metadata original está en hit.payload['metadata']
            original_metadata = hit.payload.get("metadata", {})
            results.append(
                VectorSearchResult(
                    doc_id=original_metadata.get("doc_id"), # Extrae de la metadata original
                    metadata=original_metadata,
                    similarity=hit.score,
                    # vector=hit.vector # Opcional: añadir si se necesita
                )
            )
        return results

    def delete(self) -> None:
        """Elimina la colección Qdrant."""
        try:
            if self.client.collection_exists(self.kb_id):
                 self.client.delete_collection(collection_name=self.kb_id)
                 print(f"Qdrant collection '{self.kb_id}' deleted.")
            else:
                 print(f"Qdrant collection '{self.kb_id}' does not exist, skipping deletion.")
        except Exception as e:
            print(f"Error deleting Qdrant collection '{self.kb_id}': {e}")
            # No relanzar la excepción para permitir que el programa continúe si es posible

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la configuración de QdrantDB."""
        return {
            "subclass_name": "QdrantVectorDB",
            "kb_id": self.kb_id,
            "vector_dimension": self.vector_dimension,
            **self.client_options
        }

    def get_num_vectors(self) -> int:
        """Obtiene el número de vectores en la colección."""
        try:
             count_result = self.client.count(collection_name=self.kb_id, exact=True)
             return count_result.count
        except Exception as e:
             # Si la colección no existe, devuelve 0
             if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                  return 0
             print(f"Error getting vector count for Qdrant collection '{self.kb_id}': {e}")
             return 0 # O relanzar si es un error crítico

    # --- Métodos de la clase base no implementados explícitamente ---
    # (Asume que la clase base es abstracta y no tiene save/load)
    # def save(self): pass # No aplicable para Qdrant (persistencia manejada por el servidor/local)
    # def load(self): pass # No aplicable, la conexión se establece en __init__
