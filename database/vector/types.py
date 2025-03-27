# dsrag_minimal/database/vector/types.py
from typing import Optional, Sequence, Union, List, Dict, Any # Añadido Dict, Any
from typing_extensions import TypedDict
import numpy as np

# Metadatos asociados a un chunk específico al almacenar en VectorDB
class ChunkMetadata(TypedDict, total=False): # Usa total=False para campos opcionales
    doc_id: str # Requerido
    chunk_index: int # Requerido
    chunk_header: str
    chunk_text: str # Texto del chunk (puede ser útil en algunos VDBs)
    # Metadatos adicionales del documento o específicos del chunk
    # Heredados de metadata en add_document
    # page_start: Optional[int]
    # page_end: Optional[int]
    # is_visual: Optional[bool]
    # document_title: Optional[str]
    # section_title: Optional[str]
    # ... otros campos de metadata
    # Permite cualquier otro campo
    # Nota: Qdrant prefiere campos de nivel superior para filtrado rápido
    # Considera añadir campos importantes directamente aquí si usas mucho filtrado
    # Ej: doc_id: str (ya está), page_start: Optional[int]

# Tipo para un vector de embedding
Vector = Union[List[float], np.ndarray]

# Resultado de una búsqueda vectorial
class VectorSearchResult(TypedDict):
    doc_id: Optional[str] # ID del documento del chunk encontrado
    metadata: ChunkMetadata # Metadatos completos del chunk encontrado
    similarity: float # Score de similaridad (e.g., coseno)
    vector: Optional[Vector] # Vector del chunk (opcional)

# Filtro para metadatos en búsqueda
class MetadataFilter(TypedDict):
    field: str # Campo en 'metadata' para filtrar (ej: "metadata.page_start")
    operator: str # Operador ('equals', 'in', 'greater_than', etc.)
    value: Union[str, int, float, list] # Valor(es) para comparar
