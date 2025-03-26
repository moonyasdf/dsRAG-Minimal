# dsrag_minimal/database/chunk/types.py
from typing import Optional, Dict, Any
from typing_extensions import TypedDict

# Formato para devolver información de un documento desde ChunkDB.get_document
class FormattedDocument(TypedDict):
    id: str          # doc_id
    title: Optional[str]
    content: Optional[str] # Texto completo reconstruido (si include_content=True)
    summary: Optional[str]
    created_on: Optional[int] # Timestamp Unix
    supp_id: Optional[str]
    metadata: Optional[Dict[str, Any]]
    chunk_count: int
