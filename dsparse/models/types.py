# dsrag_minimal/dsparse/models/types.py
from typing import Optional, TypedDict, List, Dict, Any # Añadido Dict, Any
# Importación absoluta
from dsparse.file_parsing.file_system import FileSystem # Asume que FileSystem está aquí

class Line(TypedDict):
    content: str
    element_type: str
    page_number: Optional[int]
    is_visual: Optional[bool]

class Section(TypedDict):
    title: str
    start: int
    end: int
    content: str

class Chunk(TypedDict):
    line_start: int
    line_end: int
    content: str
    page_start: int
    page_end: int
    section_index: int
    is_visual: bool

class SemanticSectioningConfig(TypedDict):
    use_semantic_sectioning: Optional[bool]
    llm_provider: Optional[str]
    model: Optional[str]
    language: Optional[str]

class ChunkingConfig(TypedDict):
    chunk_size: Optional[int]
    min_length_for_chunking: Optional[int]
