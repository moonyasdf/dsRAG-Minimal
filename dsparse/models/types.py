# dsrag_minimal/dsparse/models/types.py
from typing import Optional, TypedDict, List, Dict, Any # Añadido Dict, Any
# Importación absoluta
from dsparse.file_parsing.file_system import FileSystem # Asume que FileSystem está aquí

# ElementType y Element se mantienen por si se usan internamente, aunque VLM esté fuera
class ElementType(TypedDict):
    name: str
    instructions: str
    is_visual: bool

class Element(TypedDict):
    type: str
    content: str
    page_number: Optional[int]

class Line(TypedDict):
    content: str
    element_type: str
    page_number: Optional[int]
    is_visual: Optional[bool] # Mantenido por compatibilidad interna

class Section(TypedDict):
    title: str
    start: int
    end: int
    content: str

class Chunk(TypedDict):
    line_start: int
    line_end: int
    content: str
    page_start: Optional[int] # Hacer opcional por si no hay páginas
    page_end: Optional[int]   # Hacer opcional
    section_index: int
    is_visual: bool

class SemanticSectioningConfig(TypedDict, total=False): # Usa total=False
    use_semantic_sectioning: Optional[bool]
    # llm_provider: Optional[str] # Ya no se usa, se pasa instancia LLM
    # model: Optional[str] # Ya no se usa, se pasa instancia LLM
    language: Optional[str]
    llm: Optional[Any] # Para pasar la instancia LLM
    max_characters: Optional[int] # Límite para la llamada LLM

class ChunkingConfig(TypedDict, total=False): # Usa total=False
    chunk_size: Optional[int]
    min_length_for_chunking: Optional[int]
    
