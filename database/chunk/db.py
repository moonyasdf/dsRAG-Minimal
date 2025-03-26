# dsrag_minimal/database/chunk/db.py
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Tuple
from .types import FormattedDocument

class ChunkDB(ABC):
    """Clase base abstracta para almacenar el contenido de los chunks."""
    SUBCLASSES = {} # Registro

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.__name__] = cls

    @abstractmethod
    def add_document(self, doc_id: str, chunks: Dict[int, Dict[str, Any]], supp_id: str = "", metadata: dict = {}) -> None:
        """Almacena chunks y metadatos asociados a un doc_id."""
        pass

    @abstractmethod
    def remove_document(self, doc_id: str) -> None:
        """Elimina todos los chunks de un doc_id."""
        pass

    @abstractmethod
    def get_document(self, doc_id: str, include_content: bool = False) -> Optional[FormattedDocument]:
        """Obtiene metadatos y (opcionalmente) contenido completo reconstruido de un documento."""
        pass

    @abstractmethod
    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """Obtiene el texto de un chunk específico."""
        pass

    @abstractmethod
    def get_is_visual(self, doc_id: str, chunk_index: int) -> Optional[bool]:
        """Obtiene el flag is_visual de un chunk."""
        pass

    @abstractmethod
    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Tuple[Optional[int], Optional[int]]:
        """Obtiene las páginas de inicio y fin de un chunk."""
        pass

    @abstractmethod
    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """Obtiene el título del documento asociado a un chunk."""
        pass

    @abstractmethod
    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """Obtiene el resumen del documento asociado a un chunk."""
        pass

    @abstractmethod
    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """Obtiene el título de la sección asociada a un chunk."""
        pass

    @abstractmethod
    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """Obtiene el resumen de la sección asociada a un chunk."""
        pass

    @abstractmethod
    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> List[str]:
        """Obtiene todos los doc_ids únicos, opcionalmente filtrados."""
        pass

    @abstractmethod
    def get_document_count(self) -> int:
         """Devuelve el número de documentos únicos."""
         pass

    @abstractmethod
    def get_total_num_characters(self) -> int:
         """Devuelve el número total de caracteres almacenados."""
         pass

    @abstractmethod
    def delete(self) -> None:
        """Elimina la base de datos de chunks."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serializa configuración mínima."""
        return {"subclass_name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict) -> 'ChunkDB':
        """Reconstruye una instancia desde la configuración."""
        subclass_name = config.get("subclass_name")
        if subclass_name in cls.SUBCLASSES:
            Subclass = cls.SUBCLASSES[subclass_name]
            import inspect
            init_params = inspect.signature(Subclass.__init__).parameters
            valid_config = {k: v for k, v in config.items() if k in init_params}
            try:
                 return Subclass(**valid_config)
            except TypeError as e:
                 raise ValueError(f"Invalid config for {subclass_name}: {e}") from e
        else:
            raise ValueError(f"Unknown ChunkDB subclass: {subclass_name}")
