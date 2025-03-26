# dsrag_minimal/database/vector/db.py
from abc import ABC, abstractmethod
from typing import Sequence, Optional, List, Dict, Any
from .types import ChunkMetadata, Vector, VectorSearchResult, MetadataFilter

class VectorDB(ABC):
    """Clase base abstracta para bases de datos vectoriales."""
    SUBCLASSES = {} # Registro para from_dict

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.__name__] = cls

    @abstractmethod
    def add_vectors(self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]) -> None:
        """Almacena vectores y metadatos."""
        pass

    @abstractmethod
    def remove_document(self, doc_id: str) -> None:
        """Elimina vectores por doc_id."""
        pass

    @abstractmethod
    def search(self, query_vector: Vector, top_k: int = 10, metadata_filter: Optional[MetadataFilter] = None) -> List[VectorSearchResult]:
        """Busca vectores similares."""
        pass

    @abstractmethod
    def delete(self) -> None:
        """Elimina la estructura de la base de datos (e.g., colección)."""
        pass

    @abstractmethod
    def get_num_vectors(self) -> int:
        """Devuelve el número total de vectores almacenados."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serializa configuración mínima."""
        return {"subclass_name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict) -> 'VectorDB':
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
            raise ValueError(f"Unknown VectorDB subclass: {subclass_name}")
