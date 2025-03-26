# dsrag_minimal/core/reranker.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class Reranker(ABC):
    """Clase base abstracta para modelos de reranking."""
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reordena una lista de documentos basada en su relevancia para la consulta.
        Espera una lista de diccionarios, cada uno con al menos 'content' y 'metadata'.
        Debe devolver la lista reordenada, potencialmente con un score añadido.
        """
        pass

    def to_dict(self):
        """Serializa la configuración del reranker."""
        return {"subclass_name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict) -> 'Reranker':
        """Crea una instancia de Reranker desde la configuración."""
        subclass_name = config.get("subclass_name")
        if subclass_name == "NoReranker":
            return NoReranker(**config)
        # Añadir aquí lógica para otros rerankers si se incluyen como opción inyectable
        # elif subclass_name == "CohereReranker":
        #    return CohereReranker(**config)
        else:
            raise ValueError(f"Unknown or unsupported reranker subclass: {subclass_name}")

class NoReranker(Reranker):
    """Implementación que no realiza reranking, simplemente pasa los resultados."""
    def __init__(self, assign_default_score: bool = False, default_score: float = 0.5, **kwargs):
        """
        Args:
            assign_default_score: Si es True, asigna un score por defecto a cada resultado.
                                  Útil si el VectorDB no proporciona scores o no son fiables.
            default_score: El score a asignar si assign_default_score es True.
        """
        self.assign_default_score = assign_default_score
        self.default_score = default_score

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Devuelve los documentos en el orden original, opcionalmente añade score."""
        if self.assign_default_score:
            for doc in documents:
                # Añade o sobrescribe el score de similaridad
                doc['similarity'] = self.default_score
        # Si no se asigna score por defecto, se asume que 'similarity' ya existe o no es necesaria
        return documents

    def to_dict(self):
        """Serializa la configuración de NoReranker."""
        return {"subclass_name": "NoReranker",
                "assign_default_score": self.assign_default_score,
                "default_score": self.default_score}

# NOTA: Se elimina CohereReranker y VoyageReranker para minimizar.
# Si se necesitan, se pueden añadir aquí o inyectar externamente.
# El código de CohereReranker (por ejemplo) necesitaría `import cohere` y
# lógica similar a la original pero simplificada.