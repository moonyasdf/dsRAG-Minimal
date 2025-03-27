# dsrag_minimal/core/reranker.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import warnings # Añadido para warnings

# Importación opcional para CrossEncoder
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class CrossEncoder: pass # Placeholder

class Reranker(ABC):
    """Clase base abstracta para modelos de reranking."""
    SUBCLASSES = {} # Registro

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.__name__] = cls

    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reordena una lista de documentos basada en su relevancia para la consulta.
        Espera [{'content': str, 'metadata': dict, 'similarity': float}, ...].
        Debe devolver la lista reordenada, actualizando 'similarity'.
        """
        pass

    def to_dict(self):
        return {"subclass_name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict) -> 'Reranker':
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
            # Intenta cargar NoReranker como default si no se especifica subclase
            if subclass_name is None:
                return NoReranker()
            raise ValueError(f"Unknown reranker subclass: {subclass_name}")

class NoReranker(Reranker):
    """Implementación que no realiza reranking."""
    def __init__(self, assign_default_score: bool = False, default_score: float = 0.5, **kwargs):
        self.assign_default_score = assign_default_score
        self.default_score = default_score

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.assign_default_score:
            for doc in documents:
                doc['similarity'] = self.default_score
        # Ordena por score original si existe, si no, mantiene el orden
        try:
             # Intenta ordenar numéricamente
             return sorted(documents, key=lambda x: float(x.get('similarity', 0.0)), reverse=True)
        except (ValueError, TypeError):
             # Si similarity no es numérico, devuelve tal cual
             warnings.warn("Could not sort documents by similarity score in NoReranker.")
             return documents


    def to_dict(self):
        return {
            "subclass_name": "NoReranker",
            "assign_default_score": self.assign_default_score,
            "default_score": self.default_score
        }

class CrossEncoderReranker(Reranker):
    """Implementación de Reranker usando sentence_transformers.CrossEncoder."""
    def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual", device: Optional[str] = None, **kwargs):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("CrossEncoderReranker requires 'sentence-transformers'. Install with 'pip install sentence-transformers'")

        self.model_name = model_name
        # Permite pasar argumentos adicionales a CrossEncoder, como torch_dtype
        automodel_args = kwargs.get("automodel_args", {"torch_dtype": "auto"})
        trust_remote_code = kwargs.get("trust_remote_code", True) # Necesario para Jina

        print(f"Initializing CrossEncoder model '{model_name}'...")
        try:
            # Carga el modelo CrossEncoder
            self.model = CrossEncoder(
                model_name,
                max_length=1024, # Ajusta según el modelo, Jina recomienda 1024
                device=device, # None para auto-detección
                automodel_args=automodel_args,
                trust_remote_code=trust_remote_code
            )
            print("CrossEncoder model loaded.")
        except Exception as e:
            print(f"Error loading CrossEncoder model '{model_name}': {e}")
            raise

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reordena documentos usando CrossEncoder."""
        if not documents:
            return []

        # Extrae el contenido de texto relevante para el reranking
        # Usa el 'content' que se pasó al reranker (header + chunk_text)
        doc_contents = [str(doc.get('content', '')) for doc in documents]

        # Crea los pares [query, document_content]
        sentence_pairs = [[query, doc_content] for doc_content in doc_contents]

        try:
            # Calcula los scores usando predict
            # batch_size puede ajustarse según la memoria de la GPU/CPU
            scores = self.model.predict(sentence_pairs, convert_to_numpy=True, show_progress_bar=False)

            if scores is None or len(scores) != len(documents):
                print("Warning: CrossEncoder predict did not return expected scores.")
                return sorted(documents, key=lambda x: x.get('similarity', 0.0), reverse=True)

            # Actualiza la similaridad en los documentos originales
            for i, doc in enumerate(documents):
                doc['similarity'] = float(scores[i]) # Actualiza con el nuevo score

            # Ordena los documentos por el nuevo score
            documents.sort(key=lambda x: x['similarity'], reverse=True)
            return documents

        except Exception as e:
            print(f"Error during CrossEncoder reranking: {e}")
            # Fallback: Devuelve ordenado por score original
            return sorted(documents, key=lambda x: x.get('similarity', 0.0), reverse=True)

    def to_dict(self):
        """Serializa configuración."""
        # device no se serializa fácilmente, se re-detectará al cargar
        return {"subclass_name": "CrossEncoderReranker", "model_name": self.model_name}

# Añade la nueva clase al registro para from_dict
Reranker.SUBCLASSES["CrossEncoderReranker"] = CrossEncoderReranker
