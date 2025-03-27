# dsrag_minimal/core/reranker.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

# Importación opcional para JinaReranker
try:
    # Renombrado para claridad
    from jina_reranker import Reranker as JinaRerankerClient
    JINA_RERANKER_AVAILABLE = True
except ImportError:
    JINA_RERANKER_AVAILABLE = False
    class JinaRerankerClient: pass # Placeholder

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
        """Serializa la configuración del reranker."""
        return {"subclass_name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict) -> 'Reranker':
        """Crea una instancia de Reranker desde la configuración."""
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
            raise ValueError(f"Unknown reranker subclass: {subclass_name}")

class NoReranker(Reranker):
    """Implementación que no realiza reranking."""
    # (Implementación sin cambios respecto a la anterior)
    # ... (copiar de la versión anterior) ...
    def __init__(self, assign_default_score: bool = False, default_score: float = 0.5, **kwargs):
        self.assign_default_score = assign_default_score
        self.default_score = default_score

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.assign_default_score:
            for doc in documents:
                doc['similarity'] = self.default_score
        # Ordena por score original si existe, si no, mantiene el orden
        return sorted(documents, key=lambda x: x.get('similarity', 0.0), reverse=True)


    def to_dict(self):
        return {
            "subclass_name": "NoReranker",
            "assign_default_score": self.assign_default_score,
            "default_score": self.default_score
        }

class JinaReranker(Reranker):
    """Implementación de Reranker usando Jina AI."""
    def __init__(self, model_name: str = "jina-reranker-v2-base-multilingual", api_key: Optional[str] = None, device: Optional[str] = None, **kwargs):
        if not JINA_RERANKER_AVAILABLE:
            raise ImportError("JinaReranker requires 'jina-reranker'. Install with 'pip install -U jina-reranker'")

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("JINA_API_KEY") # Opcional, para API
        # Jina client infiere el device, pero podemos especificarlo
        # if device: kwargs['device'] = device

        print(f"Initializing JinaReranker model '{model_name}'...")
        try:
            # La inicialización puede variar ligeramente según la versión de jina-reranker
            # Consulta la documentación de jina-reranker si esto falla
            self.client = JinaRerankerClient(model_name, api_key=self.api_key, **kwargs)
            print("JinaReranker model loaded.")
        except Exception as e:
            print(f"Error initializing JinaReranker model '{model_name}': {e}")
            raise

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reordena documentos usando Jina Reranker."""
        if not documents:
            return []

        # Extrae el contenido de texto para el reranker
        # Asegura que el contenido exista y sea string
        doc_contents = [str(doc.get('content', '')) for doc in documents]

        try:
            # Llama a la API de Jina (o modelo local)
            # Asume que el método es 'compute_score' o similar - VERIFICAR DOCS JINA
            # El formato de salida esperado es una lista de scores o dicts {index: score}
            # Ejemplo basado en documentación común:
            reranked_results = self.client.rerank(query, doc_contents, return_documents=False) # Pide solo scores/indices
            # reranked_results suele ser [{'index': original_idx, 'relevance_score': score}, ...]

            if not isinstance(reranked_results, list):
                 print("Warning: Jina reranker did not return a list of results.")
                 return documents # Devuelve original si el formato es inesperado

            # Crea un mapeo de índice original a score nuevo
            score_map = {result['index']: result['relevance_score'] for result in reranked_results}

            # Actualiza los scores en los documentos originales y crea la lista reordenada
            updated_docs = []
            for i, doc in enumerate(documents):
                 if i in score_map:
                      doc['similarity'] = float(score_map[i]) # Actualiza score
                      updated_docs.append(doc)
                 # Opcional: ¿Qué hacer si un doc no está en score_map? ¿Excluirlo o darle score bajo?
                 # else:
                 #      doc['similarity'] = -1.0 # Ejemplo: score muy bajo
                 #      updated_docs.append(doc)

            # Ordena por el nuevo score 'similarity'
            updated_docs.sort(key=lambda x: x['similarity'], reverse=True)
            return updated_docs

        except Exception as e:
            print(f"Error during Jina reranking: {e}")
            # Fallback: Devuelve los documentos ordenados por score original si falla el reranking
            return sorted(documents, key=lambda x: x.get('similarity', 0.0), reverse=True)


    def to_dict(self):
        """Serializa configuración."""
        # No incluye api_key
        return {"subclass_name": "JinaReranker", "model_name": self.model_name}
