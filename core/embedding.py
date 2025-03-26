# dsrag_minimal/core/embedding.py
from abc import ABC, abstractmethod
from typing import Optional, List, Union
import os
import ollama
import numpy as np
from openai import OpenAI

# Tipo para vectores
Vector = Union[List[float], np.ndarray]

# Dimensiones conocidas para modelos locales (ejemplos)
# El usuario puede necesitar añadir más o pasarlas explícitamente
LOCAL_DIMENSIONALITY = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "llama3": 4096,
    # Añadir otros modelos Ollama según sea necesario
}

class Embedding(ABC):
    """Clase base abstracta para modelos de embedding."""
    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension

    @abstractmethod
    def get_embeddings(self, text: Union[str, List[str]], input_type: Optional[str] = None) -> Union[Vector, List[Vector]]:
        """Genera embeddings para el texto de entrada."""
        pass

    def to_dict(self):
        """Serializa la configuración del modelo de embedding."""
        # Devuelve solo lo esencial para la recreación mínima
        return {"subclass_name": self.__class__.__name__, "dimension": self.dimension}

    @classmethod
    def from_dict(cls, config: dict) -> 'Embedding':
        """Crea una instancia de Embedding desde un diccionario de configuración."""
        subclass_name = config.get("subclass_name")
        # Simplificado: Solo soporta las clases definidas aquí explícitamente
        if subclass_name == "OllamaEmbedding":
            return OllamaEmbedding(**config)
        elif subclass_name == "OpenAIEmbedding":
            return OpenAIEmbedding(**config)
        else:
            # Permite la inyección de clases personalizadas si se registran
            # o si se pasan directamente como instancias
            raise ValueError(f"Unknown or unsupported embedding subclass: {subclass_name}")

class OllamaEmbedding(Embedding):
    """Implementación de Embedding usando Ollama para modelos locales."""
    def __init__(self, model: str = "nomic-embed-text", dimension: Optional[int] = None, client: Optional[ollama.Client] = None, **kwargs):
        # Determina la dimensión si no se proporciona
        if dimension is None:
            dimension = LOCAL_DIMENSIONALITY.get(model)
            if dimension is None:
                raise ValueError(f"Dimension for Ollama model {model} is unknown and must be provided.")
        super().__init__(dimension)
        self.model = model
        self.client = client or ollama.Client()
        # Intenta hacer pull del modelo al inicializar, maneja errores silenciosamente
        try:
            print(f"Checking/Pulling Ollama model: {model}...")
            ollama.pull(model)
            print(f"Model {model} available.")
        except Exception as e:
            print(f"Warning: Could not pull Ollama model '{model}'. Ensure it's available locally. Error: {e}")

    def get_embeddings(self, text: Union[str, List[str]], input_type: Optional[str] = None) -> Union[Vector, List[Vector]]:
        """Obtiene embeddings usando el cliente Ollama."""
        try:
            if isinstance(text, list):
                # Procesamiento en batch si es una lista
                results = [self.client.embeddings(model=self.model, prompt=t)["embedding"] for t in text]
                return results
            else:
                # Procesamiento único si es un string
                return self.client.embeddings(model=self.model, prompt=text)["embedding"]
        except Exception as e:
            print(f"Error getting Ollama embeddings for model {self.model}: {e}")
            # Devuelve un vector de ceros con la dimensión correcta en caso de error
            # para evitar fallos posteriores, aunque esto afectará la calidad.
            if isinstance(text, list):
                return [np.zeros(self.dimension).tolist()] * len(text)
            else:
                return np.zeros(self.dimension).tolist()

    def to_dict(self):
        """Serializa la configuración del modelo Ollama."""
        # Solo incluye parámetros necesarios para la reconstrucción
        return {"subclass_name": "OllamaEmbedding", "model": self.model, "dimension": self.dimension}

class OpenAIEmbedding(Embedding):
    """Implementación de Embedding usando la API de OpenAI (como ejemplo/opción)."""
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 768, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(dimension)
        self.model = model
        # Usa variables de entorno si no se proporcionan claves/url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("DSRAG_OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it during initialization.")
        # Inicializa el cliente OpenAI
        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url
        self.client = OpenAI(**client_args)

    def get_embeddings(self, text: Union[str, List[str]], input_type: Optional[str] = None) -> Union[Vector, List[Vector]]:
        """Obtiene embeddings usando el cliente OpenAI."""
        try:
            # Asegura que la entrada sea una lista para la API
            input_texts = [text] if isinstance(text, str) else text
            response = self.client.embeddings.create(
                input=input_texts, model=self.model, dimensions=self.dimension
            )
            embeddings = [item.embedding for item in response.data]
            # Devuelve un solo vector si la entrada fue un string
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            print(f"Error getting OpenAI embeddings: {e}")
            # Devuelve ceros en caso de error
            if isinstance(text, list):
                return [np.zeros(self.dimension).tolist()] * len(text)
            else:
                return np.zeros(self.dimension).tolist()

    def to_dict(self):
        """Serializa la configuración del modelo OpenAI."""
        # No incluye api_key por seguridad
        return {"subclass_name": "OpenAIEmbedding", "model": self.model, "dimension": self.dimension, "base_url": self.base_url}

# Permitir la inyección directa de una instancia de Embedding
# Ejemplo: kb = KnowledgeBase(..., embedding_model=MiEmbeddingPersonalizado())