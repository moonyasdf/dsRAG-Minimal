# dsrag_minimal/core/embedding.py
from abc import ABC, abstractmethod
from typing import Optional, List, Union
import os
import ollama
import numpy as np
from openai import OpenAI # Necesario para OpenAIEmbedding

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class SentenceTransformer: pass
    class torch: # Placeholder simple para que el type hint funcione
         @staticmethod
         def cuda.is_available(): return False

# Tipo para vectores
Vector = Union[List[float], np.ndarray]

# Dimensiones conocidas para modelos locales (ejemplos)
LOCAL_DIMENSIONALITY = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024, # Otro modelo común
    "llama3": 4096,
    "gemma:2b": 2048, # Ejemplo Gemma
    "gemma:7b": 3072,
    "mistral": 4096,
    # Modelo específico solicitado (debe coincidir con el modelo real)
    "intfloat/multilingual-e5-large-instruct": 1024,
}

class Embedding(ABC):
    """Clase base abstracta para modelos de embedding."""
    SUBCLASSES = {} # Registro de subclases

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension

    # Decorador para registrar subclases automáticamente
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.__name__] = cls

    @abstractmethod
    def get_embeddings(self, text: Union[str, List[str]], input_type: Optional[str] = None) -> Union[Vector, List[Vector]]:
        """Genera embeddings para el texto de entrada."""
        pass

    def to_dict(self):
        """Serializa la configuración del modelo de embedding."""
        return {"subclass_name": self.__class__.__name__, "dimension": self.dimension}

    @classmethod
    def from_dict(cls, config: dict) -> 'Embedding':
        """Crea una instancia de Embedding desde un diccionario de configuración."""
        subclass_name = config.get("subclass_name")
        if subclass_name in cls.SUBCLASSES:
            # Pasa solo los argumentos esperados por el __init__ de la subclase
            # Esto evita errores si hay claves extra en config (como 'subclass_name')
            Subclass = cls.SUBCLASSES[subclass_name]
            # Obtiene los parámetros del constructor de la subclase
            import inspect
            init_params = inspect.signature(Subclass.__init__).parameters
            # Filtra config para pasar solo los parámetros válidos
            valid_config = {k: v for k, v in config.items() if k in init_params}
            try:
                 return Subclass(**valid_config)
            except TypeError as e:
                 print(f"Error initializing {subclass_name} with config {valid_config}: {e}")
                 raise ValueError(f"Invalid configuration for {subclass_name}") from e
        else:
            raise ValueError(f"Unknown embedding subclass: {subclass_name}")

class OllamaEmbedding(Embedding):
    """Implementación de Embedding usando Ollama para modelos locales."""
    def __init__(self, model: str = "nomic-embed-text", dimension: Optional[int] = None, client: Optional[ollama.Client] = None, **kwargs):
        if dimension is None:
            dimension = LOCAL_DIMENSIONALITY.get(model)
            if dimension is None:
                print(f"Warning: Dimension for Ollama model '{model}' is unknown. Trying to infer...")
                try:
                    # Intenta obtener un embedding para inferir la dimensión
                    client_temp = client or ollama.Client()
                    temp_emb = client_temp.embeddings(model=model, prompt="test")["embedding"]
                    dimension = len(temp_emb)
                    print(f"Inferred dimension for '{model}': {dimension}")
                    LOCAL_DIMENSIONALITY[model] = dimension # Cachea la dimensión inferida
                except Exception as e:
                     raise ValueError(f"Dimension for Ollama model {model} could not be inferred and must be provided. Error: {e}")

        super().__init__(dimension)
        self.model = model
        self.client = client or ollama.Client()
        try:
            print(f"Checking/Pulling Ollama model: {model}...")
            # Comprueba si el modelo existe localmente antes de intentar hacer pull
            local_models = [m['name'] for m in ollama.list().get('models', [])]
            if model not in local_models:
                 print(f"Model '{model}' not found locally. Pulling...")
                 ollama.pull(model)
                 print(f"Model {model} pulled successfully.")
            else:
                 print(f"Model {model} found locally.")
        except Exception as e:
            print(f"Warning: Could not pull or verify Ollama model '{model}'. Ensure it's available locally. Error: {e}")

    def get_embeddings(self, text: Union[str, List[str]], input_type: Optional[str] = None) -> Union[Vector, List[Vector]]:
        try:
            if isinstance(text, list):
                results = [self.client.embeddings(model=self.model, prompt=t)["embedding"] for t in text]
                return results
            else:
                return self.client.embeddings(model=self.model, prompt=text)["embedding"]
        except Exception as e:
            print(f"Error getting Ollama embeddings for model {self.model}: {e}")
            dim = self.dimension or 1 # Fallback dimension si no está definida
            if isinstance(text, list):
                return [np.zeros(dim).tolist()] * len(text)
            else:
                return np.zeros(dim).tolist()

    def to_dict(self):
        # No guarda el cliente
        return {"subclass_name": "OllamaEmbedding", "model": self.model, "dimension": self.dimension}

class OpenAIEmbedding(Embedding):
    """Implementación de Embedding usando la API de OpenAI."""
    # (Implementación sin cambios respecto a la anterior)
    # ... (copiar de la versión anterior) ...
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 768, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(dimension)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("DSRAG_OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it during initialization.")
        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url
        self.client = OpenAI(**client_args)

    def get_embeddings(self, text: Union[str, List[str]], input_type: Optional[str] = None) -> Union[Vector, List[Vector]]:
        try:
            input_texts = [text] if isinstance(text, str) else text
            # Limpia textos vacíos que causan error en OpenAI
            cleaned_texts = [t if t and t.strip() else " " for t in input_texts]
            if not cleaned_texts: return [] if isinstance(text, list) else np.zeros(self.dimension).tolist()

            response = self.client.embeddings.create(
                input=cleaned_texts, model=self.model, dimensions=self.dimension
            )
            embeddings = [item.embedding for item in response.data]
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            print(f"Error getting OpenAI embeddings: {e}")
            dim = self.dimension or 1
            if isinstance(text, list):
                return [np.zeros(dim).tolist()] * len(text)
            else:
                return np.zeros(dim).tolist()

    def to_dict(self):
        # No incluye api_key
        return {"subclass_name": "OpenAIEmbedding", "model": self.model, "dimension": self.dimension, "base_url": self.base_url}


class SentenceTransformerEmbedding(Embedding):
    """Implementación de Embedding usando sentence-transformers (Hugging Face)."""
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", dimension: Optional[int] = None, device: Optional[str] = None, **kwargs):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformerEmbedding requires 'sentence-transformers' and 'torch'. Install with 'pip install sentence-transformers torch'")

        # Intenta obtener la dimensión del modelo si no se proporciona
        if dimension is None:
            dimension = LOCAL_DIMENSIONALITY.get(model_name)
            if dimension is None:
                print(f"Warning: Dimension for SentenceTransformer model '{model_name}' not pre-defined. Trying to load model to infer...")
                try:
                    # Carga temporalmente para obtener dimensión (puede ser lento)
                    temp_model = SentenceTransformer(model_name)
                    dimension = temp_model.get_sentence_embedding_dimension()
                    print(f"Inferred dimension for '{model_name}': {dimension}")
                    LOCAL_DIMENSIONALITY[model_name] = dimension
                    del temp_model # Libera memoria
                except Exception as e:
                    raise ValueError(f"Could not infer dimension for '{model_name}' and it was not provided. Error: {e}")

        super().__init__(dimension)
        self.model_name = model_name
        # Determina el dispositivo (GPU si está disponible, si no CPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing SentenceTransformerEmbedding model '{model_name}' on device '{self.device}'...")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            # Valida la dimensión cargada
            loaded_dim = self.model.get_sentence_embedding_dimension()
            if loaded_dim != self.dimension:
                 print(f"Warning: Loaded model dimension ({loaded_dim}) differs from expected dimension ({self.dimension}). Using loaded dimension.")
                 self.dimension = loaded_dim
            print("SentenceTransformer model loaded.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            raise

    def get_embeddings(self, text: Union[str, List[str]], input_type: Optional[str] = None) -> Union[Vector, List[Vector]]:
        """Genera embeddings usando el modelo SentenceTransformer."""
        try:
            # Nota: Algunos modelos E5 usan prefijos ("query: ", "passage: ").
            # Por simplicidad, no añadimos prefijos aquí, pero podría ser necesario
            # para un rendimiento óptimo con modelos E5 específicos.
            # Ejemplo:
            # if input_type == "query" and "e5" in self.model_name:
            #     text = [f"query: {t}" for t in text] if isinstance(text, list) else f"query: {text}"
            # elif input_type == "document" and "e5" in self.model_name:
            #     text = [f"passage: {t}" for t in text] if isinstance(text, list) else f"passage: {text}"

            embeddings = self.model.encode(text, convert_to_numpy=True, device=self.device, normalize_embeddings=True) # Normalizar es común
            # Devuelve ndarray o lista de ndarray
            return embeddings.tolist() # Convierte a lista(s) para consistencia
        except Exception as e:
            print(f"Error generating embeddings with SentenceTransformer model {self.model_name}: {e}")
            dim = self.dimension or 1
            if isinstance(text, list):
                return [np.zeros(dim).tolist()] * len(text)
            else:
                return np.zeros(dim).tolist()

    def to_dict(self):
        """Serializa configuración."""
        return {"subclass_name": "SentenceTransformerEmbedding", "model_name": self.model_name, "dimension": self.dimension, "device": self.device}
