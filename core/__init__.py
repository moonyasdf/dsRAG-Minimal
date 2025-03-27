# dsrag_minimal/core/__init__.py
# (Ya corregido en respuesta anterior, asegúrate de usar esa versión)
# Importa clases clave para facilitar el acceso
from core.knowledge_base import KnowledgeBase
# Incluye SentenceTransformerEmbedding
from core.embedding import Embedding, OllamaEmbedding, OpenAIEmbedding, SentenceTransformerEmbedding
from core.llm import LLM, OllamaLLM, OpenAILLM, AnthropicLLM, get_response_via_instance
# Incluye CrossEncoderReranker
from core.reranker import Reranker, NoReranker, CrossEncoderReranker

__all__ = [
    "KnowledgeBase",
    "Embedding", "OllamaEmbedding", "OpenAIEmbedding", "SentenceTransformerEmbedding",
    "LLM", "OllamaLLM", "OpenAILLM", "AnthropicLLM", "get_response_via_instance",
    "Reranker", "NoReranker", "CrossEncoderReranker",
]
