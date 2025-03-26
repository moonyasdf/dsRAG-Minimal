# dsrag_minimal/__init__.py
from .core import KnowledgeBase, Embedding, LLM, Reranker
from .chat import create_new_chat_thread, get_chat_thread_response, ChatThreadParams, ChatResponseInput
from .create_kb import create_kb_from_file
# Opcional: Importar implementaciones específicas si se quieren exponer directamente
from .core.embedding import OllamaEmbedding, OpenAIEmbedding
from .core.llm import OllamaLLM, OpenAILLM, AnthropicLLM
from .core.reranker import NoReranker
from .database.chunk.sqlite_db import SQLiteDB
from .database.vector.qdrant_db import QdrantVectorDB
from .chat.database.sqlite_db import SQLiteChatThreadDB

__version__ = "0.1.0-minimal" # Ejemplo de versión

__all__ = [
    "KnowledgeBase",
    "Embedding", "OllamaEmbedding", "OpenAIEmbedding",
    "LLM", "OllamaLLM", "OpenAILLM", "AnthropicLLM",
    "Reranker", "NoReranker",
    "create_new_chat_thread", "get_chat_thread_response",
    "ChatThreadParams", "ChatResponseInput",
    "create_kb_from_file",
    "SQLiteDB", # Expone implementación concreta de chunk DB
    "QdrantVectorDB", # Expone implementación concreta de vector DB
    "SQLiteChatThreadDB", # Expone implementación concreta de chat history DB
]