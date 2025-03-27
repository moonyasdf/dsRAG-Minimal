# dsrag_minimal/__init__.py
__version__ = "0.1.0-minimal"

# Core components
from .core import (
    KnowledgeBase,
    Embedding, OllamaEmbedding, OpenAIEmbedding, SentenceTransformerEmbedding,
    LLM, OllamaLLM, OpenAILLM, AnthropicLLM, get_response_via_instance,
    Reranker, NoReranker, JinaReranker
)

# Database implementations (specific)
from .database.vector.qdrant_db import QdrantVectorDB
from .database.chunk.sqlite_db import SQLiteDB

# Chat functionality
from .chat import (
    create_new_chat_thread, get_chat_thread_response,
    ChatThreadParams, ChatResponseInput,
    ResponseWithCitations, Citation # Re-export from citations
)
# Chat History DB implementation (specific)
from .chat.database.sqlite_db import SQLiteChatThreadDB

# Helper functions
from .create_kb import create_kb_from_file
from .metadata import LocalMetadataStorage, MetadataStorage

# dsparse main function (opcional, si se quiere usar directamente)
# from .dsparse import parse_and_chunk

__all__ = [
    # Core
    "KnowledgeBase",
    "Embedding", "OllamaEmbedding", "OpenAIEmbedding", "SentenceTransformerEmbedding",
    "LLM", "OllamaLLM", "OpenAILLM", "AnthropicLLM", "get_response_via_instance",
    "Reranker", "NoReranker", "JinaReranker",
    # Databases
    "QdrantVectorDB", "SQLiteDB",
    # Chat
    "create_new_chat_thread", "get_chat_thread_response",
    "ChatThreadParams", "ChatResponseInput",
    "ResponseWithCitations", "Citation",
    "SQLiteChatThreadDB",
    # Helpers & Metadata
    "create_kb_from_file",
    "MetadataStorage", "LocalMetadataStorage",
    # Version
    "__version__",
]
