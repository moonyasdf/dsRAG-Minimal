# dsrag_minimal/chat/__init__.py
from .chat import create_new_chat_thread, get_chat_thread_response
from .chat_types import ChatThreadParams, ChatResponseInput, MetadataFilter
from .citations import ResponseWithCitations, Citation, PartialResponseWithCitations
from .db import ChatThreadDB # Importa la clase base de DB
# No exportamos la implementación SQLite por defecto desde aquí

__all__ = [
    'create_new_chat_thread',
    'get_chat_thread_response',
    'ChatThreadParams',
    'ChatResponseInput',
    'MetadataFilter',
    'ResponseWithCitations',
    'Citation',
    'PartialResponseWithCitations',
    'ChatThreadDB',
]
