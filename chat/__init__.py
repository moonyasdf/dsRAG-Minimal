# dsrag_minimal/chat/__init__.py
from chat.chat import create_new_chat_thread, get_chat_thread_response
from chat.chat_types import ChatThreadParams, ChatResponseInput, MetadataFilter
from chat.citations import ResponseWithCitations, Citation, PartialResponseWithCitations
from chat.database.db import ChatThreadDB # Importa la clase base de DB

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
