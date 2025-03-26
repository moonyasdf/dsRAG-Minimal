# dsrag_minimal/chat/chat_types.py
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field

# Parámetros de configuración para un hilo de chat
class ChatThreadParams(BaseModel):
    # IDs de las bases de conocimiento a usar
    kb_ids: List[str] = []
    # Nombre del modelo LLM para generar respuestas
    model: str = "llama3" # Default local
    # Temperatura para la generación LLM (creatividad)
    temperature: float = 0.1
    # Mensaje de sistema personalizado para el LLM
    system_message: str = ""
    # Nombre del modelo LLM para generar queries (puede ser el mismo)
    auto_query_model: str = "llama3" # Default local
    # Guía adicional para la generación de queries
    auto_query_guidance: str = ""
    # Parámetros para Relevant Segment Extraction (RSE)
    rse_params: Union[Dict[str, Any], str] = "balanced" # Preset o dict
    # Longitud deseada de la respuesta ("short", "medium", "long")
    target_output_length: Literal["short", "medium", "long"] = "medium"
    # Máximo de tokens a mantener en el historial de chat para contexto
    max_chat_history_tokens: int = 4000 # Reducido para modelos locales
    # ID del hilo (generalmente auto-generado)
    thread_id: Optional[str] = None
    # ID suplementario opcional para agrupar/filtrar hilos
    supp_id: Optional[str] = None

# Filtro para búsquedas en la base de datos vectorial
class MetadataFilter(BaseModel):
    field: str = Field(..., description="Campo de metadatos para filtrar (ej: 'doc_id')")
    operator: Literal['equals', 'not_equals', 'in', 'not_in', 'greater_than', 'less_than', 'greater_than_equals', 'less_than_equals'] = Field(..., description="Operador de comparación")
    value: Union[str, int, float, List[str], List[int], List[float]] = Field(..., description="Valor(es) para la comparación")

# Entrada para la función get_chat_thread_response
class ChatResponseInput(BaseModel):
    # Mensaje del usuario
    user_input: str
    # Opcional: Sobrescribe parámetros del hilo para esta interacción
    chat_thread_params: Optional[ChatThreadParams] = None
    # Opcional: Filtro para la búsqueda vectorial
    metadata_filter: Optional[MetadataFilter] = None

# (Se mantienen Citation y ResponseWithCitations de citations.py)
