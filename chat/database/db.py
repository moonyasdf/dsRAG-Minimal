# dsrag_minimal/chat/database/db.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
# Usa ruta relativa correcta a chat_types
from chat.chat_types import ChatThreadParams

class ChatThreadDB(ABC):
    """Clase base abstracta para almacenar historiales de chat."""

    @abstractmethod
    def create_chat_thread(self, chat_thread_params: ChatThreadParams) -> str:
        """
        Crea un nuevo hilo de chat y devuelve su ID único.
        Args:
            chat_thread_params: Diccionario con los parámetros iniciales del hilo.
        Returns:
            El ID del hilo creado.
        """
        pass

    @abstractmethod
    def list_chat_threads(self, supp_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lista todos los hilos de chat, opcionalmente filtrados por supp_id.
        Devuelve una lista de diccionarios, cada uno representando un hilo (id, params).
        """
        pass

    @abstractmethod
    def get_chat_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un hilo de chat completo por su ID, incluyendo parámetros e interacciones.
        Devuelve None si no se encuentra.
        """
        pass

    @abstractmethod
    def update_chat_thread_params(self, thread_id: str, chat_thread_params: ChatThreadParams) -> bool:
        """
        Actualiza los parámetros de un hilo de chat existente.
        Devuelve True si tuvo éxito, False si el hilo no existe.
        """
        pass

    @abstractmethod
    def delete_chat_thread(self, thread_id: str) -> bool:
        """
        Elimina un hilo de chat y todas sus interacciones.
        Devuelve True si tuvo éxito, False si el hilo no existe.
        """
        pass

    @abstractmethod
    def add_interaction(self, thread_id: str, interaction: Dict[str, Any]) -> Optional[str]:
        """
        Añade una nueva interacción (par usuario-respuesta) a un hilo existente.
        Devuelve el ID único de la interacción añadida, o None si falla.
        """
        pass

    @abstractmethod
    def get_interactions(self, thread_id: str, limit: Optional[int] = None, before_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene las interacciones de un hilo, opcionalmente limitadas o hasta cierto timestamp.
        Devuelve una lista de interacciones ordenadas cronológicamente (la más reciente al final).
        """
        pass
