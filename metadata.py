# dsrag_minimal/metadata.py
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

class MetadataStorage(ABC):
    """Clase base abstracta para almacenar metadatos del KB."""
    @abstractmethod
    def kb_exists(self, kb_id: str) -> bool:
        pass
    @abstractmethod
    def load(self, kb_id: str) -> Dict[str, Any]:
        pass
    @abstractmethod
    def save(self, data: Dict[str, Any], kb_id: str) -> None:
        pass
    @abstractmethod
    def delete(self, kb_id: str) -> None:
        pass

class LocalMetadataStorage(MetadataStorage):
    """Almacena metadatos como archivos JSON locales."""
    def __init__(self, storage_directory: str):
        self.storage_directory = os.path.expanduser(storage_directory)
        self.metadata_dir = os.path.join(self.storage_directory, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True) # Crea directorio si no existe

    def _get_metadata_path(self, kb_id: str) -> str:
        """Obtiene la ruta al archivo de metadatos."""
        # Usa un nombre de archivo seguro (reemplaza caracteres inválidos si es necesario)
        safe_kb_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in kb_id)
        return os.path.join(self.metadata_dir, f"{safe_kb_id}.json")

    def kb_exists(self, kb_id: str) -> bool:
        """Verifica si existe el archivo de metadatos."""
        return os.path.exists(self._get_metadata_path(kb_id))

    def load(self, kb_id: str) -> Dict[str, Any]:
        """Carga metadatos desde archivo JSON."""
        filepath = self._get_metadata_path(kb_id)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metadata file not found for kb_id: {kb_id}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding metadata JSON for {kb_id}: {e}")
            raise # Relanza el error de parseo
        except Exception as e:
            logging.error(f"Error loading metadata for {kb_id}: {e}")
            raise

    def save(self, data: Dict[str, Any], kb_id: str) -> None:
        """Guarda metadatos en archivo JSON."""
        filepath = self._get_metadata_path(kb_id)
        try:
            # Guarda con indentación para legibilidad humana
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving metadata for {kb_id}: {e}")
            raise

    def delete(self, kb_id: str) -> None:
        """Elimina el archivo de metadatos."""
        filepath = self._get_metadata_path(kb_id)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info(f"Deleted metadata file for {kb_id}: {filepath}")
        except OSError as e:
            logging.error(f"Error deleting metadata file for {kb_id}: {e}")
            # No relanza, solo loguea el error