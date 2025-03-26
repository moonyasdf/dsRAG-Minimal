# dsrag_minimal/dsparse/file_parsing/file_system.py
# (Solo clase base y LocalFileSystem)
import os
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class FileSystem(ABC):
    """Clase base abstracta para interactuar con el sistema de archivos."""
    SUBCLASSES = {} # Registro

    def __init__(self, base_path: str):
        # Resuelve el path base y asegura que exista si es local
        self.base_path = os.path.expanduser(base_path)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.SUBCLASSES[cls.__name__] = cls

    @abstractmethod
    def create_directory(self, kb_id: str, doc_id: str) -> None:
        """Crea (o limpia) el directorio para un documento."""
        pass

    @abstractmethod
    def delete_directory(self, kb_id: str, doc_id: str) -> None:
        """Elimina el directorio de un documento."""
        pass

    @abstractmethod
    def delete_kb(self, kb_id: str) -> None:
        """Elimina todos los directorios asociados a un KB."""
        pass

    @abstractmethod
    def save_json(self, kb_id: str, doc_id: str, file_name: str, data: Dict[str, Any]) -> None:
        """Guarda datos JSON."""
        pass

    @abstractmethod
    def load_data(self, kb_id: str, doc_id: str, data_name: str) -> Optional[Dict[str, Any]]:
        """Carga datos JSON."""
        pass

    @abstractmethod
    def save_page_content(self, kb_id: str, doc_id: str, page_number: int, content: str) -> None:
        """Guarda el contenido textual de una página."""
        pass

    @abstractmethod
    def load_page_content(self, kb_id: str, doc_id: str, page_number: int) -> Optional[str]:
        """Carga el contenido textual de una página."""
        pass

    @abstractmethod
    def load_page_content_range(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> List[str]:
        """Carga contenido textual para un rango de páginas."""
        pass

    # Métodos relacionados con imágenes VLM (pueden dejarse como stubs o eliminarse si no hay parseo con imágenes)
    def save_image(self, kb_id: str, doc_id: str, file_name: str, image: Any) -> None:
         """Guarda una imagen (requiere PIL.Image). Placeholder."""
         print("Warning: save_image called but VLM is disabled. Image not saved.")
         pass

    def get_files(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> List[str]:
         """Obtiene rutas de imágenes. Placeholder."""
         print("Warning: get_files called but VLM is disabled. Returning empty list.")
         return []

    def get_all_png_files(self, kb_id: str, doc_id: str) -> List[str]:
         """Obtiene todas las imágenes PNG. Placeholder."""
         print("Warning: get_all_png_files called but VLM is disabled. Returning empty list.")
         return []


    def to_dict(self) -> Dict[str, Any]:
        return {"subclass_name": self.__class__.__name__, "base_path": self.base_path}

    @classmethod
    def from_dict(cls, config: dict) -> 'FileSystem':
        subclass_name = config.get("subclass_name")
        if subclass_name == "LocalFileSystem":
             return LocalFileSystem(**{k:v for k,v in config.items() if k != 'subclass_name'})
        else:
             raise ValueError(f"Unknown FileSystem subclass: {subclass_name}")


class LocalFileSystem(FileSystem):
    """Implementación de FileSystem usando el sistema de archivos local."""
    def __init__(self, base_path: str):
        super().__init__(base_path)
        # Asegura que el directorio base exista
        os.makedirs(self.base_path, exist_ok=True)

    def _get_doc_path(self, kb_id: str, doc_id: str) -> str:
        """Obtiene la ruta al directorio del documento."""
        return os.path.join(self.base_path, kb_id, doc_id)

    def _get_file_path(self, kb_id: str, doc_id: str, file_name: str) -> str:
        """Obtiene la ruta completa a un archivo dentro del dir del documento."""
        return os.path.join(self._get_doc_path(kb_id, doc_id), file_name)

    def create_directory(self, kb_id: str, doc_id: str) -> None:
        doc_path = self._get_doc_path(kb_id, doc_id)
        # Limpia el directorio si existe, luego lo crea
        if os.path.exists(doc_path):
            import shutil
            shutil.rmtree(doc_path)
        os.makedirs(doc_path)

    def delete_directory(self, kb_id: str, doc_id: str) -> None:
        doc_path = self._get_doc_path(kb_id, doc_id)
        if os.path.isdir(doc_path):
            import shutil
            shutil.rmtree(doc_path, ignore_errors=True)

    def delete_kb(self, kb_id: str) -> None:
        kb_path = os.path.join(self.base_path, kb_id)
        if os.path.isdir(kb_path):
            import shutil
            shutil.rmtree(kb_path, ignore_errors=True)

    def save_json(self, kb_id: str, doc_id: str, file_name: str, data: Dict[str, Any]) -> None:
        filepath = self._get_file_path(kb_id, doc_id, file_name)
        try:
            # Asegura que el directorio exista antes de escribir
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving JSON to {filepath}: {e}")
            raise

    def load_data(self, kb_id: str, doc_id: str, data_name: str) -> Optional[Dict[str, Any]]:
        filepath = self._get_file_path(kb_id, doc_id, f"{data_name}.json")
        if not os.path.exists(filepath):
            # print(f"Data file not found: {filepath}")
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
             print(f"Error decoding JSON from {filepath}")
             return None
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return None

    def save_page_content(self, kb_id: str, doc_id: str, page_number: int, content: str) -> None:
        """Guarda contenido de página como JSON."""
        file_name = f"page_content_{page_number}.json"
        self.save_json(kb_id, doc_id, file_name, {"content": content})

    def load_page_content(self, kb_id: str, doc_id: str, page_number: int) -> Optional[str]:
        """Carga contenido de página desde JSON."""
        data = self.load_data(kb_id, doc_id, f"page_content_{page_number}")
        return data.get("content") if data else None

    def load_page_content_range(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> List[str]:
        """Carga un rango de contenido de páginas."""
        contents = []
        if page_start is None or page_end is None: return []
        for i in range(page_start, page_end + 1):
            content = self.load_page_content(kb_id, doc_id, i)
            if content is not None:
                contents.append(content)
            # else:
                 # Podríamos añadir un placeholder o aviso si falta una página
                 # contents.append(f"[Content for page {i} not found]")
        return contents
