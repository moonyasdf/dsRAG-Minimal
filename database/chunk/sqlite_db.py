# dsrag_minimal/database/chunk/sqlite_db.py
# (Manteniendo la implementación original de SQLite, ya que es local y eficiente)
# Asegúrate de que las importaciones y dependencias sean mínimas.

import os
import time
import sqlite3
from typing import Any, Optional, ContextManager, Dict, List, Tuple
import contextlib
import logging
import json # Importa json para serializar/deserializar metadata y rse_params

from database.chunk.db import ChunkDB # Importa la clase base simplificada
from database.chunk.types import FormattedDocument # Importa tipos

class SQLiteDB(ChunkDB):
    """Implementación de ChunkDB usando SQLite para almacenamiento local."""

    def __init__(self, kb_id: str, storage_directory: str = "~/dsRAG") -> None:
        """Inicializa la base de datos SQLite."""
        self.kb_id = kb_id
        # Resuelve y crea directorios de almacenamiento
        self.storage_directory = os.path.expanduser(storage_directory)
        self.db_path_dir = os.path.join(self.storage_directory, "chunk_storage")
        os.makedirs(self.db_path_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_path_dir, f"{kb_id}.db")

        # Define las columnas de la tabla
        self.columns = [
            # Identificadores y metadatos clave
            {"name": "doc_id", "type": "TEXT NOT NULL"},
            {"name": "chunk_index", "type": "INTEGER NOT NULL"},
            {"name": "supp_id", "type": "TEXT"}, # Identificador suplementario opcional
            {"name": "metadata", "type": "TEXT"}, # Metadatos adicionales (JSON string)
            {"name": "created_on", "type": "INTEGER"}, # Timestamp de creación (Unix epoch)
            # Contenido y contexto
            {"name": "chunk_text", "type": "TEXT"},
            {"name": "document_title", "type": "TEXT"},
            {"name": "document_summary", "type": "TEXT"},
            {"name": "section_title", "type": "TEXT"},
            {"name": "section_summary", "type": "TEXT"},
            # Información de paginación y tipo
            {"name": "chunk_page_start", "type": "INTEGER"},
            {"name": "chunk_page_end", "type": "INTEGER"},
            {"name": "is_visual", "type": "INTEGER"}, # Almacenar booleanos como 0 o 1
            # Información calculada
            {"name": "chunk_length", "type": "INTEGER"},
            # Clave primaria compuesta
            {"name": "PRIMARY KEY", "type": "(doc_id, chunk_index)"}
        ]
        self.column_names = [col['name'] for col in self.columns if col['name'] != 'PRIMARY KEY']

        # Configuración de reintentos para bloqueos
        self.timeout = 30.0  # Aumentado el timeout
        self.max_retries = 5 # Aumentado los reintentos
        self.retry_delay = 0.5 # Reducido el delay inicial

        # Crea la tabla y migra si es necesario
        self._initialize_table()

    @contextlib.contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        """Obtiene una conexión a la base de datos con manejo de timeout."""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.execute("PRAGMA journal_mode=WAL;") # Mejora la concurrencia
        try:
            yield conn
        finally:
            conn.close()

    def _execute_with_retry(self, operation: callable, *args, **kwargs) -> Any:
        """Ejecuta una operación de BD con reintentos exponenciales."""
        last_error = None
        delay = self.retry_delay
        for attempt in range(self.max_retries):
            try:
                with self.get_connection() as conn:
                    # Habilita ejecución en modo exclusivo temporalmente para escrituras si es necesario
                    # conn.execute('BEGIN EXCLUSIVE') # Descomentar si aún hay problemas de bloqueo
                    result = operation(conn, *args, **kwargs)
                    # conn.commit() # Asegura commit si la operación no lo hizo
                    return result
            except sqlite3.OperationalError as e:
                last_error = e
                if "database is locked" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        logging.warning(f"SQLite DB locked (KB: {self.kb_id}, Op: {operation.__name__}), attempt {attempt + 1}/{self.max_retries}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        delay *= 2 # Backoff exponencial
                        continue
                    else:
                        logging.error(f"SQLite DB locked after {self.max_retries} attempts (KB: {self.kb_id}, Op: {operation.__name__}). Giving up.")
                        raise # Relanza el error después de los reintentos
                else:
                    # Si el error no es de bloqueo, relanzar inmediatamente
                    logging.error(f"SQLite OperationalError (KB: {self.kb_id}, Op: {operation.__name__}): {e}")
                    raise
            except Exception as e:
                logging.error(f"Error during SQLite operation (KB: {self.kb_id}, Op: {operation.__name__}): {e}")
                raise # Relanza otras excepciones

        # Si se agotan los reintentos por bloqueo
        raise last_error

    def _initialize_table(self) -> None:
        """Crea o actualiza la tabla 'chunks'."""
        def _init_op(conn: sqlite3.Connection) -> None:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
            table_exists = c.fetchone()

            if not table_exists:
                # Crear tabla con clave primaria
                cols_defs = [f"\"{col['name']}\" {col['type']}" for col in self.columns]
                create_sql = f"CREATE TABLE chunks ({', '.join(cols_defs)})"
                c.execute(create_sql)
                # Crear índice en supp_id para búsquedas más rápidas
                c.execute("CREATE INDEX IF NOT EXISTS idx_supp_id ON chunks(supp_id)")
                conn.commit()
                logging.info(f"SQLite table 'chunks' created for kb_id: {self.kb_id}")
            else:
                # Verificar y añadir columnas faltantes
                c.execute("PRAGMA table_info(chunks)")
                existing_columns = {row[1] for row in c.fetchall()}
                for column in self.columns:
                    col_name = column['name']
                    if col_name != 'PRIMARY KEY' and col_name not in existing_columns:
                        try:
                            c.execute(f"ALTER TABLE chunks ADD COLUMN \"{col_name}\" {column['type']}")
                            conn.commit()
                            logging.info(f"Added missing column '{col_name}' to 'chunks' table for kb_id: {self.kb_id}")
                        except sqlite3.OperationalError as e:
                            # Ignora error si la columna ya existe (puede pasar en ejecuciones concurrentes)
                            if "duplicate column name" not in str(e).lower():
                                raise

        self._execute_with_retry(_init_op)


    def add_document(self, doc_id: str, chunks: Dict[int, Dict[str, Any]], supp_id: str = "", metadata: dict = {}) -> None:
        """Añade los chunks de un documento a la tabla SQLite."""
        def _add_doc_op(conn: sqlite3.Connection, data_to_insert: list) -> None:
            c = conn.cursor()
            # Construye la parte de los nombres de columna por separado
            column_names_str = '", "'.join(self.column_names)
            # Construye la parte de los placeholders
            placeholders_str = ', '.join(['?'] * len(self.column_names))
            # Usa las variables pre-construidas en la f-string
            sql = f'REPLACE INTO chunks ("{column_names_str}") VALUES ({placeholders_str})'
            c.executemany(sql, data_to_insert)
            conn.commit()

        insert_batch = []
        created_on_ts = int(time.time())
        metadata_str = json.dumps(metadata) if metadata else "{}" # Serializa metadata a JSON

        for chunk_index, chunk_data in chunks.items():
            chunk_text = chunk_data.get("chunk_text", "")
            is_visual_int = 1 if chunk_data.get("is_visual", False) else 0 # Convierte bool a int

            values = {
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "supp_id": supp_id or None, # Usa None para NULL en SQLite
                "metadata": metadata_str,
                "created_on": created_on_ts,
                "chunk_text": chunk_text,
                "document_title": chunk_data.get("document_title") or None,
                "document_summary": chunk_data.get("document_summary") or None,
                "section_title": chunk_data.get("section_title") or None,
                "section_summary": chunk_data.get("section_summary") or None,
                "chunk_page_start": chunk_data.get("chunk_page_start"), # Puede ser None
                "chunk_page_end": chunk_data.get("chunk_page_end"),     # Puede ser None
                "is_visual": is_visual_int,
                "chunk_length": len(chunk_text),
            }
            # Asegura el orden correcto de los valores según self.column_names
            ordered_values = tuple(values.get(name) for name in self.column_names)
            insert_batch.append(ordered_values)

        if insert_batch:
            self._execute_with_retry(_add_doc_op, insert_batch)

    def remove_document(self, doc_id: str) -> None:
        """Elimina todos los chunks de un documento."""
        def _remove_doc_op(conn: sqlite3.Connection, doc_id: str) -> None:
            c = conn.cursor()
            c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.commit()
            # Opcional: Ejecutar VACUUM para reducir tamaño del archivo si se eliminan muchos datos
            # c.execute("VACUUM")
            # conn.commit()

        self._execute_with_retry(_remove_doc_op, doc_id)

    def _get_rows(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Helper para ejecutar consultas SELECT."""
        def _select_op(conn: sqlite3.Connection, query: str, params: tuple) -> List[sqlite3.Row]:
            conn.row_factory = sqlite3.Row # Devuelve filas como diccionarios
            c = conn.cursor()
            c.execute(query, params)
            return c.fetchall()

        return self._execute_with_retry(_select_op, query, params)

    def get_document(self, doc_id: str, include_content: bool = False) -> Optional[FormattedDocument]:
        """Obtiene metadatos y opcionalmente contenido de un documento."""
        # Selecciona columnas necesarias, ordena por chunk_index para reconstruir contenido
        columns_to_select = ["supp_id", "document_title", "document_summary", "created_on", "metadata"]
        if include_content:
            columns_to_select.extend(["chunk_text", "chunk_index"])

        # Construye la cadena de selección de columnas por separado
        select_cols_str = '", "'.join(columns_to_select)
        # Usa la variable en la f-string
        sql = f'SELECT "{select_cols_str}" FROM chunks WHERE doc_id = ? ORDER BY chunk_index'
        rows = self._get_rows(sql, (doc_id,))

        if not rows:
            return None

        first_row = rows[0]
        content = None
        if include_content:
            # Reconstruye el contenido ordenando por chunk_index (ya ordenado por SQL)
            content = "\n".join(row["chunk_text"] for row in rows)

        # Deserializa metadata
        metadata_dict = {}
        if first_row["metadata"]:
            try:
                metadata_dict = json.loads(first_row["metadata"])
            except json.JSONDecodeError:
                logging.warning(f"Could not decode metadata for doc_id {doc_id}")

        return FormattedDocument(
            id=doc_id,
            supp_id=first_row["supp_id"],
            title=first_row["document_title"],
            content=content,
            summary=first_row["document_summary"],
            # Convierte timestamp de vuelta si es necesario (aquí se devuelve como int)
            created_on=first_row["created_on"],
            metadata=metadata_dict,
            chunk_count=len(rows)
        )

    def _get_single_chunk_field(self, doc_id: str, chunk_index: int, field_name: str) -> Optional[Any]:
        """Helper para obtener un campo específico de un chunk."""
        sql = f"SELECT \"{field_name}\" FROM chunks WHERE doc_id = ? AND chunk_index = ?"
        rows = self._get_rows(sql, (doc_id, chunk_index))
        if rows:
            return rows[0][field_name]
        return None

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        """Obtiene el texto de un chunk específico."""
        return self._get_single_chunk_field(doc_id, chunk_index, "chunk_text")

    def get_is_visual(self, doc_id: str, chunk_index: int) -> Optional[bool]:
        """Obtiene el flag 'is_visual' de un chunk."""
        is_visual_int = self._get_single_chunk_field(doc_id, chunk_index, "is_visual")
        return bool(is_visual_int) if is_visual_int is not None else None

    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Tuple[Optional[int], Optional[int]]:
        """Obtiene las páginas de inicio y fin de un chunk."""
        sql = "SELECT chunk_page_start, chunk_page_end FROM chunks WHERE doc_id = ? AND chunk_index = ?"
        rows = self._get_rows(sql, (doc_id, chunk_index))
        if rows:
            return rows[0]["chunk_page_start"], rows[0]["chunk_page_end"]
        return None, None

    # Métodos para obtener metadatos específicos (title, summary, etc.)
    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        return self._get_single_chunk_field(doc_id, chunk_index, "document_title")

    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        return self._get_single_chunk_field(doc_id, chunk_index, "document_summary")

    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        return self._get_single_chunk_field(doc_id, chunk_index, "section_title")

    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        return self._get_single_chunk_field(doc_id, chunk_index, "section_summary")

    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> List[str]:
        """Obtiene todos los doc_ids únicos, opcionalmente filtrados por supp_id."""
        if supp_id:
            sql = "SELECT DISTINCT doc_id FROM chunks WHERE supp_id = ?"
            params = (supp_id,)
        else:
            sql = "SELECT DISTINCT doc_id FROM chunks"
            params = ()
        rows = self._get_rows(sql, params)
        return [row["doc_id"] for row in rows]

    def get_document_count(self) -> int:
        """Obtiene el número de documentos únicos."""
        sql = "SELECT COUNT(DISTINCT doc_id) FROM chunks"
        rows = self._get_rows(sql)
        return rows[0][0] if rows else 0

    def get_total_num_characters(self) -> int:
        """Calcula el número total de caracteres almacenados."""
        sql = "SELECT SUM(chunk_length) FROM chunks"
        rows = self._get_rows(sql)
        return rows[0][0] or 0 # Devuelve 0 si la tabla está vacía

    def delete(self) -> None:
        """Elimina el archivo de la base de datos SQLite."""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                logging.info(f"Deleted SQLite database file: {self.db_path}")
        except OSError as e:
            logging.error(f"Error deleting SQLite database file {self.db_path}: {e}")

    def to_dict(self) -> Dict[str, str]:
        """Serializa la configuración de SQLiteDB."""
        return {
            "subclass_name": "SQLiteDB",
            "kb_id": self.kb_id,
            "storage_directory": self.storage_directory,
            # No se guardan las columnas, se definen en __init__
        }
