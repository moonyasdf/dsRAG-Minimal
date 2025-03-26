# dsrag_minimal/chat/database/sqlite_db.py
# (Adaptado de la versión original, pero asegurando que implementa la interfaz ChatThreadDB)
import sqlite3
import os
import json
import uuid
import time
import logging
import contextlib
from typing import List, Dict, Any, Optional, ContextManager

from .db import ChatThreadDB
from ..chat_types import ChatThreadParams

class SQLiteChatThreadDB(ChatThreadDB):
    """Implementación de ChatThreadDB usando SQLite."""

    def __init__(self, storage_directory: str = "~/dsrag_minimal_data"):
        """Inicializa la BD de historial de chat SQLite."""
        # Resuelve path y crea directorio
        self.storage_directory = os.path.expanduser(storage_directory)
        self.db_path_dir = os.path.join(self.storage_directory, "chat_history")
        os.makedirs(self.db_path_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_path_dir, "chat_history.db") # Un solo archivo para todos los hilos

        # Nombres y tipos de columnas (simplificados y usando TEXT para JSON)
        self.threads_cols = {
            "thread_id": "TEXT PRIMARY KEY",
            "supp_id": "TEXT",
            "params": "TEXT", # Almacena ChatThreadParams como JSON string
            "created_ts": "INTEGER",
            "last_updated_ts": "INTEGER"
        }
        self.interactions_cols = {
            "interaction_id": "TEXT PRIMARY KEY",
            "thread_id": "TEXT NOT NULL",
            "user_input": "TEXT",
            "user_ts": "TEXT", # ISO format timestamp
            "model_response": "TEXT",
            "model_ts": "TEXT", # ISO format timestamp
            "search_queries": "TEXT", # JSON string
            "relevant_segments": "TEXT", # JSON string
            "citations": "TEXT", # JSON string
            "FOREIGN KEY": "(thread_id) REFERENCES chat_threads(thread_id) ON DELETE CASCADE" # Borrado en cascada
        }
        self.thread_col_names = list(self.threads_cols.keys())
        self.interaction_col_names = [k for k in self.interactions_cols.keys() if k != 'FOREIGN KEY']

        # Configuración de reintentos
        self.timeout = 30.0
        self.max_retries = 5
        self.retry_delay = 0.5

        # Inicializa tablas
        self._initialize_tables()

    # --- Manejo de Conexión y Reintentos (similar a SQLiteDB para chunks) ---
    @contextlib.contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.execute("PRAGMA foreign_keys = ON;") # Habilita claves foráneas
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _execute_with_retry(self, operation: callable, *args, **kwargs) -> Any:
        # (Implementación idéntica a la de SQLiteDB para chunks, omitida por brevedad)
        # ... (copiar de la implementación anterior) ...
        last_error = None
        delay = self.retry_delay
        for attempt in range(self.max_retries):
            try:
                with self.get_connection() as conn:
                    result = operation(conn, *args, **kwargs)
                    return result
            except sqlite3.OperationalError as e:
                # (Manejo de bloqueo idéntico)
                # ...
                 if "database is locked" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        logging.warning(f"SQLite Chat DB locked (Op: {operation.__name__}), attempt {attempt + 1}/{self.max_retries}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        delay *= 2 # Backoff exponencial
                        continue
                    else:
                        logging.error(f"SQLite Chat DB locked after {self.max_retries} attempts (Op: {operation.__name__}). Giving up.")
                        raise
                 else:
                    logging.error(f"SQLite OperationalError (Chat DB, Op: {operation.__name__}): {e}")
                    raise
            except Exception as e:
                logging.error(f"Error during SQLite Chat DB operation (Op: {operation.__name__}): {e}")
                raise
        raise last_error # Si se agotan los reintentos

    def _initialize_tables(self) -> None:
        """Crea las tablas 'chat_threads' e 'interactions' si no existen."""
        def _init_op(conn: sqlite3.Connection) -> None:
            c = conn.cursor()
            # Crear tabla chat_threads
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_threads'")
            if not c.fetchone():
                cols_defs = [f"\"{k}\" {v}" for k, v in self.threads_cols.items()]
                create_sql = f"CREATE TABLE chat_threads ({', '.join(cols_defs)})"
                c.execute(create_sql)
                c.execute("CREATE INDEX IF NOT EXISTS idx_thread_supp_id ON chat_threads(supp_id)")
                conn.commit()
                logging.info("SQLite table 'chat_threads' created.")

            # Crear tabla interactions
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'")
            if not c.fetchone():
                cols_defs = [f"\"{k}\" {v}" for k, v in self.interactions_cols.items()]
                create_sql = f"CREATE TABLE interactions ({', '.join(cols_defs)})"
                c.execute(create_sql)
                # Índice en thread_id ya cubierto por FOREIGN KEY, pero explícito no hace daño
                c.execute("CREATE INDEX IF NOT EXISTS idx_interaction_thread_id ON interactions(thread_id)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_interaction_user_ts ON interactions(user_ts)") # Para ordenar/filtrar por tiempo
                conn.commit()
                logging.info("SQLite table 'interactions' created.")
            # Podría añadirse lógica para migrar/añadir columnas si las definiciones cambian

        self._execute_with_retry(_init_op)

    # --- Implementación de Métodos de ChatThreadDB ---

    def create_chat_thread(self, chat_thread_params: ChatThreadParams) -> str:
        """Crea un nuevo hilo de chat."""
        thread_id = chat_thread_params.get("thread_id") or str(uuid.uuid4())
        supp_id = chat_thread_params.get("supp_id")
        # Serializa los parámetros a JSON
        params_str = json.dumps(chat_thread_params)
        current_ts = int(time.time())

        def _create_op(conn: sqlite3.Connection) -> None:
            c = conn.cursor()
            sql = f"""
                INSERT INTO chat_threads (thread_id, supp_id, params, created_ts, last_updated_ts)
                VALUES (?, ?, ?, ?, ?)
            """
            c.execute(sql, (thread_id, supp_id, params_str, current_ts, current_ts))
            conn.commit()

        self._execute_with_retry(_create_op)
        return thread_id

    def list_chat_threads(self, supp_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lista hilos de chat."""
        if supp_id:
            sql = "SELECT thread_id, params FROM chat_threads WHERE supp_id = ? ORDER BY last_updated_ts DESC"
            params_tuple = (supp_id,)
        else:
            sql = "SELECT thread_id, params FROM chat_threads ORDER BY last_updated_ts DESC"
            params_tuple = ()

        rows = self._execute_with_retry(lambda conn, q, p: conn.cursor().execute(q, p).fetchall(), sql, params_tuple)

        threads = []
        for row in rows:
            thread_id, params_str = row
            try:
                params_dict = json.loads(params_str)
                # Asegura que el formato devuelto sea consistente {id: ..., params: ...}
                threads.append({"id": thread_id, "params": params_dict})
            except json.JSONDecodeError:
                logging.warning(f"Could not decode params for thread_id {thread_id}")
                # Devuelve con params vacíos o como string si falla el parseo
                threads.append({"id": thread_id, "params": {}})
        return threads

    def get_chat_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un hilo completo (params + interacciones)."""
        # Obtener parámetros
        sql_params = "SELECT params FROM chat_threads WHERE thread_id = ?"
        params_rows = self._execute_with_retry(lambda conn, q, p: conn.cursor().execute(q, p).fetchall(), sql_params, (thread_id,))

        if not params_rows:
            return None

        try:
            params_dict = json.loads(params_rows[0][0])
        except json.JSONDecodeError:
            logging.warning(f"Could not decode params for thread_id {thread_id}")
            params_dict = {}

        # Obtener interacciones
        interactions = self.get_interactions(thread_id) # Usa el método get_interactions

        return {
            "id": thread_id,
            "params": params_dict,
            "interactions": interactions
        }

    def update_chat_thread_params(self, thread_id: str, chat_thread_params: ChatThreadParams) -> bool:
        """Actualiza los parámetros de un hilo."""
        params_str = json.dumps(chat_thread_params)
        current_ts = int(time.time())

        def _update_op(conn: sqlite3.Connection) -> int:
            c = conn.cursor()
            # Actualiza también supp_id por si cambia en los params
            supp_id = chat_thread_params.get("supp_id")
            sql = """
                UPDATE chat_threads
                SET params = ?, supp_id = ?, last_updated_ts = ?
                WHERE thread_id = ?
            """
            c.execute(sql, (params_str, supp_id, current_ts, thread_id))
            conn.commit()
            return c.rowcount # Devuelve número de filas afectadas

        rows_affected = self._execute_with_retry(_update_op)
        return rows_affected > 0

    def delete_chat_thread(self, thread_id: str) -> bool:
        """Elimina un hilo y sus interacciones (ON DELETE CASCADE)."""
        def _delete_op(conn: sqlite3.Connection) -> int:
            c = conn.cursor()
            sql = "DELETE FROM chat_threads WHERE thread_id = ?"
            c.execute(sql, (thread_id,))
            conn.commit()
            return c.rowcount

        rows_affected = self._execute_with_retry(_delete_op)
        return rows_affected > 0

    def add_interaction(self, thread_id: str, interaction: Dict[str, Any]) -> Optional[str]:
        """Añade una interacción a un hilo."""
        interaction_id = interaction.get("message_id") or str(uuid.uuid4()) # Usa ID si existe, si no genera uno
        current_ts = int(time.time()) # Timestamp para actualizar el hilo

        # Extrae y serializa datos
        user_input = interaction.get("user_input", {}).get("content", "")
        user_ts = interaction.get("user_input", {}).get("timestamp", "")
        model_response = interaction.get("model_response", {}).get("content", "")
        model_ts = interaction.get("model_response", {}).get("timestamp", "")
        search_queries_str = json.dumps(interaction.get("search_queries", []))
        relevant_segments_str = json.dumps(interaction.get("relevant_segments", []))
        citations_str = json.dumps(interaction.get("model_response", {}).get("citations", []))

        def _add_interaction_op(conn: sqlite3.Connection) -> None:
            c = conn.cursor()
            # Inserta la interacción
            interaction_sql = f"""
                INSERT INTO interactions (interaction_id, thread_id, user_input, user_ts,
                                         model_response, model_ts, search_queries,
                                         relevant_segments, citations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            interaction_values = (interaction_id, thread_id, user_input, user_ts,
                                  model_response, model_ts, search_queries_str,
                                  relevant_segments_str, citations_str)
            c.execute(interaction_sql, interaction_values)

            # Actualiza el timestamp del hilo
            thread_update_sql = "UPDATE chat_threads SET last_updated_ts = ? WHERE thread_id = ?"
            c.execute(thread_update_sql, (current_ts, thread_id))
            conn.commit()

        try:
            self._execute_with_retry(_add_interaction_op)
            # Devuelve el interaction dict original (que ahora puede incluir message_id si se generó)
            interaction["message_id"] = interaction_id
            return interaction # Devuelve el dict completo de la interacción añadida
        except Exception as e:
             logging.error(f"Failed to add interaction to thread {thread_id}: {e}")
             return None # Opcional: devolver None en fallo


    def get_interactions(self, thread_id: str, limit: Optional[int] = None, before_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene interacciones de un hilo."""
        base_sql = "SELECT * FROM interactions WHERE thread_id = ?"
        params: List[Any] = [thread_id]

        if before_timestamp:
            base_sql += " AND user_ts < ?"
            params.append(before_timestamp)

        base_sql += " ORDER BY user_ts ASC" # Ordena por timestamp de usuario

        if limit:
            base_sql += " LIMIT ?"
            params.append(limit)

        rows = self._execute_with_retry(lambda conn, q, p: conn.cursor().execute(q, tuple(p)).fetchall(), base_sql, params)

        interactions = []
        for row_tuple in rows:
             # Convierte la tupla a un diccionario usando los nombres de columna
             row = dict(zip(self.interaction_col_names, row_tuple))
             try:
                 # Deserializa los campos JSON
                 search_queries = json.loads(row.get("search_queries", "[]"))
                 relevant_segments = json.loads(row.get("relevant_segments", "[]"))
                 citations = json.loads(row.get("citations", "[]"))

                 formatted_interaction = {
                     "message_id": row.get("interaction_id"), # Usa el ID guardado
                     "user_input": {
                         "content": row.get("user_input"),
                         "timestamp": row.get("user_ts")
                     },
                     "model_response": {
                         "content": row.get("model_response"),
                         "timestamp": row.get("model_ts"),
                         "citations": citations
                     },
                     "search_queries": search_queries,
                     "relevant_segments": relevant_segments
                 }
                 interactions.append(formatted_interaction)
             except json.JSONDecodeError as e:
                 logging.warning(f"Could not decode JSON fields for interaction {row.get('interaction_id')} in thread {thread_id}: {e}")
                 # Podría omitir la interacción o incluirla con campos JSON vacíos/nulos
             except Exception as e:
                 logging.error(f"Unexpected error processing interaction {row.get('interaction_id')}: {e}")

        return interactions

    def to_dict(self) -> Dict[str, str]:
        """Serializa configuración."""
        return {
            "subclass_name": "SQLiteChatThreadDB",
            "storage_directory": self.storage_directory
        }