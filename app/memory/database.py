"""
SQLite database setup — async aiosqlite/libsql with table creation on startup.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import aiosqlite
from app.config import get_settings


class LibsqlRow(dict):
    """aiosqlite.Row compatible row wrapper for libsql-client."""
    def __init__(self, colnames, row_data):
        super().__init__(zip(colnames, row_data))
        self._list = list(row_data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._list[key]
        return super().__getitem__(key)


class LibsqlCursorWrapper:
    """Cursor-like wrapper for remote libsql execution. Supports both awaiting and async with."""
    def __init__(self, connection, client, sql, parameters):
        self.connection = connection
        self.client = client
        self.sql = sql
        self.parameters = parameters
        self.result_set = None
        self.row_factory = connection.row_factory
        self._index = 0

    def __await__(self):
        return self._execute().__await__()

    async def _execute(self):
        if self.result_set is None:
            self.result_set = await self.client.execute(self.sql, self.parameters)
            if self.row_factory == aiosqlite.Row:
                self.row_factory = lambda cursor, row_data: LibsqlRow(self.result_set.columns, row_data)
        return self

    @property
    def rowcount(self) -> int:
        return getattr(self.result_set, "rows_affected", 0) if self.result_set is not None else 0

    async def fetchone(self):
        await self._execute()
        if self._index >= len(self.result_set.rows):
            return None
        row = self.result_set.rows[self._index]
        self._index += 1
        if self.row_factory:
            return self.row_factory(self, row)
        return row

    async def fetchall(self):
        await self._execute()
        rows = self.result_set.rows[self._index:]
        self._index = len(self.result_set.rows)
        if self.row_factory:
            return [self.row_factory(self, r) for r in rows]
        return rows

    async def __aenter__(self):
        await self._execute()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class LibsqlConnectionWrapper:
    """Connection-like wrapper for remote libsql client."""
    def __init__(self, client):
        self.client = client
        self._row_factory = None

    @property
    def row_factory(self):
        return self._row_factory

    @row_factory.setter
    def row_factory(self, factory):
        self._row_factory = factory

    def execute(self, sql: str, parameters: tuple = ()):
        # Return a wrapper that is both awaitable and an async context manager
        return LibsqlCursorWrapper(self, self.client, sql, parameters)

    async def executescript(self, script: str):
        # Split statements by semicolon and execute each
        statements = [stmt.strip() for stmt in script.split(";") if stmt.strip()]
        for stmt in statements:
            await self.client.execute(stmt)

    async def commit(self):
        # libsql auto-commits
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@asynccontextmanager
async def connect_db():
    """Yield a database connection wrapper based on active settings (Turso vs. local sqlite)."""
    settings = get_settings()
    if settings.TURSO_DATABASE_URL and settings.TURSO_AUTH_TOKEN:
        url = settings.TURSO_DATABASE_URL
        if url.startswith("libsql://"):
            url = "https://" + url[len("libsql://"):]
        import libsql_client
        async with libsql_client.create_client(
            url=url,
            auth_token=settings.TURSO_AUTH_TOKEN
        ) as client:
            yield LibsqlConnectionWrapper(client)
    else:
        db_path = settings.DB_PATH
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(db_path) as db:
            yield db


async def get_db_path() -> str:
    settings = get_settings()
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return settings.DB_PATH


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with connect_db() as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intent TEXT,
                confidence TEXT,
                response_id TEXT,
                citations TEXT,
                pipeline_trace TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                response_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT,
                collection_name TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                file_size_bytes INTEGER DEFAULT 0,
                uploaded_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_user_docs_user ON user_documents(user_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_response ON feedback(response_id);
        """)
        
        # Run-time schema migration check
        db.row_factory = aiosqlite.Row
        async with db.execute("PRAGMA table_info(messages)") as cur:
            columns = [row["name"] for row in await cur.fetchall()]
        
        if "citations" not in columns:
            await db.execute("ALTER TABLE messages ADD COLUMN citations TEXT")
        if "pipeline_trace" not in columns:
            await db.execute("ALTER TABLE messages ADD COLUMN pipeline_trace TEXT")
            
        await db.commit()
