
import os
import re
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from utils.logger import logger


# Statements permitted through execute_query(). LLM-generated SQL is never
# trusted to self-police — this allow-list is the first of two independent
# guards (the second is opening the DB connection itself in read-only mode).
_READ_ONLY_STATEMENT_RE = re.compile(r"^\s*(SELECT|WITH|EXPLAIN|PRAGMA\s+table_info)\b", re.IGNORECASE)

# Any of these appearing as a standalone SQL keyword anywhere in the query
# is grounds for rejection, even if the statement starts with SELECT/WITH —
# guards against smuggled mutations via CTEs, subqueries, or stacked
# statements (e.g. "SELECT 1; DROP TABLE x").
_FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|ATTACH|DETACH|"
    r"GRANT|REVOKE|VACUUM|REINDEX|PRAGMA(?!\s+table_info))\b"
    # REPLACE is only a mutation as a statement (REPLACE INTO …); replace()
    # is an everyday string function used to clean "$1,234" style values.
    r"|\bREPLACE\b(?!\s*\()",
    re.IGNORECASE,
)

# Quoted text can't be a keyword: a literal like '%update%' or an
# identifier like "Created" must not trip the checks below.
_QUOTED_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\]")


SQL_TIMEOUT_S = float(os.getenv("SQL_TIMEOUT_S", "20"))


class QueryTimeoutError(RuntimeError):
    """The query ran past SQL_TIMEOUT_S and SQLite interrupted it."""


class ForbiddenTableError(PermissionError):
    """The query read a table outside the allowed set."""


def _table_authorizer(allowed_tables: List[str]):
    """
    sqlite3 authorizer: allow reads only from allowed_tables (case-insensitive).
    CTEs and subqueries are resolved by SQLite before this runs, so a CTE
    named like a foreign table is fine and `main.t`, `'t'`, `[t]` are not.
    """
    import sqlite3
    allowed = {t.lower() for t in allowed_tables}
    pragma_ok = {"table_info", "table_xinfo"}

    def auth(action, arg1, arg2, dbname, source):
        if action == sqlite3.SQLITE_READ:
            if dbname is None:                       # a CTE or subquery, not a stored table
                return sqlite3.SQLITE_OK
            name = (arg1 or "").lower()
            if name in allowed or name in ("pragma_table_info", "pragma_table_xinfo"):
                return sqlite3.SQLITE_OK             # pragma functions are checked below
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            ok = (arg1 or "").lower() in pragma_ok and (arg2 or "").strip("'\"").lower() in allowed
            return sqlite3.SQLITE_OK if ok else sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return auth


def _assert_read_only_sql(query: str) -> None:
    """Raise ValueError if `query` is anything other than a single read-only statement."""
    stripped = _QUOTED_RE.sub("''", query.strip().rstrip(";"))
    if ";" in stripped:
        raise ValueError("Multiple/stacked SQL statements are not allowed")
    if not _READ_ONLY_STATEMENT_RE.match(stripped):
        raise ValueError("Only SELECT/WITH/EXPLAIN queries are allowed")
    if _FORBIDDEN_KEYWORDS_RE.search(stripped):
        raise ValueError("Query contains a disallowed (non-read-only) SQL keyword")


class DatabaseManager:
    """Universal database manager"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            from config import get_db_config
            config = get_db_config()
        
        self.config = config
        self.db_type = config["type"]
        
        if self.db_type == "sqlite":
            self._init_sqlite(config["path"])
        elif self.db_type == "postgresql":
            self._init_postgresql(config)
        elif self.db_type == "mysql":
            self._init_mysql(config)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        logger.info(f"📊 Database initialized: {self.db_type}")
    
    def _init_sqlite(self, db_path: str):
        import sqlite3
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.connection_string = db_path
        conn = sqlite3.connect(db_path)
        conn.close()
        logger.info(f"   Path: {db_path}")
    
    def _init_postgresql(self, config: Dict):
        try:
            import psycopg2
        except ImportError:
            raise ImportError("PostgreSQL requires psycopg2-binary: pip install psycopg2-binary")
        
        self.connection_string = (
            f"postgresql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )
        
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        conn.close()
        logger.info(f"   Host: {config['host']}:{config['port']}")
    
    def _init_mysql(self, config: Dict):
        try:
            import pymysql
        except ImportError:
            raise ImportError("MySQL requires pymysql: pip install pymysql")
        
        self.connection_string = (
            f"mysql+pymysql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )
        
        conn = pymysql.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        conn.close()
        logger.info(f"   Host: {config['host']}:{config['port']}")
    
    def get_connection(self):
        """Get database connection"""
        if self.db_type == "sqlite":
            import sqlite3
            return sqlite3.connect(self.db_path)
        elif self.db_type == "postgresql":
            import psycopg2
            config = self.config
            return psycopg2.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
        elif self.db_type == "mysql":
            import pymysql
            config = self.config
            return pymysql.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
    
    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Get database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.db_type == "sqlite":
                return self._get_sqlite_schema(cursor)
            elif self.db_type == "postgresql":
                return self._get_postgres_schema(cursor)
            elif self.db_type == "mysql":
                return self._get_mysql_schema(cursor)
        finally:
            conn.close()
    
    def _get_sqlite_schema(self, cursor) -> Dict:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Table name must be quoted — PRAGMA table_info(x) does not
            # accept bind parameters, and an unquoted identifier containing
            # any non-alphanumeric character (a hyphen, for example) is a
            # syntax error rather than a valid table reference.
            cursor.execute(f'PRAGMA table_info("{table}");')
            columns = cursor.fetchall()
            schema[table] = [
                {
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "primary_key": bool(col[5])
                }
                for col in columns
            ]
        return schema
    
    def _get_postgres_schema(self, cursor) -> Dict:
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
            """)
            columns = cursor.fetchall()
            schema[table] = [
                {
                    "name": col[0],
                    "type": col[1],
                    "notnull": col[2] == 'NO',
                    "primary_key": False
                }
                for col in columns
            ]
        return schema
    
    def _get_mysql_schema(self, cursor) -> Dict:
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            schema[table] = [
                {
                    "name": col[0],
                    "type": col[1],
                    "notnull": col[2] == 'NO',
                    "primary_key": col[3] == 'PRI'
                }
                for col in columns
            ]
        return schema
    
    def get_schema_text(self) -> str:
        """Get schema as formatted text"""
        schema = self.get_schema()
        
        schema_text = f"DATABASE SCHEMA ({self.db_type.upper()}):\n\n"
        
        for table, columns in schema.items():
            schema_text += f"Table: {table}\n"
            schema_text += "Columns:\n"
            
            for col in columns:
                nullable = "NULL" if not col["notnull"] else "NOT NULL"
                pk = " PRIMARY KEY" if col["primary_key"] else ""
                schema_text += f"  - {col['name']} ({col['type']}) {nullable}{pk}\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def execute_query(self, query: str, allowed_tables: Optional[List[str]] = None,
                      timeout_s: Optional[float] = None) -> pd.DataFrame:
        """
        Execute a read-only SQL query (the path LLM-generated SQL runs through).

        Independent guards, since the SQL text here originates from an LLM
        and must never be trusted to only contain SELECT statements on its own:
          1. Statement allow-list — reject anything that isn't a single
             read-only SELECT/WITH/EXPLAIN/PRAGMA-table-info statement before
             it ever reaches the database driver.
          2. A connection opened read-only at the OS/driver level (SQLite URI
             mode=ro; Postgres/MySQL default-transaction READ ONLY) as
             defense-in-depth, so even a guard bypass can't mutate data.
          3. allowed_tables (SQLite): the engine itself refuses to read any
             other table. Regex checks on SQL text can be dodged with
             main.t, 't', /**/ comments or subqueries; the authorizer sees
             the tables SQLite actually resolved.
          4. A time limit (SQLite): a recursive CTE without a LIMIT or a
             3-way self-join would otherwise pin a worker thread forever.
        """
        _assert_read_only_sql(query)
        logger.info(f"🔍 Executing query: {query[:100]}...")

        conn, limit = None, 0.0
        try:
            if self.db_type == "sqlite":
                import sqlite3
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
                if allowed_tables is not None:
                    conn.set_authorizer(_table_authorizer(allowed_tables))
                limit = timeout_s if timeout_s is not None else SQL_TIMEOUT_S
                if limit and limit > 0:
                    deadline = time.monotonic() + limit
                    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            else:
                from sqlalchemy import create_engine
                engine = create_engine(
                    self.connection_string,
                    execution_options={"postgresql_readonly": True} if self.db_type == "postgresql" else {},
                )
                conn = engine.connect()
                if self.db_type == "mysql":
                    conn.execute("SET SESSION TRANSACTION READ ONLY")

            try:
                df = pd.read_sql_query(query, conn)
            except Exception as e:
                text = str(e)
                if "interrupted" in text.lower():
                    raise QueryTimeoutError(
                        f"The query took longer than {limit:g}s and was stopped. "
                        "Try a narrower question.") from e
                if "not authorized" in text.lower() or "prohibited" in text.lower():
                    raise ForbiddenTableError(
                        "The query tried to read data outside this conversation's files.") from e
                raise

            logger.info(f"✅ Query returned {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()
    
    def get_table_sample(self, table: str, limit: int = 5) -> pd.DataFrame:
        """Get sample rows"""
        query = f'SELECT * FROM "{table}" LIMIT {int(limit)}'
        return self.execute_query(query)
    
    def list_tables(self) -> List[str]:
        """List all tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.db_type == "sqlite":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            elif self.db_type == "postgresql":
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
            elif self.db_type == "mysql":
                cursor.execute("SHOW TABLES")
            
            tables = [row[0] for row in cursor.fetchall()]
            return tables
        finally:
            conn.close()
    
    def get_db_info(self) -> Dict[str, Any]:
        """Get database information"""
        return {
            "type": self.db_type,
            "tables": self.list_tables(),
            "connection": self.connection_string if self.db_type != "sqlite" else self.db_path
        }


if __name__ == "__main__":
    print("\n🧪 Testing Database Connection...\n")
    try:
        db = DatabaseManager()
        print(f"✅ Connected to {db.db_type}")
        print(f"\nTables: {db.list_tables()}")
        print(f"\nSchema preview:")
        print(db.get_schema_text()[:500] + "...")
    except Exception as e:
        print(f"❌ Connection failed: {e}")