


import re
import pandas as pd
from typing import Dict, Any, Optional

from database.db_manager import DatabaseManager
from tools.column_value_index import ColumnValueIndex
from utils.logger import logger


class SQLExecutor:
    """Execute SQL queries and format results, with automatic value normalization."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.value_index = ColumnValueIndex()
        logger.info("🔧 SQL Executor initialized")

    # ================================================================
    # INDEX MANAGEMENT
    # ================================================================

    def index_table(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Register a table's schema with the value index.
        Call this right after importing a CSV/Excel into SQLite.

        Args:
            df:         The source DataFrame (same data that was imported)
            table_name: The SQLite table name it was saved as
        """
        self.value_index.build_from_dataframe(df, table_name)

    # ================================================================
    # SQL CLEANING
    # ================================================================

    def clean_sql(self, sql: str) -> str:
        """Raw SQL → Cleaned SQL (removes markdown, comments, extra whitespace)."""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)

        # Remove SQL comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

        # Clean whitespace
        sql = sql.strip().rstrip(';')
        return sql

    # ================================================================
    # TABLE NAME EXTRACTION
    # ================================================================

    def _extract_primary_table(self, sql: str) -> Optional[str]:
        """
        Extract the first table name from a SQL query.
        Used to select which index to use for value normalization.
        Handles: FROM table, JOIN table, UPDATE table, INSERT INTO table
        """
        # Match: FROM <name>, JOIN <name>, UPDATE <name>, INTO <name>
        pattern = r'\b(?:FROM|JOIN|UPDATE|INTO)\s+([`"\[]?[\w]+[`"\]]?)'
        matches = re.findall(pattern, sql, flags=re.IGNORECASE)
        if matches:
            # Strip any quoting characters
            table = matches[0].strip('`"[]')
            return table
        return None

    # ================================================================
    # MAIN EXECUTION PIPELINE
    # ================================================================

    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        SQL string → Query result dict.

        Pipeline:
          1. Clean SQL (remove markdown, comments)
          2. Identify which table is being queried
          3. Rewrite SQL string literals to match actual DB values
          4. Execute
          5. Return structured result
        """
        try:
            # Step 1: Clean
            cleaned_sql = self.clean_sql(sql)
            logger.info(f"📝 Executing SQL: {cleaned_sql[:100]}...")

            # Step 2: Identify table
            table_name = self._extract_primary_table(cleaned_sql)

            # Step 3: Rewrite literals if we have an index for this table
            if table_name and self.value_index.has_table(table_name):
                rewritten_sql = self.value_index.rewrite_sql(cleaned_sql, table_name)
                if rewritten_sql != cleaned_sql:
                    logger.info(f"✏️ SQL normalized for table '{table_name}'")
                    logger.info(f"   Original:  {cleaned_sql[:120]}")
                    logger.info(f"   Rewritten: {rewritten_sql[:120]}")
                    cleaned_sql = rewritten_sql
            else:
                if table_name:
                    logger.debug(f"⚠️ No index for table '{table_name}', skipping normalization")

            # Step 4: Execute
            logger.info(f"🔍 Executing query: {cleaned_sql[:100]}...")
            df = self.db.execute_query(cleaned_sql)

            # Step 5: Return result
            result = {
                "success": True,
                "sql": cleaned_sql,
                "row_count": len(df),
                "columns": list(df.columns),
                "data": (
                    df.to_dict('records') if len(df) <= 100
                    else df.head(100).to_dict('records')
                ),
                "data_preview": df.head(10).to_dict('records'),
                "truncated": len(df) > 100,
            }

            logger.info(f"✅ Query successful: {len(df)} rows returned")
            return result

        except Exception as e:
            logger.error(f"❌ SQL execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": sql,
            }

    # ================================================================
    # RESULT FORMATTING
    # ================================================================

    def format_results_as_text(self, result: Dict[str, Any]) -> str:
        """Query result dict → Formatted text for agent response."""
        if not result["success"]:
            return f"❌ Error: {result['error']}"

        df = pd.DataFrame(result["data_preview"])

        if df.empty:
            return "No results found."

        # Single value (aggregation)
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            return f"Result: {value}"

        # Multiple rows
        text = f"Found {result['row_count']} result(s):\n\n"
        for _, row in df.iterrows():
            row_data = ", ".join([f"{col}: {val}" for col, val in row.items()])
            text += f"• {row_data}\n"

        if result.get("truncated"):
            text += f"\n... (showing first 10 of {result['row_count']} rows)"

        return text
