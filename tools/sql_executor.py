# tools/sql_executor.py
"""
SQL Execution Tool
Handles SQL query execution and result formatting
"""
import pandas as pd
from typing import Dict, Any
from database.db_manager import DatabaseManager
from utils.logger import logger
import re


class SQLExecutor:
    """Execute SQL queries and format results"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        logger.info("🔧 SQL Executor initialized")
    
    def clean_sql(self, sql: str) -> str:
        """
        Clean SQL query by removing markdown and comments
        
        Args:
            sql: Raw SQL query
        
        Returns:
            Cleaned SQL query
        """
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove SQL comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Clean whitespace
        sql = sql.strip()
        sql = sql.rstrip(';')
        
        return sql
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query and return formatted results
        
        Args:
            sql: SQL query string
        
        Returns:
            Dictionary with results and metadata
        """
        try:
            # Clean the SQL
            cleaned_sql = self.clean_sql(sql)
            logger.info(f"📝 Executing SQL: {cleaned_sql[:100]}...")
            
            # Execute query
            df = self.db.execute_query(cleaned_sql)
            
            # Format results
            result = {
                "success": True,
                "sql": cleaned_sql,
                "row_count": len(df),
                "columns": list(df.columns),
                "data": df.to_dict('records') if len(df) <= 100 else df.head(100).to_dict('records'),
                "data_preview": df.head(10).to_dict('records'),
                "truncated": len(df) > 100
            }
            
            logger.info(f"✅ Query successful: {len(df)} rows returned")
            return result
            
        except Exception as e:
            logger.error(f"❌ SQL execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": sql
            }
    
    def format_results_as_text(self, result: Dict[str, Any]) -> str:
        """
        Format query results as human-readable text
        
        Args:
            result: Query result dictionary
        
        Returns:
            Formatted text string
        """
        if not result["success"]:
            return f"❌ Error: {result['error']}"
        
        df = pd.DataFrame(result["data_preview"])
        
        if df.empty:
            return "No results found."
        
        # For single value (aggregation)
        if len(df) == 1 and len(df.columns) == 1:
            value = df.iloc[0, 0]
            return f"Result: {value}"
        
        # For multiple rows
        text = f"Found {result['row_count']} result(s):\n\n"
        
        # Show preview
        for idx, row in df.iterrows():
            row_data = ", ".join([f"{col}: {val}" for col, val in row.items()])
            text += f"• {row_data}\n"
        
        if result.get("truncated"):
            text += f"\n... (showing first 10 of {result['row_count']} rows)"
        
        return text