# parsers/hybrid_csv_processor.py
"""
Hybrid CSV Processor - Loads CSV/Excel into BOTH SQL and RAG
Enables both precise calculations AND semantic understanding
"""
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any
from langchain.schema import Document

from database.db_manager import DatabaseManager
from parsers.document_parser import DocumentParser
from utils.logger import logger
from config import CSV_TABLE_PREFIX


class HybridCSVProcessor:
    """
    Process CSV/Excel files for hybrid querying
    Loads into SQL for precise calculations AND RAG for semantic understanding
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.doc_parser = DocumentParser()
        logger.info("📊 Hybrid CSV Processor initialized")
    
    def is_structured_file(self, filename: str) -> bool:
        """Check if file is CSV/Excel (structured data)"""
        return filename.lower().endswith(('.csv', '.xlsx', '.xls'))
    
    def process_file(self, file_path: str, conversation_id: str) -> Dict[str, Any]:
        """
        Process CSV/Excel file with hybrid approach
        
        Returns:
            Dict with SQL table name and RAG documents
        """
        filename = Path(file_path).name
        
        if not self.is_structured_file(filename):
            raise ValueError(f"File {filename} is not a structured data file")
        
        logger.info(f"📊 Processing structured file: {filename}")
        logger.info(f"   Mode: HYBRID (SQL + RAG)")
        
        # Step 1: Import to SQL for precise calculations
        sql_table = self._import_to_sql(file_path, conversation_id)
        
        # Step 2: Parse to RAG for semantic understanding
        rag_docs = self._parse_to_rag(file_path)
        
        # Step 3: Get metadata
        metadata = self._extract_metadata(file_path)
        
        result = {
            "filename": filename,
            "sql_table": sql_table,
            "rag_documents": rag_docs,
            "metadata": metadata,
            "capabilities": {
                "sql_calculations": True,
                "semantic_search": True,
                "hybrid_analysis": True
            }
        }
        
        logger.info(f"""
✅ Hybrid Processing Complete: {filename}
   📊 SQL Table: {sql_table} ({metadata['row_count']} rows)
   📄 RAG Chunks: {len(rag_docs)} documents
   ✨ Capabilities: SQL calculations + Semantic understanding
        """)
        
        return result
    
    def _import_to_sql(self, file_path: str, conversation_id: str) -> str:
        """Import CSV/Excel to SQL database"""
        
        # Read file
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"❌ Failed to read file: {e}")
            raise
        
        # Generate table name
        table_name = self._generate_table_name(file_path, conversation_id)
        
        # Import to database
        try:
            conn = self.db.get_connection()
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            logger.info(f"   ✅ SQL Import: {table_name} ({len(df)} rows, {len(df.columns)} columns)")
            return table_name
            
        except Exception as e:
            logger.error(f"❌ SQL import failed: {e}")
            raise
    
    def _parse_to_rag(self, file_path: str) -> List[Document]:
        """Parse CSV/Excel to RAG documents"""
        
        try:
            # Use existing document parser
            if file_path.endswith('.csv'):
                docs = self.doc_parser.parse_csv(file_path)
            else:
                docs = self.doc_parser.parse_excel(file_path)
            
            logger.info(f"   ✅ RAG Parsing: {len(docs)} document chunks")
            return docs
            
        except Exception as e:
            logger.error(f"❌ RAG parsing failed: {e}")
            raise
    
    def _generate_table_name(self, file_path: str, conversation_id: str) -> str:
        """Generate SQL table name for uploaded file"""
        
        # Get base name without extension
        base_name = Path(file_path).stem
        
        # Sanitize: only alphanumeric and underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).lower()
        
        # Add prefix and session ID
        table_name = f"{CSV_TABLE_PREFIX}{conversation_id[:8]}_{sanitized}"
        
        # Ensure valid SQL identifier (max 63 chars for PostgreSQL)
        if len(table_name) > 63:
            table_name = table_name[:63]
        
        return table_name
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from CSV/Excel"""
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            return {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Metadata extraction failed: {e}")
            return {}
    
    def get_uploaded_tables(self, conversation_id: str) -> List[str]:
        """Get list of SQL tables for this session"""
        
        all_tables = self.db.list_tables()
        prefix = f"{CSV_TABLE_PREFIX}{conversation_id[:8]}_"
        
        session_tables = [t for t in all_tables if t.startswith(prefix)]
        
        logger.info(f"📊 Found {len(session_tables)} uploaded tables for session")
        return session_tables
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about uploaded table"""
        
        try:
            # Get sample data
            sample_df = self.db.get_table_sample(table_name, limit=5)
            
            # Get full count
            count_df = self.db.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = count_df.iloc[0]['count']
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "columns": list(sample_df.columns),
                "sample_data": sample_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get table info: {e}")
            return {}


# Convenience function
def process_csv_hybrid(file_path: str, conversation_id: str, 
                       db_manager: DatabaseManager) -> Dict[str, Any]:
    """
    Convenience function to process CSV/Excel in hybrid mode
    
    Args:
        file_path: Path to CSV/Excel file
        conversation_id: Session ID
        db_manager: Database manager instance
    
    Returns:
        Processing result with SQL table and RAG documents
    """
    processor = HybridCSVProcessor(db_manager)
    return processor.process_file(file_path, conversation_id)