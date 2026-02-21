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
    
    def __init__(self, db_manager: DatabaseManager, sql_executor=None):
        self.db = db_manager
        self.doc_parser = DocumentParser()
        self.sql_executor = sql_executor
        logger.info("📊 Hybrid CSV Processor initialized")
    
    def is_structured_file(self, filename: str) -> bool:
        return filename.lower().endswith(('.csv', '.xlsx', '.xls'))
    
    def process_file(self, file_path: str, conversation_id: str) -> Dict[str, Any]:
        filename = Path(file_path).name
        
        if not self.is_structured_file(filename):
            raise ValueError(f"File {filename} is not a structured data file")
        
        logger.info(f"📊 Processing structured file: {filename}")
        logger.info(f"   Mode: HYBRID (SQL + RAG)")
        
        sql_table = self._import_to_sql(file_path, conversation_id)
        rag_docs = self._parse_to_rag(file_path)
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
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"❌ Failed to read file: {e}")
            raise
        
        table_name = self._generate_table_name(file_path, conversation_id)
        
        try:
            conn = self.db.get_connection()
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            logger.info(f"   ✅ SQL Import: {table_name} ({len(df)} rows, {len(df.columns)} columns)")

            if self.sql_executor is not None:
                self.sql_executor.index_table(df, table_name)
                logger.info(f"   📇 Value index built for '{table_name}'")

            return table_name
            
        except Exception as e:
            logger.error(f"❌ SQL import failed: {e}")
            raise
    
    def _parse_to_rag(self, file_path: str) -> List[Document]:
        try:
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
        base_name = Path(file_path).stem
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).lower()
        table_name = f"{CSV_TABLE_PREFIX}{conversation_id[:8]}_{sanitized}"
        if len(table_name) > 63:
            table_name = table_name[:63]
        return table_name
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
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
        all_tables = self.db.list_tables()
        prefix = f"{CSV_TABLE_PREFIX}{conversation_id[:8]}_"
        session_tables = [t for t in all_tables if t.startswith(prefix)]
        logger.info(f"📊 Found {len(session_tables)} uploaded tables for session")
        return session_tables
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        try:
            sample_df = self.db.get_table_sample(table_name, limit=5)
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


def process_csv_hybrid(file_path: str, conversation_id: str,
                       db_manager: DatabaseManager) -> Dict[str, Any]:
    processor = HybridCSVProcessor(db_manager)
    return processor.process_file(file_path, conversation_id)
