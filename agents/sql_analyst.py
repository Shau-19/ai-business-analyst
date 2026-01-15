# agents/sql_analyst.py
"""
SQL Analyst Agent - Multilingual Business Analyst
Handles natural language questions, generates SQL, executes queries,
and explains results in the user's language
"""
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from database.db_manager import DatabaseManager
from tools.language_detector import LanguageDetector
from tools.sql_executor import SQLExecutor
from config import settings
from utils.logger import logger, log_section


class SQLAnalystAgent:
    """
    Multilingual SQL Analyst Agent
    
    Capabilities:
    - Detect question language
    - Generate SQL from natural language
    - Execute queries
    - Explain results in original language
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.language_detector = LanguageDetector()
        self.sql_executor = SQLExecutor(db_manager)
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        # SQL Generation Prompt
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""You are an expert SQL analyst. Generate a SQLite query to answer the question.

DATABASE SCHEMA:
{schema}

RULES:
1. Generate ONLY the SQL query (no markdown, no explanation)
2. Use proper SQLite syntax
3. Use JOINs when needed across multiple tables
4. For aggregations, use COUNT(), SUM(), AVG(), MAX(), MIN()
5. Use proper GROUP BY and ORDER BY clauses
6. Return clean SQL without comments

QUESTION: {question}

SQL QUERY:"""
        )
        
        # Result Explanation Prompt
        self.explanation_prompt = PromptTemplate(
            input_variables=["question", "sql", "results", "language", "language_name"],
            template="""You are a business analyst explaining SQL query results.

ORIGINAL QUESTION: {question}

SQL QUERY EXECUTED:
{sql}

QUERY RESULTS:
{results}

INSTRUCTIONS:
1. Explain the results clearly and concisely
2. Include key insights and numbers
3. Format nicely with bullet points if multiple items
4. Respond ENTIRELY in {language_name} language (language code: {language})
5. Be professional but conversational

EXPLANATION IN {language_name}:"""
        )
        
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        self.explanation_chain = LLMChain(llm=self.llm, prompt=self.explanation_prompt)
        
        logger.info("🤖 SQL Analyst Agent initialized")
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Main analysis method
        
        Args:
            question: Natural language question in any language
        
        Returns:
            Dictionary with analysis results
        """
        log_section(f"ANALYZING QUESTION")
        logger.info(f"❓ Question: {question}")
        
        try:
            # Step 1: Detect language
            lang_code = self.language_detector.detect_language(question)
            lang_name = self.language_detector.get_language_name(lang_code)
            
            # Step 2: Get database schema
            schema = self.db.get_schema_text()
            
            # Step 3: Generate SQL
            logger.info("🔧 Generating SQL query...")
            sql_query = self.sql_chain.run(schema=schema, question=question)
            sql_query = sql_query.strip()
            
            logger.info(f"📝 Generated SQL: {sql_query}")
            
            # Step 4: Execute SQL
            logger.info("⚙️  Executing query...")
            result = self.sql_executor.execute_query(sql_query)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "question": question,
                    "language": lang_code
                }
            
            # Step 5: Format results for explanation
            results_text = self.sql_executor.format_results_as_text(result)
            
            # Step 6: Generate explanation in user's language
            logger.info(f"💬 Generating explanation in {lang_name}...")
            explanation = self.explanation_chain.run(
                question=question,
                sql=sql_query,
                results=results_text,
                language=lang_code,
                language_name=lang_name
            )
            
            logger.info("✅ Analysis complete")
            
            return {
                "success": True,
                "question": question,
                "language": lang_code,
                "language_name": lang_name,
                "sql_query": sql_query,
                "row_count": result["row_count"],
                "data": result["data_preview"],
                "explanation": explanation.strip(),
                "raw_results": result
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information
        
        Returns:
            Dictionary with database schema and stats
        """
        tables = self.db.list_tables()
        schema = self.db.get_schema()
        
        info = {
            "tables": tables,
            "schema": schema,
            "total_tables": len(tables)
        }
        
        return info