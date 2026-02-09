# agents/sql_analyst.py
"""
SQL Analyst Agent - Production Ready
==================================
Professional business analyst that:
- Converts natural language to SQL queries
- Executes queries and generates insights
- Creates charts (bar/line/pie) based on user requests
- Intelligently sorts temporal data (months, quarters)
- Provides concise, actionable business insights

Author: AI Business Analyst System
Version: 3.0 (Production)
"""

from typing import Dict, Any
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from database.db_manager import DatabaseManager
from tools.language_detector import LanguageDetector
from tools.sql_executor import SQLExecutor
from config import settings
from utils.logger import logger, log_section
from utils.plot_builder import PlotBuilder


class SQLAnalystAgent:
    """
    SQL Analyst Agent
    ================
    Transforms natural language questions into SQL queries and business insights.
    
    Features:
    - Multilingual query understanding
    - Smart temporal data sorting (months in chronological order)
    - Chart generation (respects user's chart type requests)
    - Concise business-focused output (<100 words)
    - Handles CSV/Excel hybrid data sources
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize SQL Analyst Agent
        
        Args:
            db_manager: Database connection manager
        """
        # Core components
        self.db = db_manager
        self.language_detector = LanguageDetector()
        self.sql_executor = SQLExecutor(db_manager)
        self.plot_builder = PlotBuilder()
        
        # LLM for query generation and explanation
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        
        # ============================================================
        # MONTH SORTING: Handles all common month formats
        # ============================================================
        # Maps month names/numbers to chronological order (1-12)
        # Handles: January, january, JANUARY, Jan, JAN, jan, 01, 1
        self.MONTH_ORDER = {
            # Full month names
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            # Short month names (3 letters)
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,
            'oct': 10, 'nov': 11, 'dec': 12,
            # Numeric with leading zero
            '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6,
            '07': 7, '08': 8, '09': 9, '10': 10, '11': 11, '12': 12,
            # Numeric without leading zero
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9
        }
        
        # ============================================================
        # SQL GENERATION PROMPT
        # ============================================================
        # Converts natural language to SQL queries
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question", "allowed_tables"],
            template="""You are an SQL expert. Generate queries from natural language.

DATABASE SCHEMA:
{schema}

ALLOWED TABLES ONLY:
{allowed_tables}

RULES:
1. Use ONLY the allowed tables listed above
2. Return ONLY the SQL query (no markdown, no explanation)
3. Use proper SQLite syntax
4. For charts/graphs: Add LIMIT 50 and ORDER BY appropriately
5. NEVER use UNION between different upload tables

QUESTION: {question}

SQL:"""
        )
        
        # ============================================================
        # BUSINESS EXPLANATION PROMPT
        # ============================================================
        # Generates concise, actionable business insights
        self.explanation_prompt = PromptTemplate(
            input_variables=["question", "sql", "results", "language", "language_name"],
            template="""You are a professional business analyst. Provide concise, actionable insights.

Question: {question}
Data: {results}

RESPONSE RULES:
1. Start with a one-sentence summary
2. List top 3-5 key insights as bullet points
3. Each bullet: metric name + number (no extra words)
4. Use natural language (NO markdown symbols like **, --, ##)
5. Keep total response under 100 words
6. Focus on business value, not technical details

EXAMPLE FORMAT:
Summary: Revenue peaked in March at $43,636 and declined 40% by June.

Top Insights:
• March: $43,636 (highest)
• January: $37,500 (second highest)  
• June: $25,833 (lowest)
• Overall decline: 40% from March to June
• Average deal size: $34,182

Answer in {language_name}:"""
        )
        
        # Initialize LLM chains
        self.sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        self.explanation_chain = LLMChain(llm=self.llm, prompt=self.explanation_prompt)
        
        logger.info("🤖 SQL Analyst Agent initialized")
    
    # ================================================================
    # SMART RESULT SORTING
    # ================================================================
    
    def _smart_sort_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently sort query results for temporal data
        
        Why: SQL sorts months alphabetically (Apr, Feb, Jan...) 
        Fix: Sort chronologically (Jan, Feb, Mar...) after query execution
        
        Handles:
        - Month names: January, Jan, JAN, january, jan
        - Month numbers: 01, 1, 02, 2, etc.
        - Mixed formats in same column
        - Non-month data (safely ignored)
        
        Args:
            result: Query result dictionary with 'data_preview' key
            
        Returns:
            Result with chronologically sorted data if months detected
        """
        # Skip if no data to sort
        if not result.get('success') or not result.get('data_preview'):
            return result
        
        try:
            # Convert to DataFrame for easy manipulation
            df = pd.DataFrame(result['data_preview'])
            
            # Check each column for month data
            for col in df.columns:
                col_lower = col.lower()
                
                # Is this a month column? (by name)
                if 'month' in col_lower:
                    # Check if values are actually months (not just column name)
                    if len(df) > 0:
                        sample_val = str(df[col].iloc[0]).lower().strip()
                        
                        # Is the first value a recognized month?
                        if sample_val in self.MONTH_ORDER:
                            # Yes! Sort chronologically
                            # Create temporary sort key (1-12)
                            df['_month_sort_key'] = (
                                df[col]
                                .str.lower()
                                .str.strip()
                                .map(self.MONTH_ORDER)
                            )
                            
                            # Sort by the numeric key
                            df = df.sort_values('_month_sort_key')
                            
                            # Remove temporary column
                            df = df.drop('_month_sort_key', axis=1)
                            
                            # Update result with sorted data
                            result['data_preview'] = df.to_dict('records')
                            
                            logger.info(f"📅 Sorted '{col}' chronologically (Jan→Dec)")
                            break  # Only sort by first month column found
            
        except Exception as e:
            # Don't break the system if sorting fails
            # Just use original SQL order
            logger.warning(f"⚠️ Smart sort failed: {e}, using SQL order")
        
        return result
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def _is_simple_aggregation(self, question: str) -> bool:
        """
        Check if question asks for a simple total/count
        
        Examples:
        - "how many total sales?"
        - "what is the total revenue?"
        - "count of customers"
        
        Returns:
            True if simple aggregation detected
        """
        q = question.lower()
        return any(kw in q for kw in [
            "total", "total no", "total number",
            "how many", "count", "number of",
            "sum of", "overall total"
        ])
    
    def _extract_entity_from_question(self, question: str, result: Dict[str, Any]) -> str:
        """
        Smart entity detection for aggregation results
        
        Tries to determine what's being counted/summed:
        1. Check question for keywords (customers, sales, etc.)
        2. Fallback to column name
        
        Args:
            question: User's question
            result: Query result
            
        Returns:
            Entity name (plural): "customers", "sales", "repositories"
        """
        question_lower = question.lower()
        
        # Common entity keywords
        entity_map = {
            'repo': 'repositories',
            'repository': 'repositories',
            'customer': 'customers',
            'employee': 'employees',
            'department': 'departments',
            'sale': 'sales',
            'issue': 'issues',
            'user': 'users',
            'order': 'orders',
            'product': 'products',
            'transaction': 'transactions',
            'file': 'files',
            'document': 'documents',
        }
        
        # Check question for entity keywords
        for keyword, entity in entity_map.items():
            if keyword in question_lower:
                return entity
        
        # Fallback: extract from column name
        first_row = result.get("data_preview", [{}])[0]
        if first_row:
            column_name = list(first_row.keys())[0]
            # Clean up SQL aggregate functions
            entity = (
                column_name
                .replace('_', ' ')
                .replace('COUNT(DISTINCT ', '')
                .replace('COUNT(', '')
                .replace(')', '')
                .strip()
            )
            return entity
        
        return 'items'
    
    # ================================================================
    # MAIN ANALYSIS METHOD
    # ================================================================
    
    def analyze(self, question: str, allowed_tables: list[str] | None = None) -> Dict[str, Any]:
        """
        Main analysis pipeline
        
        Flow:
        1. Detect language (for multilingual support)
        2. Generate SQL from natural language
        3. Execute SQL query
        4. Smart sort results (fix month ordering)
        5. Generate business insights
        6. Create chart if requested
        
        Args:
            question: Natural language question
            allowed_tables: List of table names this query can access
            
        Returns:
            Dictionary with:
            - success: bool
            - explanation: Business insights (text)
            - data: Query results
            - plot: Chart specification (if chart requested)
            - sql_query: Generated SQL (for debugging)
        """
        log_section(f"ANALYZING QUESTION")
        logger.info(f"❓ Question: {question}")
        
        try:
            # ========================================================
            # STEP 1: Detect Language
            # ========================================================
            lang_code = self.language_detector.detect_language(question)
            lang_name = self.language_detector.get_language_name(lang_code)
            
            # ========================================================
            # STEP 2: Generate SQL Query
            # ========================================================
            schema = self.db.get_schema_text()
            
            logger.info("🔧 Generating SQL query...")
            sql_query = self.sql_chain.run(
                schema=schema, 
                question=question,
                allowed_tables=", ".join(allowed_tables or []) if allowed_tables else "NONE"
            )
            sql_query = sql_query.strip()
            
            logger.info(f"📝 Generated SQL: {sql_query}")
            
            # ========================================================
            # STEP 3: Execute SQL Query
            # ========================================================
            logger.info("⚙️  Executing query...")
            result = self.sql_executor.execute_query(sql_query)
            
            # Check for SQL errors
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "question": question,
                    "language": lang_code
                }
            
            # ========================================================
            # STEP 4: Smart Sort Results (Fix Month Ordering)
            # ========================================================
            # This ensures months appear in chronological order
            # (Jan, Feb, Mar...) instead of alphabetical (Apr, Feb, Jan...)
            result = self._smart_sort_results(result)
            
            # ========================================================
            # STEP 5: Handle Simple Aggregations
            # ========================================================
            # For "how many total X?" questions, provide simple answer
            if self._is_simple_aggregation(question):
                numeric_values = []
                
                # Extract all numeric values from result
                for row in result.get("data_preview", []):
                    for value in row.values():
                        if isinstance(value, (int, float)):
                            numeric_values.append(value)
                
                if numeric_values:
                    total_value = sum(numeric_values)
                    entity = self._extract_entity_from_question(question, result)
                    
                    # Simple, clean format: "Total: 150 customers"
                    explanation = f"Total: {int(total_value)} {entity}"
                    
                    response = {
                        "success": True,
                        "question": question,
                        "language": lang_code,
                        "language_name": lang_name,
                        "sql_query": sql_query,
                        "row_count": result["row_count"],
                        "data": result["data_preview"],
                        "explanation": explanation,
                        "raw_results": result
                    }
                    
                    # Add chart if user requested one
                    if self.plot_builder.needs_plot(question):
                        plot_spec = self.plot_builder.build_plot_spec(result, question)
                        if plot_spec:
                            response["plot"] = plot_spec
                            logger.info(f"✅ Plot spec: {plot_spec['type']} chart")
                    
                    return response
            
            # ========================================================
            # STEP 6: Generate Business Insights
            # ========================================================
            results_text = self.sql_executor.format_results_as_text(result)
            logger.info(f"💬 Generating business analysis...")
            
            explanation = self.explanation_chain.run(
                question=question,
                sql=sql_query,
                results=results_text,
                language=lang_code,
                language_name=lang_name
            )
            
            logger.info("✅ Analysis complete")
            
            # ========================================================
            # STEP 7: Prepare Response
            # ========================================================
            response = {
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
            
            # ========================================================
            # STEP 8: Add Chart (if requested)
            # ========================================================
            # PlotBuilder respects user's chart type request:
            # - "bar chart" → bar
            # - "line chart" → line
            # - "pie chart" → pie
            if self.plot_builder.needs_plot(question):
                plot_spec = self.plot_builder.build_plot_spec(result, question)
                if plot_spec:
                    response["plot"] = plot_spec
                    logger.info(f"✅ Plot spec: {plot_spec['type']} chart")
            
            return response
            
        except Exception as e:
            # ========================================================
            # ERROR HANDLING
            # ========================================================
            logger.error(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database schema information
        
        Returns:
            Dictionary with tables, schema, and table count
        """
        tables = self.db.list_tables()
        schema = self.db.get_schema()
        
        return {
            "tables": tables,
            "schema": schema,
            "total_tables": len(tables)
        }


