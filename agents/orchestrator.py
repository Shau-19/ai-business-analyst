# agents/orchestrator.py
"""
Smart Query Orchestrator - Production Ready
==========================================
Intelligent routing agent that directs queries to the appropriate handler:
- SQL Agent: For database queries and calculations
- RAG Agent: For document understanding and semantic search
- HYBRID Mode: For queries needing both numerical data AND contextual explanation

Key Features:
- Smart hybrid detection (definition + data questions)
- CSV-aware routing with calculation vs semantic distinction
- Session-aware document handling
- Multi-agent coordination with A2A protocol

Author: AI Business Analyst System
Version: 3.0 (Production with Hybrid Fix)
"""

from typing import Dict, Any
import re
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from protocols import A2AAgent, AgentCapability, MessageType, A2AMessage, A2ARegistry
from agents.sql_analyst import SQLAnalystAgent
from agents.session_rag import SessionAwareRAGAgent
from database.db_manager import DatabaseManager
from config import settings, ENABLE_HYBRID_CSV
from utils.logger import logger, log_section

# ================================================================
# CSV ROUTER INITIALIZATION
# ================================================================
# Optional CSV router for hybrid SQL+RAG queries on structured data
if ENABLE_HYBRID_CSV:
    try:
        from tools.csv_query_router import CSVQueryRouter
        CSV_ROUTER_AVAILABLE = True
    except ImportError:
        logger.warning("⚠️ CSV Router not available - hybrid CSV disabled")
        CSV_ROUTER_AVAILABLE = False
else:
    CSV_ROUTER_AVAILABLE = False


class OrchestratorAgent(A2AAgent):
    """
    Smart Query Orchestrator
    =======================
    Routes queries to appropriate agents based on query type and session context.
    
    Routing Logic Priority:
    1. HYBRID indicators (definition + data) - HIGHEST
    2. CSV router recommendations
    3. Keyword detection (document vs SQL)
    4. LLM classification - FALLBACK
    
    Modes:
    - SQL: Pure database queries (calculations, aggregations)
    - DOCUMENT: Pure document queries (definitions, explanations)
    - HYBRID: Combined queries needing both (e.g., "explain X and show top 5")
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize orchestrator with sub-agents
        
        Args:
            db_manager: Database connection manager
        """
        super().__init__(
            agent_id="orchestrator",
            name="Smart Query Orchestrator",
            description="Routes queries intelligently to SQL, RAG, or HYBRID"
        )
        
        # ============================================================
        # INITIALIZE SUB-AGENTS
        # ============================================================
        self.registry = A2ARegistry()
        self.sql_agent = SQLAnalystAgent(db_manager)
        self.rag_agent = SessionAwareRAGAgent(db_manager=db_manager)
        
        # Initialize optional CSV router
        if CSV_ROUTER_AVAILABLE:
            self.csv_router = CSVQueryRouter()
            logger.info("✅ CSV Hybrid Router enabled")
        else:
            self.csv_router = None
            logger.info("ℹ️ CSV Hybrid Router disabled")
        
        # Wrap agents for A2A protocol
        self.sql_agent_a2a = self._wrap_sql_agent()
        self.rag_agent_a2a = self._wrap_rag_agent()
        
        # Register agents in A2A registry
        self.registry.register_agent(self.sql_agent_a2a)
        self.registry.register_agent(self.rag_agent_a2a)
        
        # ============================================================
        # LLM FOR CLASSIFICATION
        # ============================================================
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0  # Deterministic for routing
        )
        
        # ============================================================
        # ROUTING PROMPT (FALLBACK CLASSIFIER)
        # ============================================================
        self.routing_prompt = PromptTemplate(
            input_variables=["question", "has_documents", "document_files"],
            template="""You are a smart query router.

CONTEXT:
- User has uploaded documents: {has_documents}
- Uploaded files: {document_files}

Available Systems:
1. SQL - For queries about the LIVE DATABASE
   - Tables: employees, departments, sales, products, customers
   - Use ONLY when asking about data in these specific database tables

2. DOCUMENT - For queries about UPLOADED FILES
   - Use when question is about uploaded files
   - Examples: "roadmap", "review", "report", "uploaded data"

3. HYBRID - Compare database with uploaded files
   - Example: "Compare database sales with targets in uploaded file"

CRITICAL RULES:
- If question is about uploaded file content → DOCUMENT
- Only use SQL for questions about database tables
- When in doubt and user has documents → DOCUMENT

Question: {question}

Respond with ONLY ONE WORD: SQL, DOCUMENT, or HYBRID

Classification:"""
        )
        
        # Register capabilities and handlers
        self._register_capabilities()
        self._register_handlers()
        
        logger.info("🎯 Smart Hybrid Orchestrator initialized")
    
    # ================================================================
    # AGENT WRAPPERS (A2A PROTOCOL)
    # ================================================================
    
    def _wrap_sql_agent(self) -> A2AAgent:
        """
        Wrap SQL Analyst for A2A protocol
        
        Creates an A2A-compatible agent that handles SQL queries
        """
        agent = A2AAgent(
            agent_id="sql_agent",
            name="SQL Analyst",
            description="Natural language to SQL queries"
        )
        
        agent.register_capability(AgentCapability(
            name="execute_sql",
            description="Execute SQL queries from natural language",
            input_schema={"type": "object", "properties": {"question": {"type": "string"}}},
            output_schema={"type": "object"}
        ))
        
        async def handle_query(message: A2AMessage) -> Dict[str, Any]:
            question = message.payload.get("question")
            allowed_tables = message.payload.get("allowed_tables")
            result = self.sql_agent.analyze(question=question, allowed_tables=allowed_tables)
            return result
        
        agent.register_handler(MessageType.QUERY, handle_query)
        return agent
    
    def _wrap_rag_agent(self) -> A2AAgent:
        """
        Wrap RAG Agent for A2A protocol
        
        Creates an A2A-compatible agent that handles document queries
        """
        agent = A2AAgent(
            agent_id="rag_agent",
            name="Document RAG Agent",
            description="Search and answer questions about session documents"
        )
        
        agent.register_capability(AgentCapability(
            name="query_documents",
            description="Query uploaded documents including CSV analytics",
            input_schema={"type": "object", "properties": {
                "question": {"type": "string"},
                "conversation_id": {"type": "string"}
            }},
            output_schema={"type": "object"}
        ))
        
        async def handle_query(message: A2AMessage) -> Dict[str, Any]:
            question = message.payload.get("question")
            conversation_id = message.payload.get("conversation_id")
            language = message.payload.get("language")
            
            result = self.rag_agent.query(conversation_id, question, language)
            return result
        
        agent.register_handler(MessageType.QUERY, handle_query)
        return agent
    
    # ================================================================
    # CAPABILITY REGISTRATION
    # ================================================================
    
    def _register_capabilities(self):
        """Register orchestrator capabilities"""
        self.register_capability(AgentCapability(
            name="route_query",
            description="Route queries intelligently with CSV hybrid support",
            input_schema={"type": "object", "properties": {
                "question": {"type": "string"},
                "conversation_id": {"type": "string"}
            }},
            output_schema={"type": "object"}
        ))
    
    def _register_handlers(self):
        """Register message handlers"""
        self.register_handler(MessageType.QUERY, self._handle_query_message)
    
    async def _handle_query_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle incoming query messages"""
        question = message.payload.get("question")
        conversation_id = message.payload.get("conversation_id")
        
        if not question:
            return {"error": "No question provided"}
        
        return await self.route_query(question, conversation_id)
    
    # ================================================================
    # KEYWORD DETECTION HELPERS
    # ================================================================
    
    def _check_document_keywords(self, question: str) -> bool:
        """
        Check for document-specific keywords
        
        Keywords that suggest the query is about uploaded documents
        rather than database tables.
        
        Returns:
            True if document keywords detected
        """
        document_keywords = [
            'meeting', 'strategic', 'planning', 'roadmap', 'review',
            'document', 'report', 'notes', 'file', 'presentation',
            'performance review', 'summary', 'overview', 'uploaded',
            'approved', 'decision', 'action item', 'target', 'goal',
            'initiative', 'project', 'plan', 'recommendation',
            'according to', 'mentions', 'states',
            'says about', 'from the', 'in the document', 'in the file',
            'in the roadmap', 'in the spreadsheet', 'in the excel'
        ]
        
        question_lower = question.lower()
        matches = []
        
        for keyword in document_keywords:
            if ' ' in keyword:  # Multi-word phrase
                if keyword in question_lower:
                    matches.append(keyword)
            else:  # Single word - use word boundary
                if re.search(rf'\b{re.escape(keyword)}\b', question_lower):
                    matches.append(keyword)
        
        if matches:
            logger.info(f"📄 Document keywords: {matches}")
            return True
        
        return False
    
    def _check_sql_keywords(self, question: str) -> bool:
        """
        Check for SQL-specific keywords
        
        Keywords that suggest the query needs database calculations
        rather than document understanding.
        
        Returns:
            True if SQL keywords detected
        """
        sql_phrases = [
            'how many', 'list all', 'show me', 'group by', 'order by',
            'total sales', 'average salary', 'in database', 'from database'
        ]
        
        sql_words = [
            'count', 'average', 'total', 'maximum', 'minimum', 'avg',
            'employee', 'employees', 'department', 'salary',
            'sales', 'product', 'customer', 'transaction',
            'find', 'filter'
        ]
        
        question_lower = question.lower()
        matches = []
        
        for phrase in sql_phrases:
            if phrase in question_lower:
                matches.append(phrase)
        
        if not matches:
            for word in sql_words:
                if re.search(rf'\b{re.escape(word)}\b', question_lower):
                    matches.append(word)
        
        if matches:
            logger.info(f"🗄️ SQL keywords: {matches}")
            return True
        
        return False
    
    # ================================================================
    # MAIN CLASSIFICATION LOGIC
    # ================================================================
    
    def _classify_query(self, question: str, conversation_id: str = None) -> str:
        """
        Classify query into SQL, DOCUMENT, or HYBRID
        
        Priority Order:
        1. HYBRID indicators (HIGHEST PRIORITY - NEW!)
           - Detects "definition + data" patterns
           - Detects comparison/correlation requests
           
        2. CSV router recommendations
           - For structured data (CSV/Excel)
           - Distinguishes calculations vs semantic understanding
           
        3. Keyword detection
           - Document-specific vs SQL-specific words
           
        4. LLM classification (FALLBACK)
           - Uses GPT for ambiguous cases
        
        Args:
            question: User's natural language question
            conversation_id: Session ID for context
            
        Returns:
            "SQL", "DOCUMENT", or "HYBRID"
        """
        # ========================================================
        # STEP 1: GET SESSION CONTEXT
        # ========================================================
        session_has_docs = False
        session_has_csv = False
        document_files = []
        
        if conversation_id:
            session_status = self.rag_agent.get_session_status(conversation_id)
            session_has_docs = session_status["vectorstore_ready"]
            session_has_csv = session_status.get("has_csv_data", False)
            document_files = session_status.get("loaded_files", [])
            
            logger.info(f"📊 Session context:")
            logger.info(f"   - Has documents: {session_has_docs}")
            logger.info(f"   - Has CSV data: {session_has_csv}")
            logger.info(f"   - Files: {document_files}")
        
        # ========================================================
        # PRIORITY 1: HYBRID INDICATORS (NEW - HIGHEST PRIORITY!)
        # ========================================================
        # This catches queries that need BOTH SQL data AND RAG explanation
        # Example: "Explain what AAVIS means and show top 5 districts"
        
        question_lower = question.lower()
        
        # Definition/explanation keywords
        definition_keywords = [
            'explain what', 'what does', 'what is', 'define', 
            'meaning of', 'means', 'significance of', 'represents'
        ]
        
        # Data/listing keywords
        data_keywords = [
            'show', 'list', 'top', 'districts', 'count', 
            'calculate', 'scores', 'data', 'numbers'
        ]
        
        # Comparison/correlation keywords (always HYBRID)
        comparison_keywords = [
            'compare', 'correlation', 'correlated', 'relationship',
            'factors', 'contribute', 'why', 'causes', 'difference between'
        ]
        
        # Check for each pattern
        has_definition = any(kw in question_lower for kw in definition_keywords)
        has_data = any(kw in question_lower for kw in data_keywords)
        has_comparison = any(kw in question_lower for kw in comparison_keywords)
        
        # HYBRID ROUTING RULES (only if CSV data available)
        if session_has_csv:
            # Rule 1: Comparison/correlation queries → HYBRID
            if has_comparison:
                logger.info("🔀 HYBRID: Comparison/correlation detected")
                return "HYBRID"
            
            # Rule 2: Definition + Data → HYBRID
            if has_definition and has_data:
                logger.info("🔀 HYBRID: Definition + Data detected")
                return "HYBRID"
            
            # Rule 3: "Factors" or "Why" questions → HYBRID
            if any(kw in question_lower for kw in ['factors', 'why', 'causes']):
                logger.info("🔀 HYBRID: Causal explanation detected")
                return "HYBRID"
        
        # ========================================================
        # PRIORITY 2: CSV ROUTER (For pure SQL or RAG on CSV)
        # ========================================================
        if session_has_csv and self.csv_router:
            csv_route = self.csv_router.route(question, has_csv_data=True)
            
            if csv_route == "SQL":
                logger.info("📊 CSV Hybrid Route: SQL (precise calculations)")
                return "SQL"
            elif csv_route == "RAG":
                logger.info("📄 CSV Hybrid Route: RAG (semantic understanding)")
                return "DOCUMENT"
            elif csv_route == "HYBRID":
                logger.info("🔀 CSV Hybrid Route: HYBRID (both needed)")
                return "HYBRID"
        
        # ========================================================
        # PRIORITY 3: KEYWORD DETECTION
        # ========================================================
        has_doc_keywords = self._check_document_keywords(question)
        has_sql_keywords = self._check_sql_keywords(question)
        
        if has_doc_keywords and session_has_docs:
            logger.info("🎯 Classification: DOCUMENT (keyword + has docs)")
            return "DOCUMENT"
        
        if has_sql_keywords and not has_doc_keywords and not session_has_docs:
            logger.info("🎯 Classification: SQL (clear SQL query)")
            return "SQL"
        
        # ========================================================
        # PRIORITY 4: LLM CLASSIFICATION (FALLBACK)
        # ========================================================
        try:
            prompt = self.routing_prompt.format(
                question=question,
                has_documents=str(session_has_docs),
                document_files=", ".join(document_files) if document_files else "None"
            )
            
            response = self.llm.invoke(prompt)
            classification = response.content.strip().upper()
            
            logger.info(f"🧭 LLM Classification: {classification}")
            
            if "DOCUMENT" in classification:
                if session_has_docs:
                    return "DOCUMENT"
                else:
                    logger.info("⚠️ LLM suggested DOCUMENT but no docs, using SQL")
                    return "SQL"
            elif "SQL" in classification:
                return "SQL"
            elif "HYBRID" in classification:
                if session_has_docs:
                    return "HYBRID"
                else:
                    return "SQL"
            else:
                # Default fallback
                if session_has_docs:
                    logger.info("📄 Defaulting to DOCUMENT (has docs)")
                    return "DOCUMENT"
                else:
                    logger.info("🗄️ Defaulting to SQL")
                    return "SQL"
        
        except Exception as e:
            logger.warning(f"⚠️ Classification error: {e}")
            if session_has_docs:
                return "DOCUMENT"
            return "SQL"
    
    # ================================================================
    # MAIN ROUTING METHOD
    # ================================================================
    
    async def route_query(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Main routing method - directs queries to appropriate agents
        
        Flow:
        1. Classify query (SQL/DOCUMENT/HYBRID)
        2. Route to appropriate agent(s)
        3. Combine results if HYBRID
        4. Return formatted response
        
        Args:
            question: User's natural language question
            conversation_id: Session ID for document context
            
        Returns:
            Dictionary with:
            - success: bool
            - answer/explanation: Response text
            - routing: "sql", "document", or "hybrid"
            - agent: Which agent(s) handled it
        """
        log_section("ORCHESTRATING QUERY")
        logger.info(f"❓ Question: {question}")
        logger.info(f"💬 Session: {conversation_id or 'None'}")
        
        # Classify the query
        classification = self._classify_query(question, conversation_id)
        
        try:
            # ====================================================
            # ROUTE 1: SQL ONLY
            # ====================================================
            if classification == "SQL":
                logger.info("📤 Routing to SQL Agent")
                
                # Get allowed tables for this session
                session_tables = self.rag_agent.get_csv_tables(conversation_id)
                
                # Send query to SQL agent
                message = await self.send_message(
                    to_agent="sql_agent",
                    message_type=MessageType.QUERY,
                    payload={
                        "question": question,
                        "allowed_tables": session_tables
                    }
                )
                
                # Get response
                response = await self.registry.route_message(message)
                result = response.payload
                
                # Add routing metadata
                result["routing"] = "sql"
                result["agent"] = "sql_agent"
                result["routing_reason"] = "Query about database records or uploaded CSV calculations"
            
            # ====================================================
            # ROUTE 2: DOCUMENT ONLY
            # ====================================================
            elif classification == "DOCUMENT":
                logger.info("📤 Routing to RAG Agent")
                
                # Send query to RAG agent
                message = await self.send_message(
                    to_agent="rag_agent",
                    message_type=MessageType.QUERY,
                    payload={
                        "question": question,
                        "conversation_id": conversation_id
                    }
                )
                
                # Get response
                response = await self.registry.route_message(message)
                result = response.payload
                
                # Add routing metadata
                result["routing"] = "document"
                result["agent"] = "rag_agent"
                result["routing_reason"] = "Query about documents or semantic understanding"
            
            # ====================================================
            # ROUTE 3: HYBRID (BOTH AGENTS)
            # ====================================================
            else:  # HYBRID
                logger.info("📤 Routing to BOTH agents (HYBRID)")
                
                # ============================================
                # Step 1: Query SQL Agent
                # ============================================
                session_tables = self.rag_agent.get_csv_tables(conversation_id)
                sql_message = await self.send_message(
                    to_agent="sql_agent",
                    message_type=MessageType.QUERY,
                    payload={
                        "question": question,
                        "allowed_tables": session_tables
                    }
                )
                
                # ============================================
                # Step 2: Query RAG Agent
                # ============================================
                rag_message = await self.send_message(
                    to_agent="rag_agent",
                    message_type=MessageType.QUERY,
                    payload={
                        "question": question,
                        "conversation_id": conversation_id
                    }
                )
                
                # ============================================
                # Step 3: Get Both Responses
                # ============================================
                sql_response = await self.registry.route_message(sql_message)
                rag_response = await self.registry.route_message(rag_message)
                
                # ============================================
                # Step 4: Combine Results
                # ============================================
                # Extract text from each agent
                sql_text = sql_response.payload.get(
                    "explanation",
                    "the database did not return a significant numerical result"
                )
                
                rag_text = rag_response.payload.get(
                    "answer",
                    "no additional contextual insight was found in the uploaded documents"
                )
                
                # Format combined response
                combined_answer = (
                    f"Quantitative Analysis:\n{sql_text.strip()}\n\n"
                    f"Contextual Insights:\n{rag_text.strip()}\n\n"
                    f"Conclusion: The data and context together provide a complete picture of the situation."
                )
                
                # Build result dictionary
                result = {
                    "success": True,
                    "question": question,
                    "answer": combined_answer,
                    "explanation": combined_answer,
                    "sql_result": sql_response.payload,
                    "rag_result": rag_response.payload,
                    "routing": "hybrid",
                    "agent": "both",
                    "routing_reason": "Query needs both precise calculations AND contextual understanding"
                }
            
            logger.info(f"✅ Routed to: {classification}")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Routing error: {e}")
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status for all agents
        
        Returns:
            Dictionary with status of orchestrator, SQL agent, RAG agent
        """
        rag_stats = self.rag_agent.get_global_stats()
        
        return {
            "orchestrator": {
                "agent_id": self.agent_id,
                "status": "active",
                "csv_hybrid_enabled": CSV_ROUTER_AVAILABLE
            },
            "sql_agent": {
                "status": "active",
                "database": settings.DB_PATH
            },
            "rag_agent": {
                "status": "active",
                "total_sessions": rag_stats["total_sessions"],
                "active_sessions": rag_stats["active_sessions"],
                "total_vectors": rag_stats["total_vectors"]
            },
            "a2a_registry": {
                "registered_agents": len(self.registry.agents),
                "agent_ids": list(self.registry.agents.keys())
            }
        }


# ====================================================================
# USAGE EXAMPLE
# ====================================================================
"""
Example usage:

    from database import DatabaseManager
    
    # Initialize
    db = DatabaseManager()
    orchestrator = OrchestratorAgent(db)
    
    # Route a query
    result = await orchestrator.route_query(
        question="Explain what AAVIS means and show top 5 districts",
        conversation_id="session-123"
    )
    
    # Access results
    print(result['routing'])  # "hybrid"
    print(result['answer'])   # Combined SQL + RAG response
    
Expected behavior for HYBRID queries:
- "Explain X and show top 5" → HYBRID (definition + data)
- "Compare A and B correlation" → HYBRID (comparison)
- "What factors contribute to Y?" → HYBRID (causal explanation)
- "What does X mean? List districts" → HYBRID (definition + list)

The orchestrator now prioritizes HYBRID detection BEFORE CSV routing,
ensuring questions that need both agents actually get routed to both.
"""