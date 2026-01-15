import re
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from protocols import A2AAgent, AgentCapability, MessageType, A2AMessage, A2ARegistry
from agents.sql_analyst import SQLAnalystAgent
from agents.session_rag import SessionAwareRAGAgent
from database.db_manager import DatabaseManager
from config import settings, ENABLE_HYBRID_CSV
from utils.logger import logger, log_section

# Import CSV router if hybrid enabled
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
    
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__(
            agent_id="orchestrator",
            name="Smart Query Orchestrator",
            description="Routes queries intelligently to SQL, RAG, or HYBRID"
        )
        
        self.registry = A2ARegistry()
        self.sql_agent = SQLAnalystAgent(db_manager)
        self.rag_agent = SessionAwareRAGAgent(db_manager=db_manager)
        
        # Initialize CSV router
        if CSV_ROUTER_AVAILABLE:
            self.csv_router = CSVQueryRouter()
            logger.info("✅ CSV Hybrid Router enabled")
        else:
            self.csv_router = None
            logger.info("ℹ️ CSV Hybrid Router disabled")
        
        self.sql_agent_a2a = self._wrap_sql_agent()
        self.rag_agent_a2a = self._wrap_rag_agent()
        
        self.registry.register_agent(self.sql_agent_a2a)
        self.registry.register_agent(self.rag_agent_a2a)
        
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0
        )
        
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
        
        self._register_capabilities()
        self._register_handlers()
        
        logger.info("🎯 Smart Hybrid Orchestrator initialized")
    
    def _wrap_sql_agent(self) -> A2AAgent:
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
            result = self.sql_agent.analyze(question)
            return result
        
        agent.register_handler(MessageType.QUERY, handle_query)
        return agent
    
    def _wrap_rag_agent(self) -> A2AAgent:
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
    
    def _register_capabilities(self):
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
        self.register_handler(MessageType.QUERY, self._handle_query_message)
    
    async def _handle_query_message(self, message: A2AMessage) -> Dict[str, Any]:
        question = message.payload.get("question")
        conversation_id = message.payload.get("conversation_id")
        
        if not question:
            return {"error": "No question provided"}
        
        return await self.route_query(question, conversation_id)
    
    def _check_document_keywords(self, question: str) -> bool:
        """Check for document-specific keywords"""
        document_keywords = [
            'meeting', 'strategic', 'planning', 'roadmap', 'review',
            'document', 'report', 'notes', 'file', 'presentation',
            'performance review', 'summary', 'overview', 'uploaded',
            'approved', 'decision', 'action item', 'target', 'goal',
            'initiative', 'project', 'plan', 'recommendation',
            'according to', 'what does', 'mentions', 'states',
            'says about', 'from the', 'in the document', 'in the file',
            'in the roadmap', 'in the spreadsheet', 'in the excel'
        ]
        
        question_lower = question.lower()
        matches = []
        
        for keyword in document_keywords:
            if ' ' in keyword:
                if keyword in question_lower:
                    matches.append(keyword)
            else:
                if re.search(rf'\b{re.escape(keyword)}\b', question_lower):
                    matches.append(keyword)
        
        if matches:
            logger.info(f"📄 Document keywords: {matches}")
            return True
        
        return False
    
    def _check_sql_keywords(self, question: str) -> bool:
        """Check SQL keywords"""
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
    
    def _classify_query(self, question: str, conversation_id: str = None) -> str:
        
        #Priority:
        #1. CSV Hybrid Routing (if session has CSV data)
        #2. Document keyword detection
        #3. SQL keyword detection
        #4. LLM-based classification
        
        
        # Get session context
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
        
        # PRIORITY 1: CSV Hybrid Routing
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
        
        # PRIORITY 2: Document keywords
        has_doc_keywords = self._check_document_keywords(question)
        has_sql_keywords = self._check_sql_keywords(question)
        
        if has_doc_keywords and session_has_docs:
            logger.info("🎯 Classification: DOCUMENT (keyword + has docs)")
            return "DOCUMENT"
        
        if has_sql_keywords and not has_doc_keywords and not session_has_docs:
            logger.info("🎯 Classification: SQL (clear SQL query)")
            return "SQL"
        
        # PRIORITY 3: LLM classification
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
    
    async def route_query(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """Route query with hybrid CSV support"""
        log_section("ORCHESTRATING QUERY")
        logger.info(f"❓ Question: {question}")
        logger.info(f"💬 Session: {conversation_id or 'None'}")
        
        classification = self._classify_query(question, conversation_id)
        
        try:
            if classification == "SQL":
                logger.info("📤 Routing to SQL Agent")
                
                message = await self.send_message(
                    to_agent="sql_agent",
                    message_type=MessageType.QUERY,
                    payload={"question": question}
                )
                
                response = await self.registry.route_message(message)
                result = response.payload
                result["routing"] = "sql"
                result["agent"] = "sql_agent"
                result["routing_reason"] = "Query about database records or uploaded CSV calculations"
                
            elif classification == "DOCUMENT":
                logger.info("📤 Routing to RAG Agent")
                
                message = await self.send_message(
                    to_agent="rag_agent",
                    message_type=MessageType.QUERY,
                    payload={
                        "question": question,
                        "conversation_id": conversation_id
                    }
                )
                
                response = await self.registry.route_message(message)
                result = response.payload
                result["routing"] = "document"
                result["agent"] = "rag_agent"
                result["routing_reason"] = "Query about documents or semantic understanding"
                
            else:  # HYBRID
                logger.info("📤 Routing to BOTH agents (HYBRID)")
                
                # Query SQL agent
                sql_message = await self.send_message(
                    to_agent="sql_agent",
                    message_type=MessageType.QUERY,
                    payload={"question": question}
                )
                
                # Query RAG agent
                rag_message = await self.send_message(
                    to_agent="rag_agent",
                    message_type=MessageType.QUERY,
                    payload={
                        "question": question,
                        "conversation_id": conversation_id
                    }
                )
                
                # Get responses
                sql_response = await self.registry.route_message(sql_message)
                rag_response = await self.registry.route_message(rag_message)
                
                # Combine results
                combined_answer = "### 🗄️ From Database/Calculations:\n"
                combined_answer += sql_response.payload.get("explanation", "No SQL result")
                combined_answer += "\n\n### 📄 From Documents/Context:\n"
                combined_answer += rag_response.payload.get("answer", "No document result")
                
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
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