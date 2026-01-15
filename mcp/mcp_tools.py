# mcp/tool_handlers.py

from typing import Dict, Any
from utils.logger import logger


class MCPToolHandlers:
    
    
    def __init__(self, orchestrator):
        
        self.orchestrator = orchestrator
        logger.info("🛠️  MCP Tool Handlers initialized")
    
    async def handle_smart_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        
        question = args.get("question")
        
        if not question:
            return self._error_response("No question provided")
        
        logger.info(f"🛠️  Smart Query: {question[:50]}...")
        
        # Route via orchestrator (A2A magic happens here)
        result = await self.orchestrator.route_query(question)
        
        if not result.get("success"):
            return self._error_response(result.get("error", "Query failed"))
        
        # Format based on routing
        return self._format_query_result(result, question)
    
    async def handle_query_sql(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query SQL database directly"""
        question = args.get("question")
        
        if not question:
            return self._error_response("No question provided")
        
        result = self.orchestrator.sql_agent.analyze(question)
        
        if not result.get("success"):
            return self._error_response(result.get("error"))
        
        return self._format_sql_result(result)
    
    async def handle_query_documents(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Query documents directly"""
        question = args.get("question")
        
        if not question:
            return self._error_response("No question provided")
        
        result = self.orchestrator.rag_agent.query(question)
        
        if not result.get("success"):
            return self._error_response(result.get("error"))
        
        return self._format_document_result(result)
    
    async def handle_get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status"""
        status = self.orchestrator.get_system_status()
        
        status_text = "**System Status**\n\n"
        status_text += f"✅ SQL Agent: Active\n"
        status_text += f"✅ RAG Agent: {status['rag_agent']['status'].title()}\n"
        status_text += f"✅ Orchestrator: Active\n"
        status_text += f"\n📊 Documents: {status['rag_agent']['total_documents']}\n"
        status_text += f"🔢 Vectors: {status['rag_agent']['total_vectors']}\n"
        
        return {
            "content": [{"type": "text", "text": status_text}],
            "isError": False
        }
    
    # Helper methods for clean code
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Standard error response"""
        return {
            "content": [{"type": "text", "text": f"❌ Error: {error}"}],
            "isError": True
        }
    
    def _format_query_result(self, result: Dict, question: str) -> Dict[str, Any]:
        """Format smart query result"""
        routing = result.get("routing", "unknown")
        
        if routing == "sql":
            text = f"**🗄️ SQL Result**\n\n{result.get('explanation', 'No answer')}"
        elif routing == "document":
            text = f"**📄 Document Result**\n\n{result.get('answer', 'No answer')}"
        else:  # hybrid
            text = f"**🔀 Combined Result**\n\n{result.get('answer', 'No answer')}"
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False
        }
    
    def _format_sql_result(self, result: Dict) -> Dict[str, Any]:
        """Format SQL result"""
        text = f"**SQL Query:**\n```sql\n{result.get('sql_query', 'N/A')}\n```\n\n"
        text += f"**Answer:**\n{result.get('explanation', 'No answer')}"
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False
        }
    
    def _format_document_result(self, result: Dict) -> Dict[str, Any]:
        """Format document result"""
        text = f"{result.get('answer', 'No answer')}\n"
        
        sources = result.get("sources", [])
        if sources:
            text += f"\n**Sources:** {len(sources)} documents"
        
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False
        }


class MCPResourceHandlers:
    """Simple resource handlers"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        logger.info("📦 MCP Resource Handlers initialized")
    
    async def handle_database_schema(self) -> str:
        """Get database schema"""
        return self.orchestrator.sql_agent.db.get_schema_text()
    
    async def handle_document_list(self) -> str:
        """Get loaded documents list"""
        import json
        status = self.orchestrator.rag_agent.get_status()
        return json.dumps({
            "loaded_files": status.get("loaded_files", []),
            "total_documents": status.get("total_documents", 0)
        }, indent=2)