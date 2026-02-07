# main.py
"""
AI Business Analyst - Production FastAPI Server
Session-aware RAG with hybrid search, SQL routing, and chat history
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import shutil
import uvicorn
import asyncio

from database import DatabaseManager, create_sample_database
from agents import OrchestratorAgent
from mcp import MCPServer, MCPTool
from chat.chat_history import ChatHistoryManager
from config import settings
from utils.logger import logger, log_section

# ================================================================
# FASTAPI SETUP
# ================================================================

app = FastAPI(
    title="AI Business Analyst",
    description="Session-aware SQL + RAG with hybrid BM25+FAISS retrieval",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================================================================
# INITIALIZE CORE COMPONENTS
# ================================================================

db_manager = DatabaseManager()
orchestrator = OrchestratorAgent(db_manager)
chat_manager = ChatHistoryManager(
    db_path="./data/chat_history.db",
    redis_host="localhost",
    redis_port=6379
)
mcp = MCPServer(name="ai-business-analyst", version="3.0.0")

# ================================================================
# REQUEST MODELS
# ================================================================

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ConversationCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

# ================================================================
# UTILITIES
# ================================================================

def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """Extract user ID from header or use default"""
    return x_user_id or "default_user"

async def process_files_background(conversation_id: str, file_paths: List[str]):
    """Process uploaded files in background to avoid blocking response"""
    try:
        logger.info(f"📤 Background processing: {len(file_paths)} files")
        orchestrator.rag_agent.load_documents(conversation_id, file_paths)
        logger.info(f"✅ Processing complete: {conversation_id}")
    except Exception as e:
        logger.error(f"❌ Background processing error: {e}")

# ================================================================
# MCP TOOL HANDLERS
# ================================================================

class MCPToolHandlers:
    """Handlers for MCP tool calls"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    async def handle_smart_query(self, args: dict) -> dict:
        """Route query to SQL/RAG/Hybrid based on content"""
        question = args.get("question")
        conversation_id = args.get("conversation_id")
        
        if not question:
            return {
                "content": [{"type": "text", "text": "❌ No question provided"}],
                "isError": True
            }
        
        result = await self.orchestrator.route_query(question, conversation_id)
        
        if not result.get("success"):
            return {
                "content": [{"type": "text", "text": f"❌ Error: {result.get('error')}"}],
                "isError": True
            }
        
        # Format response based on routing
        routing = result.get("routing", "unknown")
        
        if routing == "sql":
            text = f"Database analysis: {result.get('explanation', 'No answer')}"
        elif routing == "document":
            text = f"Document search (session {conversation_id}): {result.get('answer', 'No answer')}"
        else:
            text = f"Hybrid analysis: {result.get('answer', 'No answer')}"
        
        return {"content": [{"type": "text", "text": text}], "isError": False}
    
    async def handle_get_status(self, args: dict) -> dict:
        """Get system status"""
        status = self.orchestrator.get_system_status()
        
        text = (
            f"**System Status**\n\n"
            f"SQL Agent: ✅\n"
            f"RAG Agent: ✅\n"
            f"Orchestrator: ✅\n\n"
            f"Sessions: {status['rag_agent']['total_sessions']}\n"
            f"Active: {status['rag_agent']['active_sessions']}\n"
            f"Vectors: {status['rag_agent']['total_vectors']}"
        )
        
        return {"content": [{"type": "text", "text": text}], "isError": False}

mcp_tools = MCPToolHandlers(orchestrator)

# Register MCP tools
mcp.register_tool(MCPTool(
    name="smart_query",
    description="Smart routing: SQL/Document/Hybrid",
    inputSchema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "conversation_id": {"type": "string"}
        },
        "required": ["question"]
    },
    handler=mcp_tools.handle_smart_query
))

mcp.register_tool(MCPTool(
    name="get_status",
    description="System status and session stats",
    inputSchema={"type": "object", "properties": {}},
    handler=mcp_tools.handle_get_status
))

# ================================================================
# API ENDPOINTS - CONVERSATIONS
# ================================================================

@app.get("/")
async def root():
    """Service info"""
    return {
        "service": "AI Business Analyst",
        "version": "3.0.0",
        "features": [
            "Session-aware RAG (isolated docs per chat)",
            "Hybrid BM25 + FAISS + reranking",
            "Smart SQL/RAG/Hybrid routing",
            "Multilingual support",
            "Chat history (Redis + SQLite)"
        ]
    }

@app.post("/conversations")
async def create_conversation(
    request: ConversationCreate,
    x_user_id: Optional[str] = Header(None)
):
    """Create new chat with isolated document space"""
    user_id = x_user_id or request.user_id
    conversation_id = chat_manager.create_conversation(user_id, request.title)
    orchestrator.rag_agent.get_or_create_session(conversation_id)
    
    logger.info(f"🆕 New conversation: {conversation_id}")
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "message": "Chat created"
    }

@app.get("/conversations")
async def get_conversations(
    x_user_id: Optional[str] = Header(None),
    limit: int = 20
):
    """List user's conversations with document counts"""
    user_id = get_user_id(x_user_id)
    conversations = chat_manager.get_user_conversations(user_id, limit)
    
    # Add document stats to each conversation
    for conv in conversations:
        session_status = orchestrator.rag_agent.get_session_status(conv["conversation_id"])
        conv["documents"] = session_status["total_documents"]
        conv["vectors"] = session_status["total_vectors"]
    
    return {
        "success": True,
        "conversations": conversations,
        "count": len(conversations)
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, limit: int = 50):
    """Get conversation with messages and session info"""
    conversation = chat_manager.get_conversation(conversation_id, limit)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Add session metadata
    session_status = orchestrator.rag_agent.get_session_status(conversation_id)
    conversation["session"] = session_status
    
    return {"success": True, "conversation": conversation}

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation and all documents"""
    try:
        chat_manager.delete_conversation(conversation_id)
        orchestrator.rag_agent.delete_session(conversation_id)
        
        logger.info(f"🗑️ Deleted: {conversation_id}")
        return {"success": True, "message": "Deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/documents")
async def get_conversation_documents(conversation_id: str):
    """List documents in conversation"""
    session_status = orchestrator.rag_agent.get_session_status(conversation_id)
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "documents": session_status["loaded_files"],
        "total_documents": session_status["total_documents"],
        "total_vectors": session_status["total_vectors"]
    }

# ================================================================
# API ENDPOINTS - QUERY & UPLOAD
# ================================================================

@app.post("/query")
async def query(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(None)
):
    """Smart query with session-aware routing"""
    try:
        user_id = get_user_id(x_user_id)
        
        # Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = chat_manager.create_conversation(user_id)
            orchestrator.rag_agent.get_or_create_session(conversation_id)
        
        # Save user message
        chat_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.question
        )
        
        # Route query (SQL/RAG/Hybrid)
        result = await orchestrator.route_query(request.question, conversation_id)
        
        # Save assistant response
        chat_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.get("explanation") or result.get("answer", ""),
            routing=result.get("routing"),
            metadata={
                "sql_query": result.get("sql_query"),
                "sources": result.get("sources", []),
                "routing_reason": result.get("routing_reason")
            }
        )
        
        result["conversation_id"] = conversation_id
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID")
):
    """Upload documents to conversation (background processing)"""
    
    if not conversation_id:
        raise HTTPException(
            status_code=400,
            detail="Conversation ID required"
        )
    
    upload_dir = Path("./uploads") / conversation_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    filenames = []
    
    try:
        # Save files (fast)
        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(str(file_path))
            filenames.append(file.filename)
            logger.info(f"✅ Saved: {file.filename}")
        
        # Process in background (slow)
        asyncio.create_task(
            process_files_background(conversation_id, file_paths)
        )
        
        # Return immediately
        return JSONResponse(content={
            "success": True,
            "message": "Files uploaded, processing in background",
            "loaded_files": filenames,
            "processing": True
        })
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================
# API ENDPOINTS - STATUS & ANALYTICS
# ================================================================

@app.get("/status")
async def system_status():
    """System status"""
    status = orchestrator.get_system_status()
    return JSONResponse(content=status)

@app.get("/analytics")
async def get_analytics(x_user_id: Optional[str] = Header(None)):
    """User analytics"""
    user_id = get_user_id(x_user_id)
    analytics = chat_manager.get_analytics(user_id)
    
    return {
        "success": True,
        "user_id": user_id,
        "analytics": analytics
    }

# ================================================================
# MCP ENDPOINTS
# ================================================================

@app.post("/mcp/initialize")
async def mcp_initialize(request: dict):
    """Initialize MCP connection"""
    return await mcp.handle_initialize(request)

@app.get("/mcp/tools/list")
async def mcp_list_tools():
    """List available MCP tools"""
    return await mcp.handle_list_tools()

@app.post("/mcp/tools/call")
async def mcp_call_tool(tool_name: str, arguments: dict):
    """Call MCP tool"""
    result = await mcp.handle_call_tool(tool_name, arguments)
    return JSONResponse(content=result)

@app.get("/mcp/info")
async def mcp_info():
    """MCP server info"""
    return mcp.get_server_info()

# ================================================================
# STATIC FILES & UI
# ================================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
async def serve_ui():
    """Serve web UI"""
    return FileResponse("static/index.html")

# ================================================================
# STARTUP
# ================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    log_section("AI BUSINESS ANALYST STARTING")
    
    import os
    if not os.path.exists(settings.DB_PATH):
        logger.info("📊 Creating sample database...")
        create_sample_database(settings.DB_PATH)
    
    logger.info(f"🚀 Server: {settings.HOST}:{settings.PORT}")
    logger.info(f"🗄️ SQL Agent: Ready")
    logger.info(f"📄 RAG Agent: Hybrid BM25+FAISS+Reranking")
    logger.info(f"🎯 Orchestrator: Ready")
    logger.info(f"💬 Chat History: Ready")
    logger.info(f"🔌 MCP: {len(mcp.tools)} tools")
    logger.info(f"🤖 A2A: {len(orchestrator.registry.agents)} agents")
    logger.info("="*70)

# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )