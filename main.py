

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import shutil
import uvicorn

from database import DatabaseManager, create_sample_database
from agents import OrchestratorAgent
from mcp import MCPServer, MCPTool, MCPResource
from chat.chat_history import ChatHistoryManager
from config import settings
from utils.logger import logger, log_section

# ============== INITIALIZE FASTAPI ==============

app = FastAPI(
    title="AI Business Analyst - Session Aware",
    description="Multilingual SQL + RAG Agent with isolated chat sessions",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== INITIALIZE SYSTEM ==============

db_manager = DatabaseManager()
orchestrator = OrchestratorAgent(db_manager)

chat_manager = ChatHistoryManager(
    db_path="./data/chat_history.db",
    redis_host="localhost",
    redis_port=6379
)

mcp = MCPServer(name="ai-business-analyst", version="2.2.0")

# ============== PYDANTIC MODELS ==============

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ConversationCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

class UploadRequest(BaseModel):
    conversation_id: str

# ============== HELPER FUNCTIONS ==============

def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    return x_user_id or "default_user"

# ============== MCP TOOL HANDLERS ==============

class MCPToolHandlers:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    async def handle_smart_query(self, args: dict) -> dict:
        question = args.get("question")
        conversation_id = args.get("conversation_id")
        
        if not question:
            return {"content": [{"type": "text", "text": "❌ No question provided"}], "isError": True}
        
        result = await self.orchestrator.route_query(question, conversation_id)
        
        if not result.get("success"):
            return {"content": [{"type": "text", "text": f"❌ Error: {result.get('error')}"}], "isError": True}
        
        routing = result.get("routing", "unknown")
        if routing == "sql":
            text = f"**🗄️ SQL Result**\n\n{result.get('explanation', 'No answer')}"
        elif routing == "document":
            text = f"**📄 Document Result (Session: {conversation_id})**\n\n{result.get('answer', 'No answer')}"
        else:
            text = f"**🔀 Combined Result**\n\n{result.get('answer', 'No answer')}"
        
        return {"content": [{"type": "text", "text": text}], "isError": False}
    
    async def handle_get_status(self, args: dict) -> dict:
        status = self.orchestrator.get_system_status()
        
        text = "**System Status**\n\n"
        text += f"✅ SQL Agent: Active\n"
        text += f"✅ RAG Agent: Active\n"
        text += f"✅ Orchestrator: Active\n\n"
        text += f"📊 Total Sessions: {status['rag_agent']['total_sessions']}\n"
        text += f"🔢 Active Sessions: {status['rag_agent']['active_sessions']}\n"
        text += f"📦 Total Vectors: {status['rag_agent']['total_vectors']}\n"
        
        return {"content": [{"type": "text", "text": text}], "isError": False}

mcp_tools = MCPToolHandlers(orchestrator)

# ============== REGISTER MCP TOOLS ==============

mcp.register_tool(MCPTool(
    name="smart_query",
    description="Ask any question - automatically routes to SQL or Documents (session-aware)",
    inputSchema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Your question in any language"},
            "conversation_id": {"type": "string", "description": "Conversation ID for document context"}
        },
        "required": ["question"]
    },
    handler=mcp_tools.handle_smart_query
))

mcp.register_tool(MCPTool(
    name="get_status",
    description="Get system status including session statistics",
    inputSchema={"type": "object", "properties": {}},
    handler=mcp_tools.handle_get_status
))

# ============== REST API ENDPOINTS ==============

@app.get("/")
async def root():
    return {
        "service": "AI Business Analyst - Session Aware",
        "version": "2.2.0",
        "features": [
            "✅ Isolated document space per chat session",
            "✅ Multilingual SQL queries",
            "✅ Session-aware document RAG",
            "✅ Smart routing (SQL/Document/Hybrid)",
            "✅ Chat history with Redis",
            "✅ MCP & A2A protocols"
        ],
        "new": "Each chat now has its own document storage!"
    }

@app.post("/conversations")
async def create_conversation(
    request: ConversationCreate,
    x_user_id: Optional[str] = Header(None)
):
    """Create new conversation with isolated document space"""
    user_id = x_user_id or request.user_id
    conversation_id = chat_manager.create_conversation(user_id, request.title)
    
    # Initialize RAG session
    orchestrator.rag_agent.get_or_create_session(conversation_id)
    
    logger.info(f"🆕 New conversation created: {conversation_id}")
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "message": "New chat created with isolated document space"
    }

@app.get("/conversations")
async def get_conversations(
    x_user_id: Optional[str] = Header(None),
    limit: int = 20
):
    """Get user's conversations"""
    user_id = get_user_id(x_user_id)
    conversations = chat_manager.get_user_conversations(user_id, limit)
    
    # Add session info to each conversation
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
    
    # Add session info
    session_status = orchestrator.rag_agent.get_session_status(conversation_id)
    conversation["session"] = session_status
    
    return {
        "success": True,
        "conversation": conversation
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation AND its document storage"""
    try:
        # Delete chat history
        chat_manager.delete_conversation(conversation_id)
        
        # Delete RAG session
        orchestrator.rag_agent.delete_session(conversation_id)
        
        logger.info(f"🗑️ Deleted conversation and session: {conversation_id}")
        
        return {
            "success": True,
            "message": "Conversation and documents deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/documents")
async def get_conversation_documents(conversation_id: str):
    """Get documents loaded in this conversation"""
    session_status = orchestrator.rag_agent.get_session_status(conversation_id)
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "documents": session_status["loaded_files"],
        "total_documents": session_status["total_documents"],
        "total_vectors": session_status["total_vectors"]
    }

@app.post("/query")
async def query(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(None)
):
    """
    Smart Query with Session-Aware Documents
    """
    try:
        user_id = get_user_id(x_user_id)
        
        # Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = chat_manager.create_conversation(user_id)
            # Initialize session
            orchestrator.rag_agent.get_or_create_session(conversation_id)
        
        # Save user message
        chat_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.question
        )
        
        # Route via orchestrator (WITH conversation_id)
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
    """Upload documents to a specific conversation"""
    
    if not conversation_id:
        raise HTTPException(
            status_code=400, 
            detail="Conversation ID required. Please create or select a chat first."
        )
    
    upload_dir = Path("./uploads") / conversation_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    
    try:
        for file in files:
            file_path = upload_dir / file.filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(str(file_path))
            logger.info(f"✅ Saved to session {conversation_id}: {file.filename}")
        
        # Load into RAG agent for THIS conversation
        result = orchestrator.rag_agent.load_documents(conversation_id, file_paths)
        
        logger.info(f"📄 Loaded {len(file_paths)} files into session {conversation_id}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def system_status():
    """Get system status"""
    status = orchestrator.get_system_status()
    return JSONResponse(content=status)

@app.get("/analytics")
async def get_analytics(x_user_id: Optional[str] = Header(None)):
    """Get user analytics"""
    user_id = get_user_id(x_user_id)
    analytics = chat_manager.get_analytics(user_id)
    
    return {
        "success": True,
        "user_id": user_id,
        "analytics": analytics
    }

# ============== MCP ENDPOINTS ==============

@app.post("/mcp/initialize")
async def mcp_initialize(request: dict):
    return await mcp.handle_initialize(request)

@app.get("/mcp/tools/list")
async def mcp_list_tools():
    return await mcp.handle_list_tools()

@app.post("/mcp/tools/call")
async def mcp_call_tool(tool_name: str, arguments: dict):
    result = await mcp.handle_call_tool(tool_name, arguments)
    return JSONResponse(content=result)

@app.get("/mcp/info")
async def mcp_info():
    return mcp.get_server_info()

# ============== STATIC FILES ==============

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
async def serve_ui():
    return FileResponse("static/index_2.html")

# ============== STARTUP ==============

@app.on_event("startup")
async def startup_event():
    log_section("AI BUSINESS ANALYST STARTING (SESSION-AWARE)")
    
    import os
    if not os.path.exists(settings.DB_PATH):
        logger.info("📊 Creating sample database...")
        create_sample_database(settings.DB_PATH)
    
    logger.info(f"🚀 Server: {settings.HOST}:{settings.PORT}")
    logger.info(f"🗄️ SQL Agent: Ready")
    logger.info(f"📄 RAG Agent: Session-Aware Mode")
    logger.info(f"🎯 Orchestrator: Ready")
    logger.info(f"💬 Chat History: Ready")
    logger.info(f"🔌 MCP Server: {len(mcp.tools)} tools")
    logger.info(f"🤖 A2A Registry: {len(orchestrator.registry.agents)} agents")
    logger.info("✨ Each chat now has isolated document storage!")
    logger.info("="*70)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )
