# main.py — AI Business Analyst API Server v3.3.1
#
# Endpoints:
#   POST   /conversations                       create session
#   GET    /conversations                       list user sessions
#   GET    /conversations/{cid}                 get session + messages
#   DELETE /conversations/{cid}                 delete session + cleanup
#   GET    /conversations/{cid}/documents
#   GET    /conversations/{cid}/processing-status
#   POST   /query                               sync query (dashboard / benchmark)
#   POST   /query/stream                        SSE streaming query (chat UI)
#   POST   /upload                              file ingest → background indexing
#   GET    /status                              full system status
#   GET    /api/metrics                         lightweight metrics for UI polling
#   GET    /analytics                           per-user analytics
#   POST   /mcp/initialize                      MCP handshake
#   GET    /mcp/tools/list                      MCP tool catalogue
#   POST   /mcp/tools/call                      MCP tool execution
#   GET    /mcp/info                            MCP server info
#   GET    /a2a/registry                        live A2A agent registry snapshot

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import shutil, uvicorn, asyncio, json

from database import DatabaseManager
from agents import OrchestratorAgent
from mcp import MCPServer, MCPTool
from chat.chat_history import ChatHistoryManager
from config import settings
from utils.logger import logger, log_section
from auth import verify_key, limiter


# ── Core singletons ───────────────────────────────────────────────────────────

db_manager   = DatabaseManager()
orchestrator = OrchestratorAgent(db_manager)
chat_manager = ChatHistoryManager(
    db_path="./data/chat_history.db",
    redis_host="localhost",
    redis_port=6379,
)
mcp = MCPServer(name="ai-business-analyst", version="3.0.0")

# Background processing state keyed by conversation_id
_processing_state: dict = {}


# ── Startup helpers ───────────────────────────────────────────────────────────

def _drop_tables(tables: list):
    conn   = db_manager.get_connection()
    cursor = conn.cursor()
    for t in tables:
        cursor.execute(f"DROP TABLE IF EXISTS [{t}]")
        logger.info(f"🧹 Dropped table: {t}")
    conn.commit()
    conn.close()


def _cleanup_orphan_tables():
    """Remove upload_ tables left over from previous runs."""
    try:
        orphans = [t for t in db_manager.list_tables() if t.startswith("upload_")]
        if orphans:
            _drop_tables(orphans)
            logger.info(f"🧹 Dropped {len(orphans)} orphan table(s)")
        else:
            logger.info("✅ No orphan tables")
    except Exception as e:
        logger.warning(f"⚠️  Orphan cleanup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_section("AI BUSINESS ANALYST STARTING")
    _cleanup_orphan_tables()
    logger.info(
        f"🚀  {settings.HOST}:{settings.PORT} | "
        f"MCP: {len(mcp.tools)} tools | "
        f"A2A: {len(orchestrator.registry.agents)} agents"
    )
    log_section("READY")
    yield
    logger.info("🛑  Shutdown complete")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="AI Business Analyst", version="3.3.1", lifespan=lifespan)

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ConversationCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

class MCPCallRequest(BaseModel):
    name: str
    arguments: dict = {}


def _get_user_id(x: Optional[str]) -> str:
    return x or "default_user"


# ── Background file processing ────────────────────────────────────────────────

async def _process_files_background(
    conversation_id: str,
    file_paths: List[str],
    file_names: List[str],
):
    _processing_state[conversation_id] = {
        "status": "processing", "files": file_names, "error": None
    }
    try:
        logger.info(f"📤  Background indexing: {file_names}")
        await asyncio.get_event_loop().run_in_executor(
            None,
            orchestrator.rag_agent.load_documents,
            conversation_id,
            file_paths,
        )
        _processing_state[conversation_id] = {
            "status": "ready", "files": file_names, "error": None
        }
        logger.info(f"✅  Indexing complete: {conversation_id}")
    except Exception as e:
        _processing_state[conversation_id] = {
            "status": "error", "files": file_names, "error": str(e)
        }
        logger.error(f"❌  Indexing error: {e}")


# ── MCP tool handlers ─────────────────────────────────────────────────────────

class MCPToolHandlers:
    """
    Handlers for the 3 registered MCP tools.

    All handlers call orchestrator.route_query() which dispatches through
    the A2A registry → target agent → result. The full chain for forecasting:

      POST /mcp/tools/call { name: "forecast_query" }
      → handle_forecast_query()
      → orchestrator.route_query()         [classifies as FORECAST, priority 0.5]
      → A2AMessage(to="forecast_agent")
      → registry.route_message()           [real A2A dispatch]
      → _wrap_forecast_agent handler
      → forecast_agent.forecast()          [Prophet + LLM narrative]
      → response flows back up the chain
    """

    def __init__(self, orch: OrchestratorAgent):
        self.orchestrator: OrchestratorAgent = orch

    async def handle_smart_query(self, args: dict) -> dict:
        """Route any question through the full 6-priority orchestrator pipeline."""
        question = args.get("question")
        conv_id  = args.get("conversation_id")
        if not question:
            return {"content": [{"type": "text", "text": "No question provided"}], "isError": True}
        result = await self.orchestrator.route_query(question, conv_id)
        text   = result.get("explanation") or result.get("answer") or "No answer generated."
        return {
            "content":  [{"type": "text", "text": text}],
            "isError":  not result.get("success", True),
        }

    async def handle_get_status(self, args: dict) -> dict:
        """Return current system and A2A agent status."""
        s    = self.orchestrator.get_system_status()
        text = (
            f"SQL Agent:      ✅ Active\n"
            f"RAG Agent:      ✅ Active\n"
            f"Forecast Agent: ✅ Active\n"
            f"A2A Agents:     {s['a2a_registry']['registered_agents']} registered\n"
            f"Sessions:       {s['rag_agent']['total_sessions']} | "
            f"Vectors: {s['rag_agent']['total_vectors']}"
        )
        return {"content": [{"type": "text", "text": text}], "isError": False}

    async def handle_forecast_query(self, args: dict) -> dict:
        """
        Trigger Prophet forecasting via MCP → A2A → forecast_agent.
        Requires conversation_id with an uploaded CSV.
        """
        question = args.get("question", "forecast the main metric for next 6 months")
        conv_id  = args.get("conversation_id", "")
        if not conv_id:
            return {
                "content": [{"type": "text", "text": "conversation_id is required"}],
                "isError": True,
            }
        result = await self.orchestrator.route_query(question, conv_id)
        text   = (
            result.get("explanation")
            or result.get("answer")
            or result.get("error", "Forecast failed")
        )
        return {
            "content": [{"type": "text", "text": text}],
            "isError": not result.get("success", False),
        }


# ── Register MCP tools ────────────────────────────────────────────────────────

mcp_handlers = MCPToolHandlers(orchestrator)

mcp.register_tool(MCPTool(
    name="smart_query",
    description="Route any business question to SQL, RAG, Hybrid, Forecast, or General automatically",
    inputSchema={
        "type": "object",
        "properties": {
            "question":        {"type": "string", "description": "Natural language question"},
            "conversation_id": {"type": "string", "description": "Active session ID"},
        },
        "required": ["question"],
    },
    handler=mcp_handlers.handle_smart_query,
))

mcp.register_tool(MCPTool(
    name="get_status",
    description="Return system status: agents, sessions, vector counts",
    inputSchema={"type": "object", "properties": {}},
    handler=mcp_handlers.handle_get_status,
))

mcp.register_tool(MCPTool(
    name="forecast_query",
    description=(
        "Forecast future values and detect anomalies from an uploaded CSV. "
        "Uses Prophet for time-series modelling with LLM narrative generation. "
        "Routed via A2A registry → forecast_agent. "
        "Requires an active conversation_id with a CSV uploaded."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "question": {
                "type":        "string",
                "description": "e.g. 'forecast revenue for next 6 months'",
            },
            "conversation_id": {
                "type":        "string",
                "description": "Active session ID with uploaded CSV",
            },
        },
        "required": ["question", "conversation_id"],
    },
    handler=mcp_handlers.handle_forecast_query,
))


# ── Basic routes ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "AI Business Analyst", "version": "3.3.1"}


@app.get("/ui")
async def serve_ui():
    return FileResponse("static/index.html")


# ── Conversation management ───────────────────────────────────────────────────

@app.post("/conversations")
async def create_conversation(
    request: ConversationCreate,
    x_user_id: Optional[str] = Header(None),
):
    user_id = x_user_id or request.user_id
    cid     = chat_manager.create_conversation(user_id, request.title)
    orchestrator.rag_agent.get_or_create_session(cid)
    logger.info(f"🆕  New conversation: {cid}")
    return {"success": True, "conversation_id": cid, "user_id": user_id}


@app.get("/conversations")
async def get_conversations(
    x_user_id: Optional[str] = Header(None),
    limit: int = 20,
):
    convs = chat_manager.get_user_conversations(_get_user_id(x_user_id), limit)
    for c in convs:
        s = orchestrator.rag_agent.get_session_status(c["conversation_id"])
        c["documents"] = s["total_documents"]
        c["vectors"]   = s["total_vectors"]
    return {"success": True, "conversations": convs, "count": len(convs)}


@app.get("/conversations/{cid}")
async def get_conversation(cid: str, limit: int = 50):
    conv = chat_manager.get_conversation(cid, limit)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    conv["session"] = orchestrator.rag_agent.get_session_status(cid)
    return {"success": True, "conversation": conv}


@app.delete("/conversations/{cid}")
async def delete_conversation(cid: str):
    chat_manager.delete_conversation(cid)
    orchestrator.rag_agent.delete_session(cid)
    try:
        to_drop = [t for t in db_manager.list_tables() if t.startswith(f"upload_{cid[:8]}_")]
        if to_drop:
            _drop_tables(to_drop)
    except Exception as e:
        logger.warning(f"⚠️  Table cleanup on delete failed: {e}")
    return {"success": True}


@app.get("/conversations/{cid}/documents")
async def get_documents(cid: str):
    s = orchestrator.rag_agent.get_session_status(cid)
    return {
        "success":         True,
        "conversation_id": cid,
        "documents":       s["loaded_files"],
        "total_documents": s["total_documents"],
        "total_vectors":   s["total_vectors"],
    }


@app.get("/conversations/{cid}/processing-status")
async def processing_status(cid: str):
    state = _processing_state.get(cid)
    if not state:
        s = orchestrator.rag_agent.get_session_status(cid)
        if s.get("total_documents", 0) > 0:
            return {"status": "ready", "files": s.get("loaded_files", []), "error": None}
        return {"status": "unknown", "files": [], "error": None}
    return {"status": state["status"], "files": state["files"], "error": state["error"]}


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.post("/query", dependencies=[Depends(verify_key)])
@limiter.limit("30/minute")
async def query(
    request: Request,
    body: QueryRequest,
    x_user_id: Optional[str] = Header(None),
):
    """
    Synchronous query — dashboard tab, benchmark.py, and MCP tool internals.
    Returns full JSON: data, plot spec, explanation, routing, sql_query, sources.
    """
    try:
        uid = _get_user_id(x_user_id)
        cid = body.conversation_id or chat_manager.create_conversation(uid)
        orchestrator.rag_agent.get_or_create_session(cid)

        chat_manager.add_message(conversation_id=cid, role="user", content=body.question)
        result = await orchestrator.route_query(body.question, cid)
        chat_manager.add_message(
            conversation_id=cid,
            role="assistant",
            content=result.get("explanation") or result.get("answer", ""),
            routing=result.get("routing"),
            metadata={
                "sql_query": result.get("sql_query"),
                "sources":   result.get("sources", []),
            },
        )
        result["conversation_id"] = cid
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌  Query error: {e}")
        raise HTTPException(500, str(e))


@app.post("/query/stream", dependencies=[Depends(verify_key)])
@limiter.limit("20/minute")
async def query_stream(
    request: Request,
    body: QueryRequest,
    x_user_id: Optional[str] = Header(None),
):
    """
    SSE streaming endpoint for the Chat UI.

    Event sequence:
      event: meta   → JSON with routing, plot, data, sql_query, sources
      event: token  → one word chunk of the explanation text
      event: done   → end of stream signal
      event: error  → on exception

    Note: slowapi requires the starlette Request parameter to be named
    exactly "request". The Pydantic body uses "body" to avoid conflict.
    """
    uid = _get_user_id(x_user_id)
    cid = body.conversation_id or chat_manager.create_conversation(uid)
    orchestrator.rag_agent.get_or_create_session(cid)
    chat_manager.add_message(conversation_id=cid, role="user", content=body.question)

    async def event_generator():
        full_text = ""
        try:
            result = await orchestrator.route_query(body.question, cid)

            # Send non-text fields first so the UI can render routing tag
            # and charts before the explanation text starts streaming.
            meta = {
                "success":         result.get("success", True),
                "routing":         result.get("routing", ""),
                "plot":            result.get("plot"),
                "data":            result.get("data", []),
                "sql_query":       result.get("sql_query"),
                "sources":         result.get("sources", []),
                "conversation_id": cid,
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            # Stream explanation word-by-word. route_query() runs the full
            # pipeline synchronously then we re-stream the result. A small
            # sleep between words creates smooth progressive UX without
            # requiring per-token streaming changes in each agent.
            explanation = (result.get("explanation") or result.get("answer") or "").strip()
            words = explanation.split(" ")
            for i, word in enumerate(words):
                chunk      = word + (" " if i < len(words) - 1 else "")
                full_text += chunk
                yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.018)

            yield f"event: done\ndata: {json.dumps({'conversation_id': cid})}\n\n"

            chat_manager.add_message(
                conversation_id=cid,
                role="assistant",
                content=full_text,
                routing=result.get("routing"),
                metadata={
                    "sql_query": result.get("sql_query"),
                    "sources":   result.get("sources", []),
                },
            )

        except Exception as e:
            logger.error(f"❌  Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload", dependencies=[Depends(verify_key)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    files: List[UploadFile] = File(...),
    conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
):
    if not conversation_id:
        raise HTTPException(400, "X-Conversation-ID header required")

    upload_dir = Path("./uploads") / conversation_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths, names = [], []
    for f in files:
        dest = upload_dir / f.filename
        with open(dest, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        paths.append(str(dest))
        names.append(f.filename)
        logger.info(f"✅  Saved: {f.filename}")

    asyncio.create_task(_process_files_background(conversation_id, paths, names))
    return JSONResponse({"success": True, "loaded_files": names, "processing": True})


# ── Status / metrics ──────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return JSONResponse(orchestrator.get_system_status())


@app.get("/api/metrics")
async def api_metrics():
    sys = orchestrator.get_system_status()
    return {
        "success":        True,
        "sql_agent":      sys["sql_agent"]["status"],
        "rag_agent":      sys["rag_agent"]["status"],
        "total_sessions": sys["rag_agent"]["total_sessions"],
        "total_vectors":  sys["rag_agent"]["total_vectors"],
        "a2a_agents":     sys["a2a_registry"]["registered_agents"],
    }


@app.get("/analytics")
async def analytics(x_user_id: Optional[str] = Header(None)):
    return {
        "success":   True,
        "analytics": chat_manager.get_analytics(_get_user_id(x_user_id)),
    }


# ── MCP endpoints ─────────────────────────────────────────────────────────────

@app.post("/mcp/initialize")
async def mcp_initialize(req: dict):
    return await mcp.handle_initialize(req)


@app.get("/mcp/tools/list")
async def mcp_tools_list():
    return await mcp.handle_list_tools()


@app.post("/mcp/tools/call")
async def mcp_tools_call(body: MCPCallRequest):
    return JSONResponse(await mcp.handle_call_tool(body.name, body.arguments))


@app.get("/mcp/info")
async def mcp_info():
    return mcp.get_server_info()


# ── A2A registry ──────────────────────────────────────────────────────────────

@app.get("/a2a/registry")
async def a2a_registry():
    """
    Live snapshot of the A2A agent registry.

    Returns all registered agents with their capabilities and peer connections.
    Registered at startup:
      sql_agent       — execute_sql
      rag_agent       — query_documents
      forecast_agent  — forecast, anomaly_detection

    All inter-agent calls go through registry.route_message() using
    structured A2AMessage envelopes — no direct method calls.
    """
    registered = {
        agent_id: {
            "agent_id":        agent.agent_id,
            "name":            agent.name,
            "description":     agent.description,
            "capabilities":    list(agent.capabilities.keys()),
            "connected_peers": list(agent.connected_agents.keys()),
        }
        for agent_id, agent in orchestrator.registry.agents.items()
    }
    return {
        "status":           "active",
        "total_agents":     len(registered),
        "agents":           registered,
        "protocol_version": "a2a-v1",
        "mcp_tools":        list(mcp.tools.keys()),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
