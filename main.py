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
from typing import List, Optional, Any
from pathlib import Path
import math, os, re, shutil, uvicorn, asyncio, json

from database import DatabaseManager
from agents import OrchestratorAgent
from mcp import MCPServer, MCPTool
from chat.chat_history import ChatHistoryManager
from parsers.hybrid_csv_processor import HybridCSVProcessor
from config import settings
from utils.logger import logger, log_section
from auth import (
    verify_key, limiter, enforce_daily_quota,
    auth_required, key_is_valid, quota_status,
)


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
_background_tasks: set = set()
_MAX_DASHBOARD_BYTES = 5 * 1024 * 1024
_MAX_UPLOAD_BYTES    = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024


# ── Input safety helpers ──────────────────────────────────────────────────────

# conversation_id is used to build filesystem paths (uploads/<cid>,
# data/vector_stores/<cid>) and is rmtree'd on delete — so it must never be
# able to contain path separators or be "." / "..".
_CID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _require_cid(cid: Optional[str]) -> str:
    if not cid or not _CID_RE.match(cid):
        raise HTTPException(400, "Invalid conversation_id")
    return cid


def _safe_filename(name: Optional[str]) -> str:
    """Reduce a client-supplied filename to a bare basename (no ../ or ..\\)."""
    base = Path((name or "").replace("\\", "/")).name.strip()
    if base in ("", ".", ".."):
        raise HTTPException(400, f"Invalid filename: {name!r}")
    return base


def _json_safe(obj: Any) -> Any:
    """Replace NaN/±inf floats (invalid JSON) with None, recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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
    """
    Remove upload_ tables whose session no longer exists on disk.

    Tables belonging to a session that still has a persisted vector store are
    kept, so the session's SQL/forecast features keep working after a restart.
    """
    try:
        vs_root     = Path(orchestrator.rag_agent.base_vector_store_path)
        live_prefix = {
            HybridCSVProcessor.session_table_prefix(d.name)
            for d in vs_root.iterdir() if d.is_dir()
        } if vs_root.exists() else set()
        orphans = [
            t for t in db_manager.list_tables()
            if t.startswith(settings.CSV_TABLE_PREFIX)
            and not any(t.startswith(p) for p in live_prefix)
        ]
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
    value_col: Optional[str] = None  # explicit column for forecast tab
    want_chart: bool = False         # dashboard: always build a chart from the data

class ConversationCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

class ConversationRename(BaseModel):
    title: str

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
        result = await asyncio.to_thread(
            orchestrator.rag_agent.load_documents,
            conversation_id,
            file_paths,
        )
        orchestrator.sql_agent.invalidate_schema_cache(conversation_id)  # only clear THIS session's cache
        errors = result.get("errors") or []
        if not result.get("success"):
            msg = "; ".join(errors) or result.get("message", "No documents could be loaded")
            _processing_state[conversation_id] = {
                "status": "error", "files": file_names, "error": msg
            }
            logger.error(f"❌  Indexing failed: {msg}")
            return
        _processing_state[conversation_id] = {
            "status": "ready", "files": file_names,
            # Partial success: some files loaded, others failed
            "error": "; ".join(errors) if errors else None,
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
        if conv_id is not None and not _CID_RE.match(str(conv_id)):
            return {"content": [{"type": "text", "text": "Invalid conversation_id"}], "isError": True}
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
        if not conv_id or not _CID_RE.match(str(conv_id)):
            return {
                "content": [{"type": "text", "text": "A valid conversation_id is required"}],
                "isError": True,
            }
        result = await self.orchestrator.route_query(
            question, conv_id, value_col=args.get("value_col"))
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
    _require_cid(cid)
    conv = chat_manager.get_conversation(cid, limit)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    conv["session"] = orchestrator.rag_agent.get_session_status(cid)
    return {"success": True, "conversation": conv}


@app.delete("/conversations/{cid}", dependencies=[Depends(verify_key)])
async def delete_conversation(cid: str):
    _require_cid(cid)
    chat_manager.delete_conversation(cid)
    orchestrator.rag_agent.delete_session(cid)
    orchestrator.sql_agent.invalidate_schema_cache(cid)
    _processing_state.pop(cid, None)
    try:
        prefix  = HybridCSVProcessor.session_table_prefix(cid)
        to_drop = [t for t in db_manager.list_tables() if t.startswith(prefix)]
        if to_drop:
            _drop_tables(to_drop)
    except Exception as e:
        logger.warning(f"⚠️  Table cleanup on delete failed: {e}")
    return {"success": True}


@app.patch("/conversations/{cid}", dependencies=[Depends(verify_key)])
async def rename_conversation(cid: str, body: ConversationRename):
    _require_cid(cid)
    title = " ".join(body.title.split())[:80]
    if not title:
        raise HTTPException(400, "Title can't be empty")
    if not chat_manager.rename_conversation(cid, title):
        raise HTTPException(404, "Conversation not found")
    return {"success": True, "conversation_id": cid, "title": title}


@app.get("/conversations/{cid}/documents")
async def get_documents(cid: str):
    _require_cid(cid)
    s = orchestrator.rag_agent.get_session_status(cid)
    return {
        "success":         True,
        "conversation_id": cid,
        "documents":       s["loaded_files"],
        "total_documents": s["total_documents"],
        "total_vectors":   s["total_vectors"],
    }



@app.get("/conversations/{cid}/schema")
async def get_schema(cid: str):
    """
    Return fully-classified columns for all CSV tables in this session.
    Classifies every column as: date | numeric | categorical | id
    Used by Forecast tab (numeric_columns) and Dashboard (all_columns).
    """
    _require_cid(cid)
    import pandas as pd

    DATE_KEYWORDS   = {"date","time","month","year","week","day","created","updated","period","timestamp","dt"}
    ID_KEYWORDS     = {"id","key","code","uuid","ref","index","no","num","number","serial"}
    NUMERIC_TYPES   = {"INT","REAL","FLOAT","DOUBLE","NUMERIC","DECIMAL","NUMBER","BIGINT","SMALLINT"}

    try:
        csv_tables = orchestrator.rag_agent.get_csv_tables(cid)
        all_columns   = []   # full classified list
        numeric_cols  = []
        date_cols     = []
        categorical   = []
        table_names   = []
        schema        = orchestrator.sql_agent.db.get_schema() if csv_tables else {}

        for table in csv_tables:
            cols     = schema.get(table, [])
            table_names.append(table)

            # Sample a few rows to help with type inference
            try:
                sample_df = orchestrator.sql_agent.db.execute_query(
                    f'SELECT * FROM "{table}" LIMIT 5'
                )
            except Exception:
                sample_df = pd.DataFrame()

            for col in cols:
                name      = col["name"]
                col_type  = col["type"].upper()
                name_low  = name.lower().replace("_"," ").replace("-"," ")
                name_words= set(name_low.split())

                # Determine role
                is_id   = bool(name_words & ID_KEYWORDS) or name_low.endswith(" id")
                is_date = (
                    bool(name_words & DATE_KEYWORDS)
                    or any(t in col_type for t in ("DATE","TIME","TIMESTAMP"))
                )
                is_num  = any(t in col_type for t in NUMERIC_TYPES)

                # Sniff from sample data if type is TEXT/empty
                if not is_num and not is_date and not sample_df.empty and name in sample_df.columns:
                    series = sample_df[name].dropna()
                    if len(series):
                        # Try numeric
                        converted = pd.to_numeric(series, errors="coerce")
                        if converted.notna().sum() == len(series):
                            is_num = True
                        else:
                            # Try date
                            try:
                                pd.to_datetime(series, errors="raise")
                                is_date = True
                            except Exception:
                                pass

                if is_id:
                    role = "id"
                elif is_date:
                    role = "date"
                    date_cols.append(name)
                elif is_num:
                    role = "numeric"
                    if not is_id:
                        numeric_cols.append(name)
                else:
                    role = "categorical"
                    categorical.append(name)

                all_columns.append({
                    "name":       name,
                    "type":       col["type"] or "TEXT",
                    "role":       role,
                    "table":      table,
                    "primary_key": col.get("primary_key", False),
                })

        return {
            "success":          True,
            "conversation_id":  cid,
            "tables":           table_names,
            "all_columns":      all_columns,
            "numeric_columns":  list(dict.fromkeys(numeric_cols)),
            "date_columns":     list(dict.fromkeys(date_cols)),
            "categorical_columns": list(dict.fromkeys(categorical)),
        }
    except Exception as e:
        logger.error(f"❌ Schema endpoint error: {e}")
        return {"success": False, "numeric_columns": [], "all_columns": [], "error": str(e)}


@app.get("/conversations/{cid}/numeric-columns")
async def get_numeric_columns(cid: str):
    """
    Numeric (forecastable) columns across this session's CSV tables.
    Called by the Forecast tab in both the React frontend and static/index.html
    to populate its column selector.
    """
    s = await get_schema(cid)
    return {
        "success": s.get("success", False),
        "columns": s.get("numeric_columns", []),
        **({"error": s["error"]} if s.get("error") else {}),
    }


@app.get("/conversations/{cid}/profile")
async def get_profile(cid: str):
    """
    Instant, LLM-free profile of the session's data for the BI overview:
    KPIs with sparklines, column roles + stats, auto charts, suggested
    questions, and sample rows. Read-only; computed with pandas.
    """
    _require_cid(cid)
    from tools.data_profile import build_profile
    try:
        status = orchestrator.rag_agent.get_session_status(cid)
        result = await asyncio.to_thread(
            build_profile, db_manager, orchestrator.rag_agent.get_csv_tables(cid), status["loaded_files"])
        return JSONResponse(_json_safe({"success": True, "conversation_id": cid, **result}))
    except Exception as e:
        logger.error(f"❌ Profile error: {e}")
        return JSONResponse({"success": False, "error": str(e), "datasets": [], "documents": [], "files": []})


@app.post("/query/silent", dependencies=[Depends(verify_key), Depends(enforce_daily_quota)])
@limiter.limit("60/minute")
async def query_silent(
    request: Request,
    body: QueryRequest,
    x_user_id: Optional[str] = Header(None),
):
    """
    Silent query — runs the full pipeline but saves NO chat history.
    Used by the Dashboard / Forecast tabs so backend queries don't leak into
    the user's chat thread.
    """
    cid = _require_cid(body.conversation_id)
    try:
        orchestrator.rag_agent.get_or_create_session(cid)
        result = await orchestrator.route_query(
            body.question, cid, value_col=body.value_col, want_chart=body.want_chart)
        result["conversation_id"] = cid
        return JSONResponse(content=_json_safe(result))
    except Exception as e:
        logger.error(f"❌ Silent query error: {e}")
        raise HTTPException(500, str(e))


@app.get("/conversations/{cid}/dashboard")
async def get_dashboard(cid: str):
    """Load persisted dashboard state for a session."""
    _require_cid(cid)
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect("./data/chat_history.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT state FROM dashboards WHERE conversation_id = ?", (cid,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            import json as _json
            return {"success": True, "state": _json.loads(row[0])}
        return {"success": True, "state": None}
    except Exception as e:
        return {"success": False, "state": None, "error": str(e)}


@app.post("/conversations/{cid}/dashboard", dependencies=[Depends(verify_key)])
async def save_dashboard(cid: str, request: Request):
    """Persist dashboard state for a session."""
    _require_cid(cid)
    import sqlite3 as _sqlite3
    try:
        body = await request.json()
        state_json = json.dumps(_json_safe(body.get("state", {})))
        if len(state_json) > _MAX_DASHBOARD_BYTES:
            return {"success": False, "error": "Dashboard state too large"}
        conn = _sqlite3.connect("./data/chat_history.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dashboards (conversation_id, state, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(conversation_id) DO UPDATE SET
                state      = excluded.state,
                updated_at = CURRENT_TIMESTAMP
        """, (cid, state_json))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/conversations/{cid}/processing-status")
async def processing_status(cid: str):
    _require_cid(cid)
    state = _processing_state.get(cid)
    if not state:
        s = orchestrator.rag_agent.get_session_status(cid)
        if s.get("total_documents", 0) > 0:
            return {"status": "ready", "files": s.get("loaded_files", []), "error": None}
        return {"status": "unknown", "files": [], "error": None}
    return {"status": state["status"], "files": state["files"], "error": state["error"]}


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.post("/query", dependencies=[Depends(verify_key), Depends(enforce_daily_quota)])
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
    if body.conversation_id:
        _require_cid(body.conversation_id)
    try:
        uid = _get_user_id(x_user_id)
        cid = body.conversation_id or chat_manager.create_conversation(uid)
        orchestrator.rag_agent.get_or_create_session(cid)

        chat_manager.add_message(conversation_id=cid, role="user", content=body.question)
        result = await orchestrator.route_query(body.question, cid, value_col=body.value_col)
        chat_manager.add_message(
            conversation_id=cid,
            role="assistant",
            content=result.get("explanation") or result.get("answer", ""),
            routing=result.get("routing"),
            plot=_json_safe(result.get("plot")),   # so the chart survives a reload
            metadata={
                "sql_query": result.get("sql_query"),
                "sources":   result.get("sources", []),
            },
        )
        result["conversation_id"] = cid
        return JSONResponse(content=_json_safe(result))

    except Exception as e:
        logger.error(f"❌  Query error: {e}")
        raise HTTPException(500, str(e))


@app.post("/query/stream", dependencies=[Depends(verify_key), Depends(enforce_daily_quota)])
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
    if body.conversation_id:
        _require_cid(body.conversation_id)
    uid = _get_user_id(x_user_id)
    cid = body.conversation_id or chat_manager.create_conversation(uid)
    orchestrator.rag_agent.get_or_create_session(cid)
    chat_manager.add_message(conversation_id=cid, role="user", content=body.question)

    async def event_generator():
        try:
            result = await orchestrator.route_query(body.question, cid, value_col=body.value_col)

            # Send non-text fields first so the UI can render routing tag
            # and charts before the explanation text starts streaming.
            # _json_safe: json.dumps would otherwise emit bare NaN, which the
            # browser's JSON.parse rejects (silently dropping the chart).
            meta = _json_safe({
                "success":         result.get("success", True),
                "routing":         result.get("routing", ""),
                "plot":            result.get("plot"),
                "data":            result.get("data", []),
                "sql_query":       result.get("sql_query"),
                "sources":         result.get("sources", []),
                "error":           result.get("error"),
                "conversation_id": cid,
            })
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            explanation = (
                result.get("explanation") or result.get("answer")
                or (f"Sorry, something went wrong: {result['error']}" if result.get("error") else "")
            ).strip()

            # Persist before streaming so a client disconnect mid-stream
            # doesn't lose the assistant message.
            chat_manager.add_message(
                conversation_id=cid,
                role="assistant",
                content=explanation,
                routing=result.get("routing"),
                plot=meta.get("plot"),   # so the chart (and its Pin button) survives a reload
                metadata={
                    "sql_query": result.get("sql_query"),
                    "sources":   result.get("sources", []),
                },
            )

            # Stream explanation word-by-word. route_query() runs the full
            # pipeline then we re-stream the result. A small sleep between
            # words creates smooth progressive UX without requiring
            # per-token streaming changes in each agent.
            words = explanation.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.018)

            yield f"event: done\ndata: {json.dumps({'conversation_id': cid})}\n\n"

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
    _require_cid(conversation_id)

    upload_dir = Path("./uploads") / conversation_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    paths, names = [], []
    for f in files:
        # Never trust the client's filename as a path: "../../main.py" or
        # "..\\main.py" would otherwise write outside the session folder.
        name = _safe_filename(f.filename)
        dest = upload_dir / name
        # Stream to disk with a size cap — an unbounded upload can fill the
        # disk and the embedding step would take minutes on a huge file.
        written = 0
        with open(dest, "wb") as buf:
            while chunk := f.file.read(1024 * 1024):
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    break
                buf.write(chunk)
        if written > _MAX_UPLOAD_BYTES:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                413, f"{name} is larger than the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
        paths.append(str(dest))
        names.append(name)
        logger.info(f"✅  Saved: {name}")

    # Mark as processing immediately so a status poll fired right after this
    # response can't see a stale "ready" from a previous upload.
    _processing_state[conversation_id] = {"status": "processing", "files": names, "error": None}
    task = asyncio.create_task(_process_files_background(conversation_id, paths, names))
    _background_tasks.add(task)               # keep a strong ref (asyncio docs)
    task.add_done_callback(_background_tasks.discard)
    return JSONResponse({"success": True, "loaded_files": names, "processing": True})


# ── Status / metrics ──────────────────────────────────────────────────────────

@app.get("/auth/check")
async def auth_check(x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")):
    """
    Lets the frontend decide whether to ask the user for an access key.
    required=False means open demo mode (API_KEYS unset).
    """
    return {"required": auth_required(), "valid": key_is_valid(x_api_key), "quota": quota_status()}


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


@app.post("/mcp/tools/call", dependencies=[Depends(verify_key), Depends(enforce_daily_quota)])
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
    uvicorn.run(
        app, host=settings.HOST, port=settings.PORT, log_level="info",
        # Behind a hosting proxy, trust its X-Forwarded-For so per-IP rate
        # limits see real client IPs instead of one shared proxy IP.
        # Set FORWARDED_ALLOW_IPS="*" only when the app is reachable solely
        # through that proxy.
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )