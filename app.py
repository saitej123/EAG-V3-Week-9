from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
import os
import threading
import uuid
from typing import Any
from loguru import logger
import sys
from dotenv import load_dotenv

from super_browser.llm_env import agent_run_max_seconds

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

_templates_dir = BASE_DIR / "templates"

_RUNTIME_MODULES = ("faiss", "trafilatura", "httpx", "playwright", "yaml")


def _missing_runtime_modules() -> list[str]:
    missing: list[str] = []
    for name in _RUNTIME_MODULES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def _project_venv_site_on_path() -> bool:
    """True when this process loads packages from ``<project>/.venv`` (uv uses a shared base python)."""
    site = BASE_DIR / ".venv" / "lib"
    if not site.is_dir():
        return False
    for entry in sys.path:
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if site.resolve() in resolved.parents or resolved == site.resolve():
            return True
    return False


def _runtime_env_hint() -> str:
    missing = _missing_runtime_modules()
    if missing and not _project_venv_site_on_path():
        return (
            "Wrong Python environment — packages are not loaded from this project's .venv. "
            "Stop uvicorn and run: ./scripts/serve.sh"
        )
    if missing:
        return "Install deps with: uv sync && ./scripts/serve.sh"
    return "Install deps with: uv sync && ./scripts/serve.sh"


def _runtime_env_detail() -> str | None:
    missing = _missing_runtime_modules()
    if not missing:
        return None
    return f"Missing Python packages: {', '.join(missing)}. {_runtime_env_hint()}"


templates = Jinja2Templates(directory=str(_templates_dir))

_agent_lock = threading.Lock()
_super_browser_agent = None


def _agent_mode() -> str:
    return (os.environ.get("AGENT_MODE") or "dag").strip().lower()


def _get_super_browser_agent():
    """Load agent (GenAI + MCP / DAG) only when needed."""
    global _super_browser_agent
    if _super_browser_agent is not None:
        return _super_browser_agent
    with _agent_lock:
        if _super_browser_agent is None:
            if _agent_mode() == "loop":
                from super_browser.agent import SuperBrowserAgent

                _super_browser_agent = SuperBrowserAgent()
            else:
                from super_browser.flow import DagAgent

                _super_browser_agent = DagAgent()
        return _super_browser_agent

# SSE queue — created on app startup so it binds to the uvicorn event loop (not import time).
log_queue: asyncio.Queue[str] | None = None
_app_loop_holder: dict[str, asyncio.AbstractEventLoop | None] = {"loop": None}


class QueueSink:
    """Send formatted log lines to the SSE queue from any thread."""

    def write(self, message: str) -> None:
        text = message.rstrip("\r\n")
        if not text:
            return
        q = log_queue
        loop = _app_loop_holder.get("loop")
        if q is None or loop is None or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(_enqueue_log, q, text)
        except RuntimeError:
            pass


def _enqueue_log(q: asyncio.Queue[str], text: str) -> None:
    try:
        q.put_nowait(text)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(text)
        except asyncio.QueueFull:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global log_queue
    log_queue = asyncio.Queue(maxsize=4096)
    _app_loop_holder["loop"] = asyncio.get_running_loop()
    detail = _runtime_env_detail()
    app.state.runtime_error = detail
    if detail:
        logger.error(f"[startup] {detail}")
    yield
    _app_loop_holder["loop"] = None
    log_queue = None
    agent = _super_browser_agent
    if agent is not None:
        try:
            if hasattr(agent, "aclose"):
                await agent.aclose()
            elif hasattr(agent, "action"):
                await agent.action.aclose()
        except Exception:
            pass


app = FastAPI(title="Super Browser Agent", lifespan=lifespan)
app.mount("/Images", StaticFiles(directory=str(BASE_DIR / "Images")), name="images")
_static_dir = BASE_DIR / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Console: ANSI colors when tty. SSE/UI sink: plain text (no markup) for reliable browser rendering.
_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<fg #e2e8f0>{message}</fg #e2e8f0>"
)
_SSE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
logger.remove()
logger.level("DEBUG", color="<fg #94a3b8>")
logger.level("INFO", color="<fg #86efac>")
logger.level("SUCCESS", color="<fg #4ade80>")
logger.level("WARNING", color="<fg #fbbf24>")
logger.level("ERROR", color="<fg #fb7185>")
logger.add(
    sys.stdout,
    format=_LOG_FORMAT,
    colorize=sys.stdout.isatty(),
)
logger.add(
    QueueSink(),
    format=_SSE_LOG_FORMAT,
    colorize=False,
    level="INFO",
)

if not _templates_dir.is_dir():
    logger.warning(f"Templates directory not found at {_templates_dir}; GET / may fail.")

class QueryRequest(BaseModel):
    query: str
    query_id: str | None = None


class ResumeRequest(BaseModel):
    session_id: str
    from_node_id: str | None = None
    replay_formatter: bool = False


class VisionRequest(BaseModel):
    prompt: str
    image_base64: str | None = None
    image: str | None = None
    mime_type: str = "image/png"
    temperature: float = 0.2
    max_tokens: int = 512


# Only one agent run / index job at a time (atomic under _ops_lock).
_ops_lock = threading.Lock()
_run_busy = False
_index_busy = False
_run_task: asyncio.Task | None = None


def _is_run_busy() -> bool:
    with _ops_lock:
        return _run_busy


def _is_index_busy() -> bool:
    with _ops_lock:
        return _index_busy


def _clear_stale_run_busy() -> None:
    """After SIGKILL the worker restarts; if the task finished but the flag stuck, clear it."""
    with _ops_lock:
        global _run_busy, _run_task
        if _run_busy and _run_task is not None and _run_task.done():
            logger.warning("[agent] Clearing stale run_busy (background task already finished)")
            _run_busy = False
            _run_task = None


def _force_end_run() -> None:
    with _ops_lock:
        global _run_busy, _run_task
        _run_busy = False
        _run_task = None


async def _stop_active_run() -> bool:
    """Cancel the in-flight agent task and clear run_busy (required before resume)."""
    global _run_task
    with _ops_lock:
        task = _run_task
        _run_busy = False
        _run_task = None
    if task is None or task.done():
        return False
    task.cancel()
    try:
        _done, pending = await asyncio.wait([task], timeout=10.0)
        for p in pending:
            p.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
    await asyncio.sleep(0.2)
    return True


async def _stop_agent_run(*, log_user_stop: bool = True) -> dict[str, Any]:
    """Cancel in-flight agent, checkpoint DAG running nodes → pending, clear busy flags."""
    cancelled = await _stop_active_run()
    if not cancelled:
        _force_end_run()
    session_id: str | None = None
    if _agent_mode() == "dag":
        try:
            from super_browser.graph_viz import latest_dag_session_id
            from super_browser.persistence import SessionStore

            sid = latest_dag_session_id()
            if sid:
                store = SessionStore(sid)
                if store.exists():
                    store.reset_running_to_pending()
                    session_id = sid
        except Exception as e:
            logger.warning(f"[agent] stop: session checkpoint failed: {e}")
    if cancelled and log_user_stop:
        logger.info("[agent] RUN_COMPLETE reason=user_stop")
    return {
        "status": "success",
        "cancelled": cancelled,
        "agent_busy": _is_run_busy(),
        "index_busy": _is_index_busy(),
        "session_id": session_id,
    }


def _try_begin_run() -> JSONResponse | None:
    runtime_error = _runtime_env_detail()
    if runtime_error:
        return JSONResponse({"status": "error", "detail": runtime_error}, status_code=503)
    _clear_stale_run_busy()
    with _ops_lock:
        global _run_busy
        if _run_busy:
            return JSONResponse(
                {"status": "busy", "detail": "An agent run is already in progress. Wait for it to finish."},
                status_code=429,
            )
        if _index_busy:
            return JSONResponse(
                {
                    "status": "busy",
                    "detail": "Indexing is in progress. Wait for it to finish before starting an agent run.",
                },
                status_code=429,
            )
        _run_busy = True
    return None


def _end_run() -> None:
    with _ops_lock:
        global _run_busy, _run_task
        _run_busy = False
        _run_task = None


def _start_run_task(coro) -> asyncio.Task:
    global _run_task
    task = asyncio.create_task(coro)
    _run_task = task
    return task


def _try_begin_index() -> JSONResponse | None:
    runtime_error = _runtime_env_detail()
    if runtime_error:
        return JSONResponse({"status": "error", "detail": runtime_error}, status_code=503)
    with _ops_lock:
        global _index_busy
        if _run_busy:
            return JSONResponse(
                {"status": "busy", "detail": "Cannot modify index while an agent run is in progress."},
                status_code=400,
            )
        if _index_busy:
            return JSONResponse(
                {"status": "busy", "detail": "Another indexing job is already in progress."},
                status_code=429,
            )
        _index_busy = True
    return None


def _end_index() -> None:
    with _ops_lock:
        global _index_busy
        _index_busy = False


def _index_busy_guard() -> JSONResponse | None:
    """Read-only check for endpoints that do not hold the index lock (e.g. upload)."""
    with _ops_lock:
        if _run_busy:
            return JSONResponse(
                {"status": "busy", "detail": "Cannot modify index while an agent run is in progress."},
                status_code=400,
            )
        if _index_busy:
            return JSONResponse(
                {"status": "busy", "detail": "Another indexing job is already in progress."},
                status_code=429,
            )
    runtime_error = _runtime_env_detail()
    if runtime_error:
        return JSONResponse({"status": "error", "detail": runtime_error}, status_code=503)
    return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/run-agent")
async def run_agent(request: QueryRequest):
    blocked = _try_begin_run()
    if blocked:
        return blocked
    qid = (request.query_id or "").strip() or None
    session_id = f"dag_{qid}_{uuid.uuid4().hex[:8]}" if qid else None
    logger.info(
        f"[UI] Starting agent"
        + (f" query_id={qid}" if qid else "")
        + f" session={session_id or '(auto)'}"
        + f" text={request.query[:120]!r}"
    )

    async def _job():
        try:
            agent = _get_super_browser_agent()
            if _agent_mode() == "dag":
                await asyncio.wait_for(
                    agent.run(request.query, session_id=session_id),
                    timeout=agent_run_max_seconds(),
                )
            else:
                await asyncio.wait_for(
                    agent.run(request.query),
                    timeout=agent_run_max_seconds(),
                )
        except asyncio.TimeoutError:
            logger.error(
                f"[agent] Global time budget exceeded ({agent_run_max_seconds()}s) — run stopped"
            )
            try:
                from super_browser.flow import log_final_answer

                log_final_answer(
                    "## Run stopped (time budget)\n\n"
                    "This run exceeded the configured wall-clock limit (`AGENT_RUN_MAX_SECONDS`). "
                    "See **Live console** for partial progress; narrow the query or raise the limit in `.env`."
                )
            except Exception:
                pass
            logger.info("[agent] RUN_COMPLETE reason=global_timeout")
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            logger.opt(exception=e).error("[agent] Run failed")
            try:
                from super_browser.flow import log_final_answer

                log_final_answer(
                    "## Run failed\n\n"
                    "An unexpected error stopped the agent. See **Live console** for the traceback."
                )
            except Exception:
                pass
            logger.info("[agent] RUN_COMPLETE reason=run_failed")
        finally:
            _end_run()

    try:
        _start_run_task(_job())
    except Exception:
        _end_run()
        raise
    return {"status": "Agent started"}


@app.post("/api/agent/stop")
async def api_agent_stop():
    """Stop the running agent (user cancel) — like Cursor stop; DAG running nodes → pending on disk."""
    body = await _stop_agent_run(log_user_stop=True)
    logger.info(f"[UI] Agent stop requested — cancelled={body.get('cancelled')}")
    return body


@app.post("/api/dag/unlock")
async def api_dag_unlock():
    """Stop in-flight agent (if any) and clear agent_busy — alias for resume/stuck recovery."""
    return await _stop_agent_run(log_user_stop=False)


@app.post("/run-agent/resume")
async def resume_agent(request: ResumeRequest):
    """Resume a persisted DAG session from state/sessions/<id>/ (running → pending)."""
    sid = (request.session_id or "").strip()
    from_node_id = (request.from_node_id or "").strip() or None
    if not sid:
        return JSONResponse({"status": "error", "detail": "session_id is required"}, status_code=400)
    if _agent_mode() != "dag":
        return JSONResponse(
            {"status": "error", "detail": "Resume is only supported when AGENT_MODE=dag"},
            status_code=400,
        )

    from super_browser.graph_viz import graph_viz_payload, prepare_session_for_resume

    await _stop_active_run()
    meta = prepare_session_for_resume(sid, from_node_id=from_node_id)
    if not meta.get("resume_enabled"):
        detail = meta.get("resume_disabled_reason") or (
            "Session cannot be resumed from current node states on disk."
        )
        return JSONResponse({"status": "error", "detail": detail}, status_code=400)

    try:
        graph_body = await asyncio.to_thread(graph_viz_payload, sid)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=404)

    blocked = _try_begin_run()
    if blocked:
        return blocked

    if from_node_id:
        logger.info(f"[UI] Resuming DAG session {sid!r} (rewind from {from_node_id!r})")
    else:
        logger.info(f"[UI] Resuming DAG session {sid!r} (checkpoint: running → pending)")

    async def _job():
        try:
            await asyncio.wait_for(
                _get_super_browser_agent().resume(sid),
                timeout=agent_run_max_seconds(),
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[agent] Global time budget exceeded on resume ({agent_run_max_seconds()}s)"
            )
            try:
                from super_browser.flow import log_final_answer

                log_final_answer(
                    "## Run stopped (time budget)\n\n"
                    "Resume exceeded the configured wall-clock limit (`AGENT_RUN_MAX_SECONDS`). "
                    "See **Live console** for partial progress."
                )
            except Exception:
                pass
            logger.info("[agent] RUN_COMPLETE reason=resume_timeout")
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            logger.opt(exception=e).error("[agent] Resume failed")
            try:
                from super_browser.flow import log_final_answer

                log_final_answer(
                    "## Resume failed\n\n"
                    "Could not continue the session. See **Live console** for details."
                )
            except Exception:
                pass
            logger.info("[agent] RUN_COMPLETE reason=resume_failed")
        finally:
            _end_run()

    try:
        _start_run_task(_job())
    except Exception:
        _end_run()
        raise
    return {
        "status": "Agent resumed",
        "session_id": sid,
        "resume_meta": meta,
        "graph": {"status": "success", **graph_body},
    }


@app.get("/stream-logs")
async def stream_logs():
    sse_headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    async def log_generator():
        ping_every = 12.0
        while True:
            q = log_queue
            if q is None:
                break
            try:
                message = await asyncio.wait_for(q.get(), timeout=ping_every)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if not message:
                continue
            lines = message.split("\n")
            chunk = "".join(f"data: {line}\n" for line in lines) + "\n"
            yield chunk

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers=sse_headers,
    )


@app.post("/reset-state")
async def reset_state():
    with _ops_lock:
        if _run_busy:
            return JSONResponse(
                {"status": "busy", "detail": "Cannot reset state while an agent run is in progress."},
                status_code=400,
            )
        if _index_busy:
            return JSONResponse(
                {"status": "busy", "detail": "Cannot reset state while indexing is in progress."},
                status_code=400,
            )
    import shutil
    state_dir = Path(BASE_DIR / "state")
    if state_dir.exists():
        try:
            await asyncio.to_thread(shutil.rmtree, state_dir)
            await _reload_agent_memory_async()
            logger.warning("[UI] State cleared (memory.json, index.faiss, index_ids.json).")
            return {"status": "success", "detail": "Durable state cleared (memory.json, FAISS index, and ids)."}
        except Exception as e:
            logger.error(f"[UI] Failed to clear state: {e}")
            return JSONResponse({"status": "error", "detail": f"Failed to clear state: {e}"}, status_code=500)
    await _reload_agent_memory_async()
    return {"status": "success", "detail": "State directory already empty or non-existent."}


async def _reload_agent_memory_async() -> None:
    """Reload FAISS/memory from disk after reset (DAG and loop agents)."""
    agent = _super_browser_agent
    if agent is not None and hasattr(agent, "memory"):
        await asyncio.to_thread(agent.memory._load_disk)
        return
    from super_browser.memory import MemoryManager

    await asyncio.to_thread(MemoryManager()._load_disk)


@app.get("/api/dag/sessions")
async def api_dag_sessions():
    """List persisted DAG sessions (newest first) for the graph viewer."""
    try:
        from super_browser.graph_viz import list_dag_sessions

        sessions = await asyncio.to_thread(list_dag_sessions)
        return {"status": "success", "sessions": sessions}
    except Exception as e:
        logger.error(f"[UI] DAG sessions list failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/dag/graph")
async def api_dag_graph(session_id: str | None = None):
    """Cytoscape/dagre graph payload for a persisted session (defaults to latest)."""
    try:
        from super_browser.graph_viz import graph_viz_payload, latest_dag_session_id
        from super_browser.persistence import SessionLoadError

        sid = (session_id or "").strip() or latest_dag_session_id()
        if not sid:
            return JSONResponse(
                {"status": "error", "detail": "No DAG sessions found. Run a query first."},
                status_code=404,
            )
        payload = await asyncio.to_thread(graph_viz_payload, sid)
        return {
            "status": "success",
            **payload,
            "agent_busy": _is_run_busy(),
            "index_busy": _is_index_busy(),
        }
    except SessionLoadError as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=404)
    except Exception as e:
        logger.error(f"[UI] DAG graph failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/dag/browser-replay")
async def api_dag_browser_replay(session_id: str, node_id: str | None = None):
    """Browser replay report (path, actions, cost, comparison table)."""
    try:
        from super_browser.browser.replay import build_browser_replay_report
        from super_browser.graph_viz import latest_dag_session_id

        sid = (session_id or "").strip() or latest_dag_session_id()
        if not sid:
            return JSONResponse(
                {"status": "error", "detail": "No DAG sessions found."},
                status_code=404,
            )
        report = await asyncio.to_thread(
            build_browser_replay_report,
            sid,
            node_id=(node_id or "").strip() or None,
        )
        return {"status": "success", **report}
    except FileNotFoundError as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=404)
    except Exception as e:
        logger.error(f"[UI] browser replay failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/queries/dag")
async def api_dag_queries():
    """Browser comparison assignment queries (COMP, creative comparisons, cascade demos)."""
    try:
        from super_browser.catalog import assignment_payload, validate_assignment_corpus

        issues = validate_assignment_corpus()
        if issues:
            return JSONResponse(
                {"status": "error", "detail": "; ".join(issues)},
                status_code=500,
            )
        return {"status": "success", **assignment_payload()}
    except Exception as e:
        logger.error(f"[UI] DAG queries failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.post("/api/browser/reseed-sessions")
async def api_browser_reseed_sessions():
    """Re-create browser reference sessions (dag_*_ref) after state reset — UI replay demos."""
    with _ops_lock:
        if _run_busy:
            return JSONResponse(
                {"status": "busy", "detail": "Cannot reseed while an agent run is in progress."},
                status_code=400,
            )
    try:
        from scripts.browser.seed_browser_sessions import seed_browser_reference_sessions

        created = await asyncio.to_thread(seed_browser_reference_sessions)
        return {"status": "success", "session_ids": created}
    except Exception as e:
        logger.error(f"[UI] browser reseed failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/api/queries/browser")
async def api_browser_queries():
    """Browser comparison assignment queries (COMP, creative comparisons, cascade demos)."""
    try:
        from super_browser.catalog import browser_queries_payload, validate_assignment_corpus

        issues = validate_assignment_corpus()
        if issues:
            return JSONResponse(
                {"status": "error", "detail": "; ".join(issues)},
                status_code=500,
            )
        return {"status": "success", **browser_queries_payload()}
    except Exception as e:
        logger.error(f"[UI] Browser queries failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.post("/v1/vision")
async def vision_endpoint(body: VisionRequest):
    """Accept an image plus prompt; route to a vision-capable Gemini model."""
    try:
        from super_browser.vision_api import decode_image_payload, vision_analyze

        image_bytes = decode_image_payload(
            image_base64=body.image_base64,
            image=body.image,
        )
        result = await asyncio.to_thread(
            vision_analyze,
            image_bytes=image_bytes,
            prompt=body.prompt.strip(),
            mime_type=body.mime_type or "image/png",
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            label="api:/v1/vision",
        )
        return {"status": "success", **result}
    except ValueError as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"[vision] endpoint failed: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.get("/health")
async def health():
    manifest = BASE_DIR / "corpus" / "MANIFEST.json"
    templates_ok = (_templates_dir / "index.html").is_file()
    runtime_error = _runtime_env_detail()
    return {
        "status": "degraded" if runtime_error else "ok",
        "templates": templates_ok,
        "corpus_manifest": manifest.is_file(),
        "agent_busy": _is_run_busy(),
        "index_busy": _is_index_busy(),
        "faiss_available": "faiss" not in (_missing_runtime_modules()),
        "python_executable": str(Path(sys.executable).resolve()),
        "runtime_error": runtime_error,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
