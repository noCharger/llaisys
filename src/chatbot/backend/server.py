import argparse
import sys
import os
import time
import uuid
import asyncio
import json
import ssl
import logging
import threading
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Integration of Control Plane components
from app.routers import admin
from app.dependencies import tenant_manager, rate_limiter
from app.middleware.auth import AuthMiddleware

from transformers import AutoTokenizer

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

try:
    import llaisys
except ImportError:
    llaisys = None

from app.models.chat import (
    Message as ChatMessage,
    ChatRequest as ChatCompletionRequest,
    ChatResponse as ChatCompletionResponse
)

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

class ContextItemRequest(BaseModel):
    id: str
    content: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
from app.services.context_manager import ContextManager

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OBSERVABILITY_ENABLED = True
except ImportError:
    OBSERVABILITY_ENABLED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llaisys.server")

class ServerState:
    """Manages the global state of the inference server."""
    def __init__(self):
        self.model: Any = None
        self.tokenizer: Any = None
        self.sessions: Dict[str, Any] = {}
        self.model_path: str = ""
        self.device_name: str = "cpu"
        self.dtype: Any = 0
        self.max_sessions: int = 10
        self.max_steps: int = 4096
        self.default_system_prompt: str = (
            "You are a helpful AI assistant.\n"
            "{% if context_items %}\n"
            "Context:\n"
            "{% for item in context_items %}\n"
            "- {{ item.content }}\n"
            "{% endfor %}\n"
            "Use the context above if relevant.\n"
            "{% endif %}\n"
            "Answer the user clearly. Adapt the length and detail of your response to the nature of the user's query: "
            "provide concise answers for simple questions, and detailed, step-by-step explanations for complex tasks."
        )
        self.max_context_len: int = 8192
        self.https_port: int = 0
        self.temperature: float = 0.7
        self.top_p: float = 0.9
        self.top_k: int = 50
        self.context_manager = ContextManager()

state = ServerState()

@dataclass
class SessionMeta:
    """Metadata for an active inference session."""
    session_ptr: Any
    history_ids: List[int] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)
    in_use: bool = False

session_store: Dict[str, SessionMeta] = {}
store_lock = threading.Lock()
def load_model() -> None:
    """Loads the model and tokenizer into global state."""
    if llaisys is None:
        logger.warning("llaisys module not found. Inference will not work.")
        return

    try:
        logger.info(f"Loading model from {state.model_path} on {state.device_name}...")

        if hasattr(llaisys, "DataType"):
             state.dtype = llaisys.DataType.F16 if state.dtype == "float16" else llaisys.DataType.F32

        device_type = llaisys.DeviceType.NVIDIA if state.device_name == "nvidia" else llaisys.DeviceType.CPU

        state.model = llaisys.models.Qwen2(state.model_path, device_type, dtype=state.dtype)
        state.tokenizer = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)

        if hasattr(state.model, "meta") and hasattr(state.model.meta, "maxseq"):
            state.max_context_len = state.model.meta.maxseq
            logger.info(f"Max context length set to {state.max_context_len}")

        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

def cleanup_sessions():
    """Cleans up all active sessions."""
    if not state.model:
        return
    logger.info("Cleaning up sessions...")
    with store_lock:
        for meta in session_store.values():
            try:
                state.model.destroy_session(meta.session_ptr)
            except Exception as e:
                logger.error(f"Error destroying session: {e}")
        session_store.clear()

def get_session_for_request(input_ids: List[int], session_id: Optional[str] = None) -> tuple[SessionMeta, int]:
    """
    Retrieves or creates a session for the request.
    Returns (SessionMeta, match_length_in_tokens).
    Thread-safe implementation with store_lock.
    """
    current_time = time.time()

    with store_lock:
        if session_id and session_id in session_store:
            meta = session_store[session_id]
            if meta.in_use:
                logger.warning(f"Requested session {session_id} is currently in use.")
                raise HTTPException(status_code=409, detail=f"Session {session_id} is busy processing another request.")

            meta.last_accessed = current_time
            meta.in_use = True

            match_len = 0
            min_len = min(len(meta.history_ids), len(input_ids))
            for i in range(min_len):
                if meta.history_ids[i] == input_ids[i]:
                    match_len += 1
                else:
                    break

            if match_len < len(meta.history_ids):
                logger.info(f"Rewinding session {session_id} from {len(meta.history_ids)} to {match_len}")
                state.model.rewind_session(meta.session_ptr, match_len)
                meta.history_ids = meta.history_ids[:match_len]

            return meta, match_len

        best_id = None
        best_meta = None
        best_len = 0

        for sid, meta in session_store.items():
            if meta.in_use:
                continue

            match_len = 0
            min_len = min(len(meta.history_ids), len(input_ids))
            for i in range(min_len):
                if meta.history_ids[i] == input_ids[i]:
                    match_len += 1
                else:
                    break

            if match_len > best_len:
                best_len = match_len
                best_id = sid
                best_meta = meta

        if best_meta is not None:
            if not session_id:
                best_meta.last_accessed = current_time
                best_meta.in_use = True

                if best_len < len(best_meta.history_ids):
                    state.model.rewind_session(best_meta.session_ptr, best_len)
                    best_meta.history_ids = best_meta.history_ids[:best_len]
                return best_meta, best_len

        if len(session_store) >= state.max_sessions:

            available_sessions = [s for s in session_store.items() if not s[1].in_use]

            if not available_sessions:
                raise HTTPException(status_code=503, detail="All sessions are busy and max capacity reached.")

            oldest_sid, _ = min(available_sessions, key=lambda item: item[1].last_accessed)

            logger.info(f"Evicting session {oldest_sid}")
            meta = session_store.pop(oldest_sid)

            state.model.rewind_session(meta.session_ptr, 0)
            meta.history_ids = []
            meta.messages = []
            meta.last_accessed = current_time
            meta.in_use = True

            new_sid = session_id if session_id else str(uuid.uuid4())
            session_store[new_sid] = meta
            return meta, 0

        new_sid = session_id if session_id else str(uuid.uuid4())
        logger.info(f"Creating new session {new_sid}")
        ptr = state.model.create_session()
        meta = SessionMeta(session_ptr=ptr, history_ids=[], messages=[], last_accessed=current_time, in_use=True)
        session_store[new_sid] = meta
        return meta, 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inject tenant_manager for dependency injection in routers
    app.state.tenant_manager = tenant_manager
    
    load_model()
    yield
    cleanup_sessions()

app = FastAPI(title="Llaisys Chatbot API", lifespan=lifespan)

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Integrate Control Plane (Tenant/Auth) features
app.add_middleware(AuthMiddleware, tenant_manager=tenant_manager, rate_limiter=rate_limiter)
app.include_router(admin.router, prefix="/v1")

if OBSERVABILITY_ENABLED:
    REQUEST_COUNT = Counter("llaisys_requests_total", "Total requests", ["method", "endpoint", "status"])
    REQUEST_LATENCY = Histogram("llaisys_request_latency_seconds", "Request latency", ["method", "endpoint"])

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)

        return response

    @app.get("/metrics")
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)

@app.get("/health")
def health_check():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "device": state.device_name, "model": state.model_path}

@app.get("/config")
def get_config(request: Request):
    """Returns public configuration for the frontend."""

    scheme = request.url.scheme
    host = request.url.hostname
    port = request.url.port

    api_url = f"{scheme}://{host}:{port}"
    return {"apiUrl": api_url}

@app.post("/v1/context/items")
def add_context_item(item: ContextItemRequest):
    """Adds or updates a context item."""
    state.context_manager.add_item(item.id, item.content, item.tags, item.metadata)
    return {"status": "success", "id": item.id}

@app.get("/v1/context/items")
def list_context_items(tag: Optional[str] = None):
    """Lists context items, optionally filtered by tag."""
    if tag:
        items = state.context_manager.get_items_by_tag(tag)
    else:
        items = state.context_manager.list_items()
    return items

@app.delete("/v1/context/items/{item_id}")
def delete_context_item(item_id: str):
    """Deletes a context item."""
    if state.context_manager.remove_item(item_id):
        return {"status": "success", "id": item_id}
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/v1/context/render")
def render_template_endpoint(template: str, extra_context: Optional[Dict[str, Any]] = None):
    """Renders a template string using the current context."""
    try:
        return {"rendered": state.context_manager.render_template(template, extra_context)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def prune_messages(messages: List[Dict[str, str]], tokenizer: Any, max_len: int) -> List[Dict[str, str]]:
    """Prunes messages to fit within context length."""
    if not messages:
        return messages

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_tokens = tokenizer.encode(full_text)

    if len(full_tokens) <= max_len:
        return messages

    logger.info(f"Context length {len(full_tokens)} exceeds limit {max_len}. Pruning...")

    pruned_msgs = list(messages)
    has_system = pruned_msgs[0]['role'] == 'system'

    while len(full_tokens) > max_len and len(pruned_msgs) > (1 if has_system else 0):

        idx_to_remove = 1 if has_system else 0
        if idx_to_remove >= len(pruned_msgs):
            break

        pruned_msgs.pop(idx_to_remove)
        full_text = tokenizer.apply_chat_template(pruned_msgs, tokenize=False, add_generation_prompt=True)
        full_tokens = tokenizer.encode(full_text)

    logger.info(f"Pruned context to {len(full_tokens)} tokens")
    return pruned_msgs

async def generate_response_stream(
    request: ChatCompletionRequest,
    session_meta: SessionMeta,
    new_input_ids: List[int],
    max_gen: int
) -> AsyncGenerator[str, None]:
    """Generator for streaming responses."""

    response_id = str(uuid.uuid4())
    created_time = int(time.time())
    model_name = request.model

    def create_chunk(content: Optional[str] = None, role: Optional[str] = None, finish_reason: Optional[str] = None):
        delta = ChatCompletionChunkDelta()
        if role: delta.role = role
        if content: delta.content = content

        chunk = ChatCompletionChunk(
            id=response_id,
            created=created_time,
            model=model_name,
            choices=[ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason
            )]
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    try:
        yield create_chunk(role="assistant")

        current_tokens = list(new_input_ids)
        full_content = ""

        if len(current_tokens) > 0:
            try:
                next_token = state.model.forward(
                    session_meta.session_ptr,
                    current_tokens,
                    request.temperature or state.temperature,
                    request.top_p or state.top_p,
                    request.top_k or state.top_k
                )
            except Exception as e:
                logger.error(f"Error during model forward: {e}")
                yield create_chunk(finish_reason="error")
                return

            session_meta.history_ids.append(next_token)

            if next_token == getattr(state.model, "end_token", 2):
                session_meta.messages.append({"role": "assistant", "content": full_content})
                yield create_chunk(finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            text = state.tokenizer.decode([next_token], skip_special_tokens=False)

            if text:
                full_content += text
                yield create_chunk(content=text)

        for _ in range(max_gen - 1):
            last_token = session_meta.history_ids[-1]

            try:
                next_token = state.model.forward(
                    session_meta.session_ptr,
                    [last_token],
                    request.temperature or state.temperature,
                    request.top_p or state.top_p,
                    request.top_k or state.top_k
                )
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                break

            session_meta.history_ids.append(next_token)

            if next_token == getattr(state.model, "end_token", 2):
                session_meta.messages.append({"role": "assistant", "content": full_content})
                yield create_chunk(finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            text = state.tokenizer.decode([next_token], skip_special_tokens=False)

            if text:
                full_content += text
                yield create_chunk(content=text)

        session_meta.messages.append({"role": "assistant", "content": full_content})
        yield create_chunk(finish_reason="length")
        yield "data: [DONE]\n\n"
    finally:
        session_meta.in_use = False
        logger.debug(f"Session released. in_use={session_meta.in_use}")

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest):
    if not state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:

        req_messages = [m.model_dump() for m in request.messages]

        max_gen = request.max_tokens or 4096
        if max_gen > state.max_steps:
             max_gen = state.max_steps

        max_input = state.max_context_len - max_gen
        messages = prune_messages(req_messages, state.tokenizer, max_input)

        has_system_prompt = len(messages) > 0 and messages[0].get("role") == "system"
        sys_prompt_content = request.system_prompt if request.system_prompt else state.default_system_prompt

        if request.use_template and sys_prompt_content:
            try:
                logger.info("Rendering system prompt template...")
                sys_prompt_content = state.context_manager.render_template(sys_prompt_content)
            except Exception as e:
                logger.error(f"Failed to render system prompt template: {e}")
                raise HTTPException(status_code=400, detail=f"Template error: {str(e)}")

        if not has_system_prompt and sys_prompt_content:
            messages.insert(0, {"role": "system", "content": sys_prompt_content})
        elif has_system_prompt and request.system_prompt:
            messages[0]["content"] = request.system_prompt

        prompt = state.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False
        )
        input_ids = state.tokenizer.encode(prompt)

        session_meta, match_len = get_session_for_request(input_ids, session_id=request.session_id)

        try:
            session_meta.messages = messages

            new_input_ids = input_ids[match_len:]
            session_meta.history_ids.extend(new_input_ids)

            stream_gen = generate_response_stream(request, session_meta, new_input_ids, max_gen)

            if request.stream:
                return StreamingResponse(stream_gen, media_type="text/event-stream")
            else:

                full_content = ""
                finish_reason = "length"

                async for chunk_str in stream_gen:
                    if chunk_str.strip() == "data: [DONE]":
                        break

                    if chunk_str.startswith("data: "):
                        json_str = chunk_str[6:].strip()
                        try:
                            chunk_data = json.loads(json_str)
                            delta = chunk_data["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                full_content += delta["content"]
                            if chunk_data["choices"][0]["finish_reason"]:
                                finish_reason = chunk_data["choices"][0]["finish_reason"]
                        except:
                            pass

                return ChatCompletionResponse(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    model=request.model,
                    choices=[ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=full_content),
                        finish_reason=finish_reason
                    )]
                )
        except Exception:
            session_meta.in_use = False
            raise

    except Exception as e:
        logger.error(f"Unexpected error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

http_app = FastAPI()

@http_app.middleware("http")
async def redirect_to_https(request: Request, call_next):
    if state.https_port:
        url = request.url.replace(scheme="https", port=state.https_port)
        return RedirectResponse(url, status_code=301)
    return await call_next(request)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file (JSON)")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--device", type=str, choices=["cpu", "nvidia"])
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], help="Model data type")
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-steps", type=int, help="Max generation steps")
    parser.add_argument("--temperature", type=float, help="Default temperature")
    parser.add_argument("--top-p", type=float, help="Default top_p")
    parser.add_argument("--top-k", type=int, help="Default top_k")
    parser.add_argument("--system-prompt", type=str, help="Default system prompt")
    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS")
    args = parser.parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)

    def get_val(cli_val, config_key, default_val):
        return cli_val if cli_val is not None else config.get(config_key, default_val)

    state.model_path = get_val(args.model, "model", "")
    state.device_name = get_val(args.device, "device", "cpu")
    state.dtype = get_val(args.dtype, "dtype", "float32")
    host = get_val(args.host, "host", "127.0.0.1")
    port = get_val(args.port, "port", 6008)
    state.max_sessions = get_val(args.max_sessions, "max_sessions", 10)
    state.max_steps = get_val(args.max_steps, "max_steps", 4096)
    state.temperature = get_val(args.temperature, "temperature", 0.7)
    state.top_p = get_val(args.top_p, "top_p", 0.9)
    state.top_k = get_val(args.top_k, "top_k", 50)
    state.default_system_prompt = get_val(args.system_prompt, "system_prompt", state.default_system_prompt)

    if not state.model_path:
        logger.error("Model path is required via --model or config file")
        sys.exit(1)

    if not os.path.exists(state.model_path) and not os.path.isdir(state.model_path) and snapshot_download:
        logger.info(f"Model path {state.model_path} not found locally, attempting HF download...")
        try:
            state.model_path = snapshot_download(state.model_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)

    cert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'certs'))
    key_path = os.path.join(cert_dir, 'key.pem')
    cert_path = os.path.join(cert_dir, 'cert.pem')

    ssl_configured = False
    if not args.no_https:
        if not (os.path.exists(key_path) and os.path.exists(cert_path)):
            os.makedirs(cert_dir, exist_ok=True)
            logger.info("Generating self-signed certificate...")

            cmd = f'openssl req -x509 -newkey rsa:4096 -keyout "{key_path}" -out "{cert_path}" -days 365 -nodes -subj "/CN=localhost" 2>/dev/null'
            if os.system(cmd) == 0:
                ssl_configured = True
            else:
                logger.error("Failed to generate certificates. Fallback to HTTP.")
        else:
            ssl_configured = True

    if ssl_configured:
        https_port = port + 1
        state.https_port = https_port

        async def run_servers():
            config_https = uvicorn.Config(
                app, host=host, port=https_port,
                ssl_keyfile=key_path, ssl_certfile=cert_path,
                log_level="info"
            )
            config_http = uvicorn.Config(http_app, host=host, port=port, log_level="warning")

            logger.info(f"Starting HTTPS server on {host}:{https_port}")
            logger.info(f"Starting HTTP redirect server on {host}:{port}")

            server_https = uvicorn.Server(config_https)
            server_http = uvicorn.Server(config_http)

            await asyncio.gather(server_https.serve(), server_http.serve())

        asyncio.run(run_servers())
    else:
        logger.warning("Running in HTTP mode (no SSL).")
        uvicorn.run(app, host=host, port=port)