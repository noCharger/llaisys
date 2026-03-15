import argparse
import sys
import os
import time
import uuid
import asyncio
import json
import ssl
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import logging

from transformers import AutoTokenizer
try:
    from huggingface_hub import snapshot_download
except ImportError:
    pass
try:
    import llaisys
except ImportError:
    llaisys = None

from schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
)

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OBSERVABILITY_ENABLED = True
except ImportError:
    OBSERVABILITY_ENABLED = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerState:
    model: Any = None
    tokenizer: Any = None
    sessions: Dict[str, Any] = {}
    session_last_accessed: Dict[str, float] = {}
    
    model_path: str = ""
    device_name: str = "cpu"
    dtype: Any = getattr(llaisys, "DataType", type("Mock", (), {"F32": 0})).F32 if llaisys else 0
    max_sessions: int = 10 
    max_steps: int = 10
    
    # System Prompt Configurations
    default_system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
        "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
        "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
    )

state = ServerState()

from dataclasses import dataclass, field

def load_model() -> None:
    if llaisys is None:
        return
    logger.info(f"Loading model from {state.model_path} on {state.device_name} with {state.dtype}...")
    device_type = llaisys.DeviceType.NVIDIA if state.device_name == "nvidia" else llaisys.DeviceType.CPU
        
    state.model = llaisys.models.Qwen2(state.model_path, device_type, dtype=state.dtype)
    state.tokenizer = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)
    logger.info("Model loaded successfully.")

@dataclass
class SessionMeta:
    session_ptr: Any
    history_ids: List[int] = field(default_factory=list)
    last_accessed: float = 0.0

session_store: Dict[str, SessionMeta] = {}

def smart_session_match(input_ids: List[int]) -> tuple[str, SessionMeta, int]:
    best_id = None
    best_meta = None
    best_len = 0
    
    for sid, meta in session_store.items():
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
            
    if best_id:
        return best_id, best_meta, best_len
    
    return None, None, 0

def get_session_for_request(input_ids: List[int], session_id: Optional[str] = None) -> tuple[SessionMeta, int]:
    if session_id and session_id in session_store:
        meta = session_store[session_id]
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
            
        meta.last_accessed = time.time()
        return meta, match_len

    if not session_id:
        sid, meta, match_len = smart_session_match(input_ids)
        
        if meta and match_len > 0:
            if match_len < len(meta.history_ids):
                logger.info(f"Rewinding session {sid} from {len(meta.history_ids)} to {match_len}")
                state.model.rewind_session(meta.session_ptr, match_len)
                meta.history_ids = meta.history_ids[:match_len]
            
            meta.last_accessed = time.time()
            return meta, match_len
    
    if len(session_store) >= state.max_sessions:
        oldest_sid = min(session_store, key=lambda k: session_store[k].last_accessed)
        logger.info(f"Recycling session {oldest_sid}")
        meta = session_store[oldest_sid]
        
        if session_id:
            del session_store[oldest_sid]
            session_store[session_id] = meta
        
        state.model.rewind_session(meta.session_ptr, 0)
        meta.history_ids = []
        meta.last_accessed = time.time()
        return meta, 0
    
    new_sid = session_id if session_id else str(uuid.uuid4())
    logger.info(f"Creating new session {new_sid}")
    ptr = state.model.create_session()
    meta = SessionMeta(session_ptr=ptr, history_ids=[], last_accessed=time.time())
    session_store[new_sid] = meta
    return meta, 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Cleaning up sessions...")
    for meta in session_store.values():
        state.model.destroy_session(meta.session_ptr)

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "device": state.device_name}

@app.middleware("http")
async def secure_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    if ssl_configured:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    messages = [m.model_dump() for m in request.messages]
    
    has_system_prompt = len(messages) > 0 and messages[0].get("role") == "system"
    
    sys_prompt_content = request.system_prompt if request.system_prompt else state.default_system_prompt
    
    if not has_system_prompt and sys_prompt_content:
        logger.info(f"Injecting system prompt into session: {request.session_id}")
        messages.insert(0, {"role": "system", "content": sys_prompt_content})
    elif has_system_prompt and request.system_prompt:
        logger.info(f"Overriding system prompt in session: {request.session_id}")
        messages[0]["content"] = request.system_prompt

    prompt = state.tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    input_ids = state.tokenizer.encode(prompt)
    
    session_meta, match_len = get_session_for_request(input_ids, session_id=request.session_id)
    new_input_ids = input_ids[match_len:]
    session_meta.history_ids.extend(new_input_ids)
    
    generation_max_tokens = request.max_tokens
    if generation_max_tokens == 512 and state.max_steps != 512:
         generation_max_tokens = state.max_steps

    async def generate_stream():
        response_id = str(uuid.uuid4())
        created_time = int(time.time())
        model_name = request.model

        def create_chunk(content=None, role=None, finish_reason=None):
            delta = {}
            if role: delta["role"] = role
            if content: delta["content"] = content
            
            return {
                "data": json.dumps({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                })
            }
        
        yield create_chunk(role="assistant")
        
        current_tokens = list(new_input_ids)
        
        if len(current_tokens) > 0:
            logger.info(f"Calling model.forward with {len(current_tokens)} tokens")
            try:
                next_token = state.model.forward(
                    session_meta.session_ptr, 
                    current_tokens, 
                    request.temperature, 
                    request.top_p, 
                    request.top_k
                )
                logger.info(f"Model returned token: {next_token}")
            except Exception as e:
                logger.error(f"Error during model.forward: {e}")
                raise e
            session_meta.history_ids.append(next_token)
            
            text = state.tokenizer.decode([next_token], skip_special_tokens=True)
            if text:
                yield create_chunk(content=text)
                
            if next_token == state.model.end_token:
                yield create_chunk(finish_reason="stop")
                yield {"data": "[DONE]"}
                return

        for _ in range(generation_max_tokens - 1):
            last_token = session_meta.history_ids[-1]
            
            next_token = state.model.forward(
                session_meta.session_ptr,
                [last_token],
                request.temperature,
                request.top_p,
                request.top_k
            )
            
            session_meta.history_ids.append(next_token)
            text = state.tokenizer.decode([next_token], skip_special_tokens=True)
            
            if text:
                yield create_chunk(content=text)
            
            if next_token == state.model.end_token:
                yield create_chunk(finish_reason="stop")
                yield {"data": "[DONE]"}
                return
                
        yield create_chunk(finish_reason="length")
        yield {"data": "[DONE]"}

    if request.stream:
        return EventSourceResponse(generate_stream())
    else:
        full_content = ""
        async for chunk in generate_stream():
            if chunk["data"] == "[DONE]":
                break
            
            data = json.loads(chunk["data"])
            delta = data["choices"][0]["delta"]
            if "content" in delta:
                full_content += delta["content"]
        
        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=full_content),
                finish_reason="stop"
            )]
        )

http_app = FastAPI()

@http_app.middleware("http")
async def redirect_to_https(request: Request, call_next):
    url = request.url.replace(scheme="https", port=request.app.state.https_port)
    return RedirectResponse(url, status_code=301)


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
    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS and run in HTTP mode only")
    args = parser.parse_args()

    config = {}
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = json.load(f)
        else:
            logger.error(f"Config file {args.config} not found")
            sys.exit(1)

    def get_val(cli_val, config_key, default_val):
        if cli_val is not None:
            return cli_val
        if config_key in config:
            return config[config_key]
        return default_val

    model_path = get_val(args.model, "model", None)
    device = get_val(args.device, "device", "cpu")
    dtype_str = get_val(args.dtype, "dtype", "float32")
    host = get_val(args.host, "host", "127.0.0.1")
    
    port = get_val(args.port, "port", 6008)
    https_port = port + 1
    
    max_sessions = get_val(args.max_sessions, "max_sessions", 10)
    max_steps = get_val(args.max_steps, "max_steps", 10)
    
    # Load optional custom system prompt from config
    if "system_prompt" in config:
        state.default_system_prompt = config["system_prompt"]

    if not model_path:
        logger.error("Model path is required (via --model or config file)")
        sys.exit(1)

    if not os.path.exists(model_path) and not os.path.isdir(model_path):
        logger.info(f"Model path {model_path} not found, assuming HF Hub ID...")
        try:
            model_path = snapshot_download(model_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)

    state.model_path = model_path
    state.device_name = device
    state.max_sessions = max_sessions
    state.max_steps = max_steps
    state.dtype = llaisys.DataType.F16 if dtype_str == "float16" else llaisys.DataType.F32

    cert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'certs'))
    key_path = os.path.join(cert_dir, 'key.pem')
    cert_path = os.path.join(cert_dir, 'cert.pem')
    
    ssl_configured = False
    
    if not args.no_https:
        if not os.path.exists(key_path) or not os.path.exists(cert_path):
            os.makedirs(cert_dir, exist_ok=True)
            res = os.system(f'openssl req -x509 -newkey rsa:4096 -keyout {key_path} -out {cert_path} -days 365 -nodes -subj "/CN=localhost" 2>/dev/null')
            ssl_configured = (res == 0)
            if not ssl_configured:
                logger.error("Failed to generate certificates. Falling back to HTTP.")
        else:
            ssl_configured = True
    else:
        logger.info("HTTPS disabled via --no-https flag.")

    if ssl_configured:
        async def run_servers():
            http_app.state.https_port = https_port
            
            config_https = uvicorn.Config(
                app, host=host, port=https_port, 
                ssl_keyfile=key_path, ssl_certfile=cert_path, 
                ssl_version=ssl.PROTOCOL_TLS_SERVER
            )
            
            config_http = uvicorn.Config(http_app, host=host, port=port)
            
            logger.info(f"Starting HTTPS server on port {https_port} (TLS 1.3)")
            logger.info(f"Starting HTTP redirect server on port {port}")
            
            await asyncio.gather(
                uvicorn.Server(config_https).serve(), 
                uvicorn.Server(config_http).serve()
            )
            
        asyncio.run(run_servers())
    else:
        logger.warning("Running in HTTP mode.")
        uvicorn.run(app, host=host, port=port)
