import argparse
import sys
import os
import time
import uuid
import asyncio
import json
import io
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import logging

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import llaisys

from schemas import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
)
import server_pytorch

# Observability imports
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OBSERVABILITY_ENABLED = True
except ImportError:
    OBSERVABILITY_ENABLED = False

# Ensure stdout uses utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
class ServerState:
    """Manages global state for the llaisys backend."""
    model: Optional[llaisys.models.Qwen2] = None
    tokenizer: Optional[AutoTokenizer] = None
    sessions: Dict[str, Any] = {} # Map session_id -> llaisys session pointer
    session_last_accessed: Dict[str, float] = {} # For LRU
    
    # Configuration
    model_path: str = ""
    device_name: str = "cpu"
    max_sessions: int = 10 
    use_pytorch: bool = False 
    max_steps: int = 10 

state = ServerState()

# --- Helper Functions ---

def load_model() -> None:
    """Loads the llaisys model and tokenizer."""
    logger.info(f"Loading model from {state.model_path} on {state.device_name}...")
    device_type = llaisys.DeviceType.CPU
    if state.device_name == "nvidia":
        device_type = llaisys.DeviceType.NVIDIA
        
    state.model = llaisys.models.Qwen2(state.model_path, device_type)
    state.tokenizer = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)
    logger.info("Model loaded successfully.")

# Improved Session Management with History
class SessionMeta:
    session_ptr: Any
    history_ids: List[int]
    last_accessed: float

session_store: Dict[str, SessionMeta] = {}

def smart_session_match(input_ids: List[int]) -> tuple[str, SessionMeta, int]:
    """Finds a session that has the longest common prefix with input_ids."""
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

def get_session_for_request(input_ids: List[int]) -> tuple[SessionMeta, int]:
    """Retrieves or creates a session based on input prefix matching."""
    # 1. Find a session with prefix match
    sid, meta, match_len = smart_session_match(input_ids)
    
    if meta and match_len > 0:
        if match_len < len(meta.history_ids):
            logger.info(f"Rewinding session {sid} from {len(meta.history_ids)} to {match_len}")
            state.model.rewind_session(meta.session_ptr, match_len)
            meta.history_ids = meta.history_ids[:match_len]
        
        meta.last_accessed = time.time()
        return meta, match_len
    
    # 2. No suitable match, create new or recycle LRU
    if len(session_store) >= state.max_sessions:
        oldest_sid = min(session_store, key=lambda k: session_store[k].last_accessed)
        logger.info(f"Recycling session {oldest_sid}")
        meta = session_store[oldest_sid]
        state.model.rewind_session(meta.session_ptr, 0)
        meta.history_ids = []
        meta.last_accessed = time.time()
        return meta, 0
    
    # 3. Create new session
    new_sid = str(uuid.uuid4())
    logger.info(f"Creating new session {new_sid}")
    ptr = state.model.create_session()
    meta = SessionMeta()
    meta.session_ptr = ptr
    meta.history_ids = []
    meta.last_accessed = time.time()
    session_store[new_sid] = meta
    return meta, 0


# --- API Endpoints ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.use_pytorch:
        server_pytorch.load_model(state.model_path, state.device_name, state.max_steps)
    else:
        load_model()
    yield
    # Cleanup
    if not state.use_pytorch:
        logger.info("Cleaning up sessions...")
        for meta in session_store.values():
            state.model.destroy_session(meta.session_ptr)

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Observability Setup
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
    if state.use_pytorch:
        return await server_pytorch.chat_completions(request)

    if not state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Prepare Prompt
    prompt = state.tokenizer.apply_chat_template(
        conversation=[m.model_dump() for m in request.messages],
        add_generation_prompt=True,
        tokenize=False
    )
    
    input_ids = state.tokenizer.encode(prompt)
    
    # 2. Session Management
    session_meta, match_len = get_session_for_request(input_ids)
    new_input_ids = input_ids[match_len:]
    session_meta.history_ids.extend(new_input_ids)
    
    # 3. Generation Logic
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
        
        # Prefill
        if len(current_tokens) > 0:
            next_token = state.model.forward(
                session_meta.session_ptr, 
                current_tokens, 
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

        # Decode Loop
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "nvidia"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=10)
    parser.add_argument("--pytorch-model", action="store_true", help="Use PyTorch backend")
    parser.add_argument("--max-steps", type=int, default=10, help="Max generation steps")
    args = parser.parse_args()

    # Set default port
    if args.port is None:
        args.port = 8002 if args.pytorch_model else 8000

    # Download model if needed
    if not os.path.exists(args.model) and not os.path.isdir(args.model):
        logger.info(f"Model path {args.model} not found, assuming HF Hub ID...")
        try:
            args.model = snapshot_download(args.model)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)

    state.model_path = args.model
    state.device_name = args.device
    state.max_sessions = args.max_sessions
    state.use_pytorch = args.pytorch_model
    state.max_steps = args.max_steps

    uvicorn.run(app, host=args.host, port=args.port)
