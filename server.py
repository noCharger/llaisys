import argparse
import sys
import os
import time
import uuid
import asyncio
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import uvicorn
import logging

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
    print("Warning: Observability packages not found. Install prometheus_client and opentelemetry-sdk.")

# Ensure stdout uses utf-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import llaisys

# --- Global State ---
class ServerState:
    model: Optional[llaisys.models.Qwen2] = None
    tokenizer: Optional[AutoTokenizer] = None
    sessions: Dict[str, Any] = {} # Map session_id -> llaisys session pointer
    session_last_accessed: Dict[str, float] = {} # For LRU
    
    # Configuration
    model_path: str = ""
    device_name: str = "cpu"
    max_sessions: int = 10 # Max concurrent active sessions

state = ServerState()

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    session_id: Optional[str] = None # Custom field for explicit session management

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Dict[str, int]] = None

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

# --- Helper Functions ---

def load_model():
    print(f"Loading model from {state.model_path} on {state.device_name}...")
    device_type = llaisys.DeviceType.CPU
    if state.device_name == "nvidia":
        device_type = llaisys.DeviceType.NVIDIA
        
    state.model = llaisys.models.Qwen2(state.model_path, device_type)
    state.tokenizer = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)
    print("Model loaded successfully.")

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Any]:
    current_time = time.time()
    
    # 1. If explicit session_id provided
    if session_id:
        if session_id in state.sessions:
            state.session_last_accessed[session_id] = current_time
            return session_id, state.sessions[session_id]
        else:
            # Create new if explicitly requested but missing?
            # Or treat as new.
            pass
    else:
        session_id = str(uuid.uuid4())

    # 2. Check limits and evict if needed
    if len(state.sessions) >= state.max_sessions:
        # Evict LRU
        oldest_session = min(state.session_last_accessed, key=state.session_last_accessed.get)
        print(f"Evicting session {oldest_session}")
        state.model.destroy_session(state.sessions[oldest_session])
        del state.sessions[oldest_session]
        del state.session_last_accessed[oldest_session]

    # 3. Create new session
    print(f"Creating new session {session_id}")
    session = state.model.create_session()
    state.sessions[session_id] = session
    state.session_last_accessed[session_id] = current_time
    
    return session_id, session

def find_common_prefix(session, input_ids: List[int]) -> int:
    # This is tricky because we don't store the token history in python explicitly in a way that maps 1:1 to cache.
    # We only know the current pos of the session.
    # Ideally, we should store the token history associated with the session in Python.
    # For now, we assume simple append-only or full reset if history mismatch.
    # To support "Prefix Matching" properly, we need to store `history_tokens` in `state.sessions_meta`.
    
    # Let's improve session management to store history.
    return 0

# Improved Session Management with History
class SessionMeta:
    session_ptr: Any
    history_ids: List[int]
    last_accessed: float

session_store: Dict[str, SessionMeta] = {}

def smart_session_match(input_ids: List[int]) -> tuple[str, SessionMeta, int]:
    """
    Finds a session that has the longest common prefix with input_ids.
    Returns (session_id, session_meta, match_length).
    """
    best_id = None
    best_meta = None
    best_len = 0
    
    for sid, meta in session_store.items():
        # Check prefix match
        match_len = 0
        min_len = min(len(meta.history_ids), len(input_ids))
        
        # Optimization: verify if it's worth checking (e.g. if we want at least some match)
        # Python list comparison is fast enough for typical context lengths
        for i in range(min_len):
            if meta.history_ids[i] == input_ids[i]:
                match_len += 1
            else:
                break
        
        if match_len > best_len:
            best_len = match_len
            best_id = sid
            best_meta = meta
            
    # Heuristic: if match length is small (e.g. < 10), maybe just pick a free one or LRU to overwrite?
    # For now, if we found *any* match, reuse it?
    # Ideally we reuse the one with *longest* match.
    
    if best_id:
        return best_id, best_meta, best_len
    
    return None, None, 0

def get_session_for_request(input_ids: List[int]) -> tuple[SessionMeta, int]:
    # 1. Try to find a session with prefix match
    sid, meta, match_len = smart_session_match(input_ids)
    
    if meta and match_len > 0:
        # Found a reusable session
        # If match_len < len(meta.history_ids), we need to rewind!
        if match_len < len(meta.history_ids):
            print(f"Rewinding session {sid} from {len(meta.history_ids)} to {match_len}")
            state.model.rewind_session(meta.session_ptr, match_len)
            meta.history_ids = meta.history_ids[:match_len]
        
        meta.last_accessed = time.time()
        return meta, match_len
    
    # 2. No suitable match, create new or recycle LRU
    if len(session_store) >= state.max_sessions:
        # Evict LRU
        oldest_sid = min(session_store, key=lambda k: session_store[k].last_accessed)
        print(f"Recycling session {oldest_sid}")
        meta = session_store[oldest_sid]
        # Reset it
        state.model.rewind_session(meta.session_ptr, 0)
        meta.history_ids = []
        meta.last_accessed = time.time()
        return meta, 0
    
    # 3. Create new
    new_sid = str(uuid.uuid4())
    print(f"Creating new session {new_sid}")
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
    load_model()
    yield
    # Cleanup
    print("Cleaning up sessions...")
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
    # Prometheus Metrics
    REQUEST_COUNT = Counter("llaisys_requests_total", "Total requests", ["method", "endpoint", "status"])
    REQUEST_LATENCY = Histogram("llaisys_request_latency_seconds", "Request latency", ["method", "endpoint"])
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        method = request.method
        path = request.url.path
        start_time = time.time()
        
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
        return Response(
            content=generate_latest(), 
            media_type=CONTENT_TYPE_LATEST
        )

    # OpenTelemetry Tracing
    # In production, use OTLP exporter instead of Console
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    FastAPIInstrumentor.instrument_app(app)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not state.model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Prepare Prompt
    # Convert messages to prompt string using tokenizer template
    prompt = state.tokenizer.apply_chat_template(
        conversation=[m.model_dump() for m in request.messages],
        add_generation_prompt=True,
        tokenize=False
    )
    
    input_ids = state.tokenizer.encode(prompt)
    
    # 2. Session Management (Prefix Matching)
    session_meta, match_len = get_session_for_request(input_ids)
    
    # New tokens to process (suffix)
    new_input_ids = input_ids[match_len:]
    
    # Update session history (we append input_ids now, output_ids later)
    session_meta.history_ids.extend(new_input_ids)
    
    # 3. Generation Logic
    # We need a generator that yields tokens for streaming
    
    async def generate_stream():
        response_id = str(uuid.uuid4())
        created_time = int(time.time())
        
        # Yield role
        yield {
            "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        
        # We need to run inference in a loop.
        # Since llaisys inference is blocking C++ call, we should run it in a thread pool ideally,
        # but Python GIL might block. However, if C++ releases GIL, it's fine.
        # Assuming we run sequentially for now (single worker).
        
        current_tokens = list(new_input_ids)
        generated_tokens = []
        
        # Prefill if needed
        if len(current_tokens) > 0:
            # For prefill, we pass all tokens. The C++ infer will return the next token.
            # But wait, Qwen2 wrapper's `generate` handles the loop.
            # We should probably expose a `generate_step` or similar to control the loop here for streaming.
            # Or modify `generate` to be a generator.
            pass
            
        # To implement streaming without modifying Qwen2 class too much,
        # we can call the underlying `llaisysQwen2ModelInfer` directly or add a `generate_iterator` to Qwen2.
        # Let's assume we use the low-level API via `state.model` wrapper helpers if possible,
        # OR we just implement the loop here using `ctypes`.
        
        # Let's use `state.model` helpers but implement the loop here.
        # We need access to `model` and `session_ptr`.
        
        import ctypes
        
        # Prefill
        if len(current_tokens) > 0:
            next_token = state.model.forward(
                session_meta.session_ptr, 
                current_tokens, 
                request.temperature, 
                request.top_p, 
                request.top_k
            )
            generated_tokens.append(next_token)
            session_meta.history_ids.append(next_token)
            
            # Yield first token
            text = state.tokenizer.decode([next_token], skip_special_tokens=True)
            if text:
                yield {
                    "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
                
            if next_token == state.model.end_token:
                yield {
                    "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                return

        # Decode Loop
        for _ in range(request.max_tokens - 1):
            last_token = session_meta.history_ids[-1]
            
            next_token = state.model.forward(
                session_meta.session_ptr,
                [last_token],
                request.temperature,
                request.top_p,
                request.top_k
            )
            
            generated_tokens.append(next_token)
            session_meta.history_ids.append(next_token)
            
            # Decode logic (simple) - for better quality we should decode accumulated buffer to handle unicode split
            # For simplicity: decode single token.
            text = state.tokenizer.decode([next_token], skip_special_tokens=True)
            
            # Handle unicode issues? Huggingface tokenizer usually handles this if we use `decode(..., stream=True)` but we don't have the streamer here easily.
            # A simple workaround: decode all generated tokens, remove printed prefix.
            # Optimized: decode last N tokens.
            
            if text:
                yield {
                    "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
            
            if next_token == state.model.end_token:
                yield {
                    "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                return
                
        yield {
            "id": response_id, "object": "chat.completion.chunk", "created": created_time, "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
        }

    if request.stream:
        return EventSourceResponse(generate_stream())
    else:
        # Non-streaming
        full_content = ""
        async for chunk in generate_stream():
            delta = chunk["choices"][0]["delta"]
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
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-sessions", type=int, default=10)
    args = parser.parse_args()

    # If model path is HF ID, download it
    if not os.path.exists(args.model) and not os.path.isdir(args.model):
        print(f"Model path {args.model} not found, assuming HF Hub ID...")
        try:
            args.model = snapshot_download(args.model)
        except Exception as e:
            print(f"Failed to download model: {e}")
            sys.exit(1)

    state.model_path = args.model
    state.device_name = args.device
    state.max_sessions = args.max_sessions

    uvicorn.run(app, host=args.host, port=args.port)
