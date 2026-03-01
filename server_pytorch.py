import logging
import time
import uuid
import asyncio
import json
import torch
from threading import Thread
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sse_starlette.sse import EventSourceResponse

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyTorchServerState:
    """Manages the global state for the PyTorch backend."""
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    device: str = "cpu"
    model_path: str = ""
    max_steps: int = 512

state = PyTorchServerState()

def load_model(model_path: str, device: str, max_steps: int = 512) -> None:
    """Loads the PyTorch model and tokenizer."""
    logger.info(f"Loading PyTorch model from {model_path} on {device}...")
    state.model_path = model_path
    state.device = device
    state.max_steps = max_steps
    
    try:
        state.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        torch_dtype = torch.float32
        if device == "cuda":
            torch_dtype = torch.bfloat16
        elif device == "mps":
             torch_dtype = torch.float16

        state.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        logger.info("PyTorch model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        raise e

async def chat_completions(request: ChatCompletionRequest):
    """Handles chat completion requests."""
    if not state.model:
        raise HTTPException(status_code=500, detail="PyTorch Model not loaded")

    start_time = time.time()
    logger.info(f"Received request for model {request.model}")

    try:
        prompt = state.tokenizer.apply_chat_template(
            conversation=[m.model_dump() for m in request.messages],
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = state.tokenizer(prompt, return_tensors="pt").to(state.device)

        if request.stream:
            return await _handle_streaming(request, inputs, start_time)
        else:
            return await _handle_non_streaming(request, inputs, start_time)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_generation_config(request: ChatCompletionRequest) -> dict:
    """Constructs generation configuration from request."""
    # Override default max_tokens if configured globally and request uses default
    max_new_tokens = request.max_tokens
    if max_new_tokens == 512 and state.max_steps != 512:
        max_new_tokens = state.max_steps

    return {
        "max_new_tokens": max_new_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "do_sample": True,
    }

async def _handle_streaming(request, inputs, start_time: float):
    """Handles streaming inference."""
    streamer = TextIteratorStreamer(state.tokenizer, skip_prompt=True, skip_special_tokens=True)
    config = _get_generation_config(request)
    
    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        streamer=streamer,
        **config
    )

    # Run generation in a separate thread to avoid blocking the event loop
    thread = Thread(target=state.model.generate, kwargs=generation_kwargs)
    thread.start()

    async def event_generator():
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

        # Yield initial role
        yield create_chunk(role="assistant")

        # Yield content tokens
        for new_text in streamer:
            if new_text:
                yield create_chunk(content=new_text)
                await asyncio.sleep(0) # Yield control to event loop

        # Yield finish
        yield create_chunk(finish_reason="stop")
        yield {"data": "[DONE]"}
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Stream finished. E2E Latency: {latency_ms:.2f}ms")

    return EventSourceResponse(event_generator())

async def _handle_non_streaming(request, inputs, start_time: float):
    """Handles non-streaming inference."""
    config = _get_generation_config(request)
    
    def generate_sync():
        with torch.no_grad():
            return state.model.generate(
                inputs.input_ids,
                **config
            )

    outputs = await asyncio.to_thread(generate_sync)
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    text = state.tokenizer.decode(generated_ids, skip_special_tokens=True)

    latency_ms = (time.time() - start_time) * 1000
    logger.info(f"Request finished. E2E Latency: {latency_ms:.2f}ms")

    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=text),
            finish_reason="stop"
        )],
        usage={
            "total_tokens": len(outputs[0]), 
            "completion_tokens": len(generated_ids), 
            "prompt_tokens": len(inputs.input_ids)
        }
    )
    
    return JSONResponse(
        content=response.model_dump(), 
        headers={"X-E2E-Latency-Ms": f"{latency_ms:.2f}"}
    )
