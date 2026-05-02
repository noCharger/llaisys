import json
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..models.chat import ChatRequest, ChatResponse
from ..services.chat_service import ChatService
from ..services.scheduler import ClipperScheduler, RequestJob, DONE
from ..dependencies import get_chat_service, get_scheduler, get_tokenizer

router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    chat_request: ChatRequest,
    request: Request,
    service: ChatService = Depends(get_chat_service),
    scheduler: ClipperScheduler = Depends(get_scheduler),
    tokenizer = Depends(get_tokenizer),
):
    tenant_id = getattr(request.state, "tenant_id", "default-tenant")

    if chat_request.stream:
        return await _stream_completion(chat_request, tenant_id, scheduler, tokenizer)

    try:
        return await service.chat_completion(chat_request, tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_completion(req: ChatRequest, tenant_id: str,
                             scheduler: ClipperScheduler,
                             tokenizer) -> StreamingResponse:
    """SSE streaming via the scheduler's per-job tokens_queue."""
    last_user = req.messages[-1].content if req.messages else ""
    input_ids = tokenizer.encode(last_user)
    if not input_ids:
        input_ids = tokenizer.encode(" ") or [0]

    job = RequestJob(
        request_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        input_ids=input_ids,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens or 256,
    )
    await scheduler.submit(job)

    async def event_stream():
        chunk_id = job.request_id
        created = int(time.time())
        accumulated_ids: list = []

        yield _sse_chunk({
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": created, "model": req.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        })

        try:
            while True:
                tok = await job.tokens_queue.get()
                if tok is DONE:
                    break
                tok_int = int(tok)
                piece = tokenizer.decode_step(accumulated_ids, tok_int)
                accumulated_ids.append(tok_int)
                if not piece:
                    continue
                yield _sse_chunk({
                    "id": chunk_id, "object": "chat.completion.chunk",
                    "created": created, "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                })
        finally:
            yield _sse_chunk({
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": created, "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": job.finish_reason or "stop",
                }],
            })
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_chunk(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
