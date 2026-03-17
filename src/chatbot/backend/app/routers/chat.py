from fastapi import APIRouter, Depends, HTTPException, Request
from ..models.chat import ChatRequest, ChatResponse
from ..services.chat_service import ChatService
from ..dependencies import get_chat_service

router = APIRouter()

@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    chat_request: ChatRequest,
    request: Request,
    service: ChatService = Depends(get_chat_service)
):

    tenant_id = getattr(request.state, "tenant_id", "default-tenant")

    try:
        return await service.chat_completion(chat_request, tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
