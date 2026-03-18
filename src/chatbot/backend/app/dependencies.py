from fastapi import Request
from .services.tenant_manager import TenantManager
from .services.model_service import ModelService
from .services.scheduler import ClipperScheduler
from .services.queue_service import MemoryQueueService
from .services.chat_service import ChatService
from .dao.session_dao import SessionDAO
from .services.rate_limiter import RateLimiter
from .services.context_manager import ContextManager

tenant_manager = TenantManager()
rate_limiter = RateLimiter(tenant_manager)
model_service = ModelService()
queue_service = MemoryQueueService()
scheduler = ClipperScheduler(model_service, queue_service)
session_dao = SessionDAO()
context_manager = ContextManager()
chat_service = ChatService(scheduler, session_dao, context_manager)

def get_tenant_manager(request: Request) -> TenantManager:
    return request.app.state.tenant_manager

def get_rate_limiter() -> RateLimiter:
    return rate_limiter

def get_chat_service() -> ChatService:
    return chat_service

def get_scheduler() -> ClipperScheduler:
    return scheduler

def get_context_manager() -> ContextManager:
    return context_manager
