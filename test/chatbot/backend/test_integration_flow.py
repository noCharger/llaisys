import pytest
from fastapi.testclient import TestClient
from chatbot.backend.app.main import app
from chatbot.backend.app.dependencies import tenant_manager, scheduler, queue_service, rate_limiter, get_chat_service, chat_service
from chatbot.backend.app.models.tenant import TenantCreate, Quota
from chatbot.backend.app.services.chat_service import ChatService
from chatbot.backend.app.services.queue_service import MemoryQueueService
from chatbot.backend.app.services.scheduler import ClipperScheduler
from chatbot.backend.app.services.model_service import ModelService
from unittest.mock import patch

from chatbot.backend.app.services.context_manager import ContextManager

@pytest.fixture
def test_setup():
    pass

@pytest.mark.anyio
async def test_e2e_chat_completion():

    if hasattr(tenant_manager.storage, '_tenants_db'):
        tenant_manager.storage._tenants_db.clear()
        tenant_manager.storage._keys_db.clear()
    if hasattr(rate_limiter, 'buckets'):
        rate_limiter.buckets.clear()

    tenant = await tenant_manager.create_tenant(TenantCreate(name="Integration Test Tenant", quotas=Quota(requests_per_minute=100)))
    tenant_id = tenant.id
    api_key = await tenant_manager.create_api_key(tenant_id)

    qs = MemoryQueueService()
    ms = ModelService()
    sched = ClipperScheduler(ms, qs)
    cm = ContextManager()

    new_chat_service = ChatService(sched, chat_service.session_dao, cm)

    app.dependency_overrides[get_chat_service] = lambda: new_chat_service

    with patch("chatbot.backend.app.main.scheduler", sched):

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "model": "gpt-3.5-turbo"
                },
                headers={"Authorization": f"Bearer {api_key}"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "message" in data["choices"][0]

            assert "Hello from AI" in data["choices"][0]["message"]["content"]

    app.dependency_overrides.clear()
