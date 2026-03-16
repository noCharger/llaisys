import pytest
import asyncio
from fastapi.testclient import TestClient
from chatbot.backend.app.main import app
from chatbot.backend.app.dependencies import tenant_manager
from chatbot.backend.app.models.tenant import TenantCreate

client = TestClient(app)

@pytest.fixture
def api_key():

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def setup():
        tenant = await tenant_manager.create_tenant(TenantCreate(name="Auth Test Tenant"))
        key = await tenant_manager.create_api_key(tenant.id)
        return key

    key = loop.run_until_complete(setup())
    loop.close()
    return key

def test_auth_missing_header():
    response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid Authorization header"

def test_auth_invalid_header_format():
    response = client.post("/v1/chat/completions", headers={"Authorization": "Basic 123"}, json={})
    assert response.status_code == 401

def test_auth_invalid_key():
    response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer invalid-key"}, json={})
    assert response.status_code == 403

from chatbot.backend.app.dependencies import get_chat_service
from chatbot.backend.app.models.chat import ChatResponse

def test_auth_success(api_key):

    class MockService:
        async def chat_completion(self, request, tenant_id):
            return ChatResponse(
                id="test-id",
                created=1234567890,
                model=request.model,
                choices=[],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

    app.dependency_overrides[get_chat_service] = lambda: MockService()

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload
    )

    app.dependency_overrides = {}

    assert response.status_code == 200
    assert response.json()["id"] == "test-id"
