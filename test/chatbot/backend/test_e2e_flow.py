import pytest
from fastapi.testclient import TestClient
from chatbot.backend.app.main import app
from chatbot.backend.app.dependencies import tenant_manager, scheduler, rate_limiter
from chatbot.backend.app.models.tenant import TenantCreate, Quota
from unittest.mock import patch

@pytest.fixture
def client():

    with TestClient(app) as c:
        yield c

@pytest.mark.anyio
async def test_e2e_flow_with_auth(client):

    tenant = await tenant_manager.create_tenant(TenantCreate(
        name="E2E Test Tenant",
        quotas=Quota(requests_per_minute=10)
    ))
    api_key = await tenant_manager.create_api_key(tenant.id)

    resp = client.get("/config")
    assert resp.status_code == 200
    assert "apiUrl" in resp.json()

    with patch.object(scheduler, 'submit') as mock_submit:
        async def mock_submit_side_effect(job):
            job.future.set_result("Hello from AI")

        mock_submit.side_effect = mock_submit_side_effect

        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "qwen2",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello from AI"

        response_fail = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer invalid-key"},
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )
        assert response_fail.status_code == 403

@pytest.mark.anyio
async def test_rate_limiting_flow(client):

    tenant = await tenant_manager.create_tenant(TenantCreate(
        name="Rate Limit E2E",
        quotas=Quota(requests_per_minute=1)
    ))
    api_key = await tenant_manager.create_api_key(tenant.id)

    with patch.object(scheduler, 'submit') as mock_submit:
        async def mock_submit_side_effect(job):
            job.future.set_result("Hello")
        mock_submit.side_effect = mock_submit_side_effect

        resp1 = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"messages": [{"role": "user", "content": "1"}]}
        )
        assert resp1.status_code == 200

        resp2 = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"messages": [{"role": "user", "content": "2"}]}
        )

        assert resp2.status_code == 429
        assert "Rate limit exceeded" in resp2.json()["detail"]
