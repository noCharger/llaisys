import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from chatbot.backend.app.models.tenant import Tenant, TenantCreate, TenantUpdate, Quota
from chatbot.backend.app.services.tenant_manager import TenantManager
from chatbot.backend.app.services.rate_limiter import RateLimiter

@pytest.fixture
def tenant_manager():
    return TenantManager()

@pytest.fixture
async def rate_limiter(tenant_manager):
    return RateLimiter(tenant_manager)

class TestTenantManager:
    @pytest.mark.anyio
    async def test_create_tenant(self, tenant_manager):
        tenant_create = TenantCreate(
            name="Test Tenant",
            quotas=Quota(requests_per_minute=60, max_tokens_per_day=100000)
        )
        tenant = await tenant_manager.create_tenant(tenant_create)
        assert tenant.id is not None
        assert tenant.name == "Test Tenant"
        assert tenant.quotas.requests_per_minute == 60
        assert tenant.api_keys == []

    @pytest.mark.anyio
    async def test_get_tenant(self, tenant_manager):
        tenant_create = TenantCreate(name="Get Tenant")
        created = await tenant_manager.create_tenant(tenant_create)
        fetched = await tenant_manager.get_tenant(created.id)
        assert fetched == created
        assert await tenant_manager.get_tenant("non-existent") is None

    @pytest.mark.anyio
    async def test_update_tenant(self, tenant_manager):
        tenant_create = TenantCreate(name="Update Tenant")
        created = await tenant_manager.create_tenant(tenant_create)

        update_data = TenantUpdate(name="Updated Name", quotas=Quota(requests_per_minute=100))
        updated = await tenant_manager.update_tenant(created.id, update_data)

        assert updated.name == "Updated Name"
        assert updated.quotas.requests_per_minute == 100

        fetched = await tenant_manager.get_tenant(created.id)
        assert fetched.name == "Updated Name"

    @pytest.mark.anyio
    async def test_delete_tenant(self, tenant_manager):
        tenant_create = TenantCreate(name="Delete Tenant")
        created = await tenant_manager.create_tenant(tenant_create)

        success = await tenant_manager.delete_tenant(created.id)
        assert success is True

        fetched = await tenant_manager.get_tenant(created.id)
        assert fetched is None

    @pytest.mark.anyio
    async def test_api_key_management(self, tenant_manager):
        tenant_create = TenantCreate(name="API Key Tenant")
        tenant = await tenant_manager.create_tenant(tenant_create)

        api_key = await tenant_manager.create_api_key(tenant.id, "test-key-1")
        assert api_key.startswith("sk-")

        valid_tenant = await tenant_manager.validate_api_key(api_key)
        assert valid_tenant.id == tenant.id

        await tenant_manager.revoke_api_key(tenant.id, api_key)
        assert await tenant_manager.validate_api_key(api_key) is None

    @pytest.mark.anyio
    async def test_feature_flags(self, tenant_manager):
        tenant_create = TenantCreate(
            name="Feature Tenant",
            features={"beta_access": True, "legacy_mode": False}
        )
        tenant = await tenant_manager.create_tenant(tenant_create)

        assert tenant.features["beta_access"] is True
        assert tenant.features["legacy_mode"] is False

        update_data = TenantUpdate(features={"beta_access": False, "new_feature": True})
        updated = await tenant_manager.update_tenant(tenant.id, update_data)

        assert updated.features["beta_access"] is False
        assert updated.features["legacy_mode"] is False
        assert updated.features["new_feature"] is True

class TestRateLimiter:
    @pytest.mark.anyio
    async def test_check_limit_basic(self, rate_limiter, tenant_manager):
        tenant = await tenant_manager.create_tenant(TenantCreate(
            name="Rate Limit Tenant",
            quotas=Quota(requests_per_minute=10)
        ))

        for _ in range(10):
            assert await rate_limiter.check_limit(tenant.id) is True

        assert await rate_limiter.check_limit(tenant.id) is False

    @pytest.mark.anyio
    async def test_token_refill(self, rate_limiter, tenant_manager):
        tenant = await tenant_manager.create_tenant(TenantCreate(
            name="Refill Tenant",
            quotas=Quota(requests_per_minute=60)
        ))

        with patch('chatbot.backend.app.services.rate_limiter.time.time') as mock_time:
            mock_time.return_value = 1000.0

            for _ in range(60):
                assert await rate_limiter.check_limit(tenant.id) is True

            assert await rate_limiter.check_limit(tenant.id) is False

            mock_time.return_value = 1001.0

            assert await rate_limiter.check_limit(tenant.id) is True
            assert await rate_limiter.check_limit(tenant.id) is False

    @pytest.mark.anyio
    async def test_concurrent_access(self, rate_limiter, tenant_manager):
        tenant = await tenant_manager.create_tenant(TenantCreate(
            name="Concurrent Tenant",
            quotas=Quota(requests_per_minute=1000)
        ))

        tasks = [rate_limiter.check_limit(tenant.id) for _ in range(100)]
        results = await asyncio.gather(*tasks)
        assert all(results)
