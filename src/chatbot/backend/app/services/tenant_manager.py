from typing import Dict, Optional, List
import uuid
import secrets
from ..models.tenant import Tenant, TenantCreate, TenantUpdate, APIKey, Quota

class TenantManager:
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._api_keys: Dict[str, str] = {}

    async def create_tenant(self, tenant_in: TenantCreate) -> Tenant:
        tenant_id = str(uuid.uuid4())
        quotas = tenant_in.quotas or Quota()
        features = tenant_in.features or {}
        tenant = Tenant(
            id=tenant_id,
            name=tenant_in.name,
            quotas=quotas,
            features=features,
            api_keys=[]
        )
        self._tenants[tenant_id] = tenant
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    async def update_tenant(self, tenant_id: str, tenant_in: TenantUpdate) -> Optional[Tenant]:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        if tenant_in.name:
            tenant.name = tenant_in.name
        if tenant_in.quotas:
            tenant.quotas = tenant_in.quotas

        if tenant_in.features is not None:
            tenant.features.update(tenant_in.features)

        return tenant

    async def delete_tenant(self, tenant_id: str) -> bool:
        if tenant_id in self._tenants:

            tenant = self._tenants[tenant_id]
            for key_obj in tenant.api_keys:
                if key_obj.key in self._api_keys:
                    del self._api_keys[key_obj.key]

            del self._tenants[tenant_id]
            return True
        return False

    async def create_api_key(self, tenant_id: str, key_hint: Optional[str] = None) -> Optional[str]:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        raw_key = secrets.token_urlsafe(32)
        api_key = f"sk-{raw_key}"

        key_obj = APIKey(key=api_key)
        tenant.api_keys.append(key_obj)
        self._api_keys[api_key] = tenant_id

        return api_key

    async def validate_api_key(self, api_key: str) -> Optional[Tenant]:
        tenant_id = self._api_keys.get(api_key)
        if not tenant_id:
            return None

        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        for k in tenant.api_keys:
            if k.key == api_key and k.is_active:
                return tenant

        return None

    async def revoke_api_key(self, tenant_id: str, api_key: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        for k in tenant.api_keys:
            if k.key == api_key:
                k.is_active = False
                if api_key in self._api_keys:
                    del self._api_keys[api_key]
                return True

        return False
