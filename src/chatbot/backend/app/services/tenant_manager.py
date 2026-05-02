from typing import Dict, Optional, List
import uuid
import secrets
import hashlib
import logging
from ..models.tenant import Tenant, TenantCreate, TenantUpdate, APIKey, Quota
from .storage import StorageBackend, MockRedisStorage

logger = logging.getLogger(__name__)

class TenantManager:
    def __init__(self, storage: Optional[StorageBackend] = None):
        self.storage = storage or MockRedisStorage()
        
    def _hash_key(self, api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

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
        await self.storage.save_tenant(tenant)
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return await self.storage.get_tenant(tenant_id)

    def get_tenant_sync(self, tenant_id: str) -> Optional[Tenant]:
        """Sync lookup; in-memory backends only. Used by KVPoolService."""
        getter = getattr(self.storage, "get_tenant_sync", None)
        if getter is not None:
            return getter(tenant_id)
        db = getattr(self.storage, "_tenants_db", None)
        if db is None:
            return None
        data = db.get(tenant_id)
        if data is None:
            return None
        if isinstance(data, Tenant):
            return data
        try:
            return Tenant.model_validate_json(data)
        except Exception:
            return None

    async def update_tenant(self, tenant_id: str, tenant_in: TenantUpdate) -> Optional[Tenant]:
        tenant = await self.storage.get_tenant(tenant_id)
        if not tenant:
            return None

        if tenant_in.name:
            tenant.name = tenant_in.name
        if tenant_in.quotas:
            tenant.quotas = tenant_in.quotas
        if tenant_in.features is not None:
            tenant.features.update(tenant_in.features)

        await self.storage.save_tenant(tenant)
        return tenant

    async def delete_tenant(self, tenant_id: str) -> bool:
        tenant = await self.storage.get_tenant(tenant_id)
        if not tenant:
            return False

        for key_obj in tenant.api_keys:
            await self.storage.delete_api_key_mapping(key_obj.key_hash)

        return await self.storage.delete_tenant(tenant_id)

    async def create_api_key(self, tenant_id: str, key_hint: Optional[str] = None) -> Optional[str]:
        tenant = await self.storage.get_tenant(tenant_id)
        if not tenant:
            return None

        raw_key = secrets.token_urlsafe(32)
        api_key = f"sk-{raw_key}"
        key_hash = self._hash_key(api_key)
        prefix = api_key[:6] + "..." + api_key[-4:]

        key_obj = APIKey(key_hash=key_hash, prefix=prefix)
        tenant.api_keys.append(key_obj)
        
        await self.storage.save_tenant(tenant)
        await self.storage.save_api_key_mapping(key_hash, tenant_id)

        return api_key

    async def validate_api_key(self, api_key: str) -> Optional[Tenant]:
        key_hash = self._hash_key(api_key)
        logger.debug(f"TenantManager: Computed Hash: {key_hash}")
        tenant_id = await self.storage.get_tenant_id_by_key(key_hash)
        if not tenant_id:
            if isinstance(self.storage, MockRedisStorage):
                logger.debug(f"TenantManager: Current keys in storage: {list(self.storage._keys_db.keys())}")
            logger.debug("TenantManager: Hash not found in storage mapping.")
            return None

        tenant = await self.storage.get_tenant(tenant_id)
        if not tenant:
            logger.warning(f"TenantManager: Tenant {tenant_id} not found in storage.")
            return None

        for k in tenant.api_keys:
            if k.key_hash == key_hash and k.is_active:
                logger.debug(f"TenantManager: Key matches and is active for Tenant {tenant.name}.")
                return tenant

        logger.debug("TenantManager: Key hash found in tenant list but is not active or mismatch.")
        return None

    async def revoke_api_key(self, tenant_id: str, api_key: str) -> bool:
        tenant = await self.storage.get_tenant(tenant_id)
        if not tenant:
            return False

        key_hash = self._hash_key(api_key)
        found = False
        for k in tenant.api_keys:
            if k.key_hash == key_hash:
                k.is_active = False
                found = True
                break
        
        if found:
            await self.storage.save_tenant(tenant)
            await self.storage.delete_api_key_mapping(key_hash)
            return True

        return False
