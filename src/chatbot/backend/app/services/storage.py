from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from ..models.tenant import Tenant

class StorageBackend(ABC):
    @abstractmethod
    async def save_tenant(self, tenant: Tenant) -> None:
        pass

    @abstractmethod
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        pass

    @abstractmethod
    async def delete_tenant(self, tenant_id: str) -> bool:
        pass

    @abstractmethod
    async def list_tenants(self) -> List[Tenant]:
        pass

    @abstractmethod
    async def save_api_key_mapping(self, api_key: str, tenant_id: str) -> None:
        pass

    @abstractmethod
    async def get_tenant_id_by_key(self, api_key: str) -> Optional[str]:
        pass

    @abstractmethod
    async def delete_api_key_mapping(self, api_key: str) -> bool:
        pass

class MockRedisStorage(StorageBackend):
    """
    A mock implementation of the storage backend simulating Redis/DB.
    In a real system, this would use aioredis/asyncpg.
    """
    def __init__(self):
        self._tenants_db: Dict[str, str] = {}
        self._keys_db: Dict[str, str] = {}

    async def save_tenant(self, tenant: Tenant) -> None:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"MockRedisStorage: Saving tenant {tenant.id} with {len(tenant.api_keys)} keys")
        self._tenants_db[tenant.id] = tenant.model_dump_json()

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        data = self._tenants_db.get(tenant_id)
        if data:
            return Tenant.model_validate_json(data)
        return None

    async def delete_tenant(self, tenant_id: str) -> bool:
        if tenant_id in self._tenants_db:
            del self._tenants_db[tenant_id]
            return True
        return False

    async def list_tenants(self) -> List[Tenant]:
        return [Tenant.model_validate_json(data) for data in self._tenants_db.values()]

    async def save_api_key_mapping(self, api_key: str, tenant_id: str) -> None:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"MockRedisStorage: Saving key mapping {api_key} -> {tenant_id}")
        self._keys_db[api_key] = tenant_id

    async def get_tenant_id_by_key(self, api_key: str) -> Optional[str]:
        return self._keys_db.get(api_key)

    async def delete_api_key_mapping(self, api_key: str) -> bool:
        if api_key in self._keys_db:
            del self._keys_db[api_key]
            return True
        return False
