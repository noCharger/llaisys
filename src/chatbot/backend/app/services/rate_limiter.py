import time
import asyncio
from typing import Dict, Tuple, Optional
from .tenant_manager import TenantManager

class RateLimiter:
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self.buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def check_limit(self, tenant_id: str, cost: float = 1.0) -> bool:
        tenant = await self.tenant_manager.get_tenant(tenant_id)
        if not tenant:
            return False

        rpm = tenant.quotas.requests_per_minute
        if rpm <= 0:
            return False

        rate = rpm / 60.0
        capacity = float(rpm)

        now = time.time()

        async with self.lock:
            if tenant_id not in self.buckets:
                self.buckets[tenant_id] = (capacity, now)

            tokens, last_refill = self.buckets[tenant_id]

            elapsed = now - last_refill
            refill = elapsed * rate
            tokens = min(capacity, tokens + refill)

            if tokens >= cost:
                tokens -= cost
                self.buckets[tenant_id] = (tokens, now)
                return True
            else:
                self.buckets[tenant_id] = (tokens, now)
                return False
