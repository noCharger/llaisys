"""KV Cache Pool service. Tenant-scoped wrapper around the C++ pool.

Two modes: `paged` (real PagedKVCache) and `stub` (in-memory dict for tests).
Quotas are pulled from TenantManager and pushed into the C pool on first use.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any

logger = logging.getLogger("llaisys.kv_pool_service")


@dataclass
class _StubBlock:
    tenant_id: str = ""           # "" = free
    in_use: bool = False
    pos: int = 0
    last_used_ns: int = 0
    prefix: List[int] = field(default_factory=list)


class KVPoolService:
    """Tenant-scoped wrapper around a C++ KV pool (paged or contiguous)."""

    def __init__(self,
                 paged_pool: Optional[Any] = None,
                 n_blocks: int = 0, max_len_per_block: int = 0,
                 tenant_manager: Optional[Any] = None,
                 pool: Optional[Any] = None):
        """paged_pool=None → stub mode with n_blocks slots."""
        if pool is not None:
            logger.warning("KVPoolService: 'pool=' argument is removed; ignored.")

        self._paged = paged_pool
        self._tenant_manager = tenant_manager
        self._max_len = max_len_per_block
        self._n_blocks = n_blocks
        if paged_pool is not None:
            self._n_blocks = int(paged_pool.n_pages)
            self._max_len = int(paged_pool.page_size)

        self._lock = threading.Lock()
        self._stub_blocks: List[_StubBlock] = (
            [_StubBlock() for _ in range(n_blocks)]
            if paged_pool is None
            else []
        )
        self._paged_block_tenants: dict = {}
        self._paged_quota_applied: set = set()

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def max_len(self) -> int:
        return self._max_len

    @property
    def is_real(self) -> bool:
        return self._paged is not None

    @property
    def is_paged(self) -> bool:
        return self._paged is not None

    def underlying(self):
        """Real backend (paged) for model_service.forward; None in stub mode."""
        return self._paged

    # ---------- per-tenant quota (paged mode) ----------

    def _ensure_paged_quota(self, tenant_id: str):
        """Push tenant.quotas into the C pool on first use; idempotent."""
        if not self.is_paged or tenant_id in self._paged_quota_applied:
            return
        if self._tenant_manager is None:
            self._paged_quota_applied.add(tenant_id)
            return  # use C pool defaults
        try:
            tenant = self._tenant_manager.get_tenant_sync(tenant_id) \
                if hasattr(self._tenant_manager, "get_tenant_sync") else None
        except Exception:
            tenant = None
        if tenant is None:
            self._paged_quota_applied.add(tenant_id)
            return
        q = tenant.quotas
        if q.kv_pages_max > 0 or q.kv_pages_burst > 0 or q.kv_pages_reservation_floor > 0:
            self._paged.set_tenant_quota(
                tenant_id,
                int(q.kv_pages_reservation_floor),
                int(q.kv_pages_max) if q.kv_pages_max > 0 else self._n_blocks,
                int(q.kv_pages_burst),
            )
        self._paged_quota_applied.add(tenant_id)

    # ---------- entry points (tenant_id mandatory) ----------

    def acquire(self, tenant_id: str,
                prefix_tokens: Optional[List[int]] = None) -> Tuple[int, int]:
        """Returns (block_id, matched_prefix_len), or (-1, 0) on exhaustion."""
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        prefix_tokens = prefix_tokens or []

        if self._paged is not None:
            self._ensure_paged_quota(tenant_id)
            bid, matched = self._paged.acquire(tenant_id, prefix_tokens)
            if bid >= 0:
                self._paged_block_tenants[bid] = tenant_id
            return bid, matched

        # Stub: best-fit prefix match, else free, else LRU (cross-tenant wipe).
        with self._lock:
            return self._stub_acquire(tenant_id, prefix_tokens)

    def release(self, tenant_id: str, block_id: int):
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if self._paged is not None:
            owner = self._paged_block_tenants.get(block_id)
            if owner is not None and owner != tenant_id:
                logger.warning(
                    "release: tenant %s tried to release block owned by %s",
                    tenant_id, owner)
                return
            self._paged.release(block_id)
            self._paged_block_tenants.pop(block_id, None)
            return
        with self._lock:
            if 0 <= block_id < len(self._stub_blocks):
                b = self._stub_blocks[block_id]
                if b.tenant_id != tenant_id:
                    logger.warning(
                        "release: tenant %s tried to release block owned by %s",
                        tenant_id, b.tenant_id)
                    return
                b.in_use = False
                b.last_used_ns = _now_ns()

    def commit(self, tenant_id: str, block_id: int,
               new_pos: int, tokens: Optional[List[int]] = None):
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        tokens = tokens or []
        if self._paged is not None:
            owner = self._paged_block_tenants.get(block_id)
            if owner is not None and owner != tenant_id:
                logger.warning(
                    "commit: tenant %s tried to commit block owned by %s",
                    tenant_id, owner)
                return
            self._paged.commit(block_id, new_pos, tokens)
            return
        with self._lock:
            if 0 <= block_id < len(self._stub_blocks):
                b = self._stub_blocks[block_id]
                if b.tenant_id != tenant_id:
                    return
                b.pos = new_pos
                b.last_used_ns = _now_ns()
                if tokens:
                    b.prefix = list(tokens[:512])

    def append(self, tenant_id: str, block_id: int, n_new_tokens: int) -> List[int]:
        """Paged-only. Returns the slot mapping for n_new_tokens."""
        if self._paged is None:
            raise RuntimeError("KVPoolService.append requires paged mode")
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        owner = self._paged_block_tenants.get(block_id)
        if owner is not None and owner != tenant_id:
            raise ValueError(f"tenant {tenant_id} cannot append to block owned by {owner}")
        return self._paged.append(block_id, n_new_tokens)

    def block_pos(self, block_id: int) -> int:
        if self._paged is not None:
            return self._paged.block_pos(block_id)
        with self._lock:
            return self._stub_blocks[block_id].pos if 0 <= block_id < len(self._stub_blocks) else 0

    # ---------- diagnostics ----------

    def usage(self):
        if self._paged is not None:
            return {
                "n_pages": self.n_blocks,
                "page_size": self.max_len,
                "is_real": True,
                "global_free": int(self._paged.global_pages_free),
            }
        with self._lock:
            free = sum(1 for b in self._stub_blocks if b.tenant_id == "")
            in_use = sum(1 for b in self._stub_blocks if b.in_use)
            tenants = {}
            for b in self._stub_blocks:
                if b.tenant_id:
                    tenants[b.tenant_id] = tenants.get(b.tenant_id, 0) + 1
            return {
                "n_blocks": self.n_blocks,
                "max_len": self.max_len,
                "is_real": False,
                "free": free,
                "in_use": in_use,
                "tenants": tenants,
            }

    # ---------- internal ----------

    _CHUNK = 16  # match C++ pool's chunk granularity

    def _stub_acquire(self, tenant_id: str, prefix_tokens: List[int]):
        best_match_id = -1
        best_match_len = 0
        free_id = -1
        same_tenant_id = -1
        same_tenant_lru = float("inf")
        lru_other_id = -1
        lru_other_ts = float("inf")

        for i, b in enumerate(self._stub_blocks):
            if b.in_use:
                continue
            if b.tenant_id == tenant_id and b.prefix and prefix_tokens:
                lcp_tok = _lcp(b.prefix, prefix_tokens)
                if lcp_tok > b.pos:
                    lcp_tok = b.pos
                lcp = (lcp_tok // self._CHUNK) * self._CHUNK
                if lcp > best_match_len:
                    best_match_len = lcp
                    best_match_id = i
            if b.tenant_id == "":
                if free_id < 0:
                    free_id = i
            elif b.tenant_id == tenant_id:
                if b.last_used_ns < same_tenant_lru:
                    same_tenant_lru = b.last_used_ns
                    same_tenant_id = i
            else:
                if b.last_used_ns < lru_other_ts:
                    lru_other_ts = b.last_used_ns
                    lru_other_id = i

        chosen = -1
        matched = 0
        if best_match_id >= 0 and best_match_len > 0:
            chosen = best_match_id
            matched = best_match_len
        elif free_id >= 0:
            chosen = free_id
            b = self._stub_blocks[chosen]
            b.tenant_id = tenant_id
            b.pos = 0
            b.prefix = []
        elif same_tenant_id >= 0:
            chosen = same_tenant_id
            b = self._stub_blocks[chosen]
            b.pos = 0
            b.prefix = []
        elif lru_other_id >= 0:
            chosen = lru_other_id
            b = self._stub_blocks[chosen]
            b.tenant_id = tenant_id  # cross-tenant wipe simulated
            b.pos = 0
            b.prefix = []
        else:
            return -1, 0

        b = self._stub_blocks[chosen]
        b.in_use = True
        b.last_used_ns = _now_ns()
        return chosen, matched


def _lcp(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _now_ns() -> int:
    import time
    return time.monotonic_ns()
