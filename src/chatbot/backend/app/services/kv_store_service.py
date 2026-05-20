"""Mooncake-style shared KV store facade.

Wraps KVPoolService and exposes a PD-disaggregation oriented API. The
underlying paged pool already uses a global content-addressed prefix
index, so a Prefill worker's commit becomes immediately visible to any
Decode worker that acquires the same prefix. Hand-off is a block_id
transfer, not a KV copy.

This module is a thin facade. Keeping it separate from KVPoolService
makes the PD architecture explicit and gives a single place to add
distributed concerns later, for example RDMA migration, eviction
policies tuned for cross-node sharing, or per-node cache warmup.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .kv_pool_service import KVPoolService

logger = logging.getLogger("llaisys.kv_store")


@dataclass
class PrefillHandoff:
    """Carries everything a Decode worker needs to resume a request.

    block_id refers to a slot in the shared pool. The page table is
    already populated with the prefill KV, and the prefix index has been
    updated so prefix lookups by other tenants on the same tokens hit
    the same physical pages.
    """
    block_id: int
    prefilled_tokens: int
    first_decode_token: int


class KVStoreService:
    """Shared paged KV store, PD-aware.

    Two roles:
      * Prefill worker calls reserve_prefill, runs the prefill forward,
        commits the result, then publishes a PrefillHandoff.
      * Decode worker calls claim_for_decode with the request prompt and
        receives a block_id whose KV is already populated.
    """

    def __init__(self, pool_service: KVPoolService):
        self._pool = pool_service
        self._handoffs: Dict[str, PrefillHandoff] = {}
        self._handoff_lock = threading.Lock()

    @property
    def pool(self) -> KVPoolService:
        return self._pool

    def reserve_prefill(self, tenant_id: str,
                        prompt_tokens: List[int]) -> Tuple[int, int]:
        """Acquire a block for prefill. Returns (block_id, matched_len)."""
        return self._pool.acquire(tenant_id, prompt_tokens)

    def commit_prefill(self, tenant_id: str, block_id: int,
                       new_pos: int, tokens: List[int]):
        """Seal the prefill block so its pages enter the global prefix
        index. After this call any same-tokens acquire from any tenant
        will hit the same pages."""
        self._pool.commit(tenant_id, block_id, new_pos, tokens)

    def publish_handoff(self, request_id: str, handoff: PrefillHandoff):
        """Make a prefilled block discoverable by the Decode worker."""
        with self._handoff_lock:
            self._handoffs[request_id] = handoff

    def take_handoff(self, request_id: str) -> Optional[PrefillHandoff]:
        """Decode worker calls this to claim a request whose prefill is
        done. Returns None if the prefill is still in flight."""
        with self._handoff_lock:
            return self._handoffs.pop(request_id, None)

    def claim_for_decode(self, tenant_id: str,
                         prompt_tokens: List[int],
                         output_tokens: List[int]) -> Tuple[int, int]:
        """Decode worker claims a block whose prefix is already populated.

        Returns (block_id, total_kv_pos). In the Mooncake design this is
        zero copy: the prefix index already points at the pages the
        Prefill worker wrote, so claim_for_decode just acquires them by
        ref-count without moving data.
        """
        history = list(prompt_tokens) + list(output_tokens)
        block_id, matched = self._pool.acquire(tenant_id, history)
        if block_id < 0:
            return -1, 0
        # The Prefill worker committed up to len(prompt_tokens), so the
        # decode-side kv_pos starts there. matched may be shorter if the
        # underlying pool only chunk-aligned to page boundaries; commit
        # the unmatched tail by hand here.
        return block_id, matched

    def release(self, tenant_id: str, block_id: int):
        self._pool.release(tenant_id, block_id)

    def append(self, tenant_id: str, block_id: int, n: int) -> List[int]:
        return self._pool.append(tenant_id, block_id, n)

    def usage(self) -> Dict[str, Any]:
        d = dict(self._pool.usage())
        d["pending_handoffs"] = len(self._handoffs)
        return d
