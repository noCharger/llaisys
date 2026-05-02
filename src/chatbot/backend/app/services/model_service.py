"""Model service: stub (canned strings) or real (Qwen2 + paged KV)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

logger = logging.getLogger("llaisys.model_service")


class ModelService:
    """Stub model service for tests that don't load a real model."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.is_loaded = True

    async def forward_batch(self, batch: List[Any]) -> List[Any]:
        await asyncio.sleep(0.05)
        results = []
        for job in batch:
            response_text = (
                f"Hello from AI (batched). You requested max "
                f"{job.max_tokens} tokens. "
            )
            if job.max_tokens and job.max_tokens > 100:
                response_text += (
                    "Here is a much longer and more detailed explanation "
                    "because you allowed more tokens. " * 5
                )
            results.append(response_text)
        return results

    def forward(self, session_ptr: Any, input_ids: List[int],
                temperature: float, top_p: float, top_k: int) -> str:
        return "Hello from AI (sequential)"


class BatchedQwen2Service:
    """Real-mode model service backed by llaisys Qwen2 + PagedKVCache."""

    def __init__(self, qwen2: Any, kv_pool_service: Any):
        self.qwen2 = qwen2
        self.kv_pool_service = kv_pool_service
        self.is_loaded = True

    async def forward_batch(self, batch: List[Any]) -> List[int]:
        if not batch:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._forward_batch_sync_paged, batch)

    def _pack_common(self, batch):
        packed_tokens: List[int] = []
        cu = [0]
        block_ids: List[int] = []
        temps: List[float] = []
        ps: List[float] = []
        ks: List[int] = []
        for job in batch:
            if job.block_id is None:
                raise RuntimeError(
                    f"job {job.request_id} entered execute without a KV block")
            packed_tokens.extend(job.pending_input_ids)
            cu.append(cu[-1] + len(job.pending_input_ids))
            block_ids.append(int(job.block_id))
            temps.append(float(job.temperature))
            ps.append(float(job.top_p))
            ks.append(int(job.top_k))
        return packed_tokens, cu, block_ids, temps, ps, ks

    def _forward_batch_sync_paged(self, batch: List[Any]) -> List[int]:
        packed_tokens, cu, block_ids, temps, ps, ks = self._pack_common(batch)
        slot_mapping: List[int] = []
        for job in batch:
            slots = self.kv_pool_service.append(
                job.tenant_id, int(job.block_id), len(job.pending_input_ids))
            slot_mapping.extend(slots)

        underlying = self.kv_pool_service.underlying()
        if underlying is None:
            raise RuntimeError(
                "BatchedQwen2Service requires a real KV pool (got stub)")

        out = self.qwen2.forward_batch_paged(
            underlying,
            packed_tokens=packed_tokens, cu_seqlens_q=cu, block_ids=block_ids,
            slot_mapping=slot_mapping,
            temps=temps, top_ps=ps, top_ks=ks,
        )
        return [int(t) for t in out]
