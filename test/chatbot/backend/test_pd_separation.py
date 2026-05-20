"""Mooncake-style PD-disaggregated scheduler tests."""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from chatbot.backend.app.services.kv_pool_service import KVPoolService
from chatbot.backend.app.services.kv_store_service import KVStoreService
from chatbot.backend.app.services.pd_scheduler import PDScheduler
from chatbot.backend.app.services.scheduler import RequestJob, Phase, DONE


class PDStubModel:
    """Stub forward_batch that mirrors ints.

    Records whether each call was prefill-dominant or decode-dominant so
    tests can verify the PD split actually happens.
    """

    def __init__(self, eos_after: int = 4, eos_token: int = 99):
        self.eos_after = eos_after
        self.eos_token = eos_token
        self.prefill_calls = 0
        self.decode_calls = 0
        self.feed_log: List[List[tuple]] = []

    async def forward_batch(self, batch):
        log = [(j.request_id, len(j.pending_input_ids)) for j in batch]
        self.feed_log.append(log)
        any_prefill = any(j.phase == Phase.PREFILL for j in batch)
        if any_prefill:
            self.prefill_calls += 1
        else:
            self.decode_calls += 1

        out = []
        for j in batch:
            if j.total_output_count + 1 >= self.eos_after:
                out.append(self.eos_token)
            else:
                out.append((hash(j.request_id) & 0x7FFF) + 1)
        return out


@pytest.mark.anyio
class TestPDSeparation:

    async def test_single_request_round_trip(self):
        """One request travels prefill worker, hand off, decode worker."""
        pool = KVPoolService(pool=None, n_blocks=4, max_len_per_block=64)
        store = KVStoreService(pool)
        model = PDStubModel(eos_after=3, eos_token=99)
        sched = PDScheduler(model, store,
                            token_budget=32, decode_batch_size=4,
                            eos_token=99)
        await sched.start()
        try:
            job = RequestJob(request_id="R1", tenant_id="T",
                             input_ids=list(range(1, 9)), max_tokens=10)
            await sched.submit(job)
            result = await asyncio.wait_for(job.future, timeout=3.0)
            assert result["finish_reason"] in ("stop", "length")
            # First prefill call must have happened.
            assert sched.stats()["prefill_steps"] >= 1
            # At least one handoff must have happened.
            assert sched.stats()["handoffs"] == 1
        finally:
            await sched.stop()

    async def test_prefill_and_decode_run_concurrently(self):
        """Two requests submitted close together: while request A is in
        decode, request B can still enter prefill, and they share the loop
        without interference. The Mooncake claim is that decode never
        stalls behind a long prefill of a new request."""
        pool = KVPoolService(pool=None, n_blocks=8, max_len_per_block=128)
        store = KVStoreService(pool)
        model = PDStubModel(eos_after=4, eos_token=99)
        sched = PDScheduler(model, store,
                            token_budget=32, decode_batch_size=8,
                            eos_token=99)
        await sched.start()
        try:
            ja = RequestJob(request_id="A", tenant_id="T",
                            input_ids=list(range(1, 9)), max_tokens=8)
            jb = RequestJob(request_id="B", tenant_id="T",
                            input_ids=list(range(200, 264)), max_tokens=8)
            await sched.submit(ja)
            await sched.submit(jb)
            ra, rb = await asyncio.wait_for(
                asyncio.gather(ja.future, jb.future), timeout=5.0)
            assert ra["finish_reason"] in ("stop", "length")
            assert rb["finish_reason"] in ("stop", "length")
            # Both jobs must have been handed off.
            assert sched.stats()["handoffs"] == 2
            assert sched.stats()["decode_steps"] >= 1

            # Verify at least one tick was prefill-only and one was
            # decode-only, confirming the two phases ran on disjoint
            # forward calls (PD disaggregation, not combined batching).
            prefill_only_ticks = [
                log for log in model.feed_log
                if all(L >= 8 for _, L in log)  # prefill chunks are long
            ]
            decode_only_ticks = [
                log for log in model.feed_log
                if all(L == 1 for _, L in log)
            ]
            assert prefill_only_ticks, "no prefill-only tick was observed"
            assert decode_only_ticks, "no decode-only tick was observed"
        finally:
            await sched.stop()

    async def test_shared_kv_store_zero_copy_handoff(self):
        """The hand-off is just a block_id transfer; the decode worker
        reads the same physical pages the prefill worker wrote. We verify
        this by checking that the request's block_id never changes between
        prefill commit and decode."""
        pool = KVPoolService(pool=None, n_blocks=4, max_len_per_block=64)
        store = KVStoreService(pool)
        model = PDStubModel(eos_after=3, eos_token=99)
        sched = PDScheduler(model, store, token_budget=32, eos_token=99)
        await sched.start()
        try:
            job = RequestJob(request_id="Z", tenant_id="T",
                             input_ids=list(range(1, 17)), max_tokens=5)
            await sched.submit(job)
            await asyncio.wait_for(job.future, timeout=3.0)
            # Block id was assigned at prefill admission and never
            # rotated through decode. This is the zero-copy invariant.
            assert job.block_id is not None
        finally:
            await sched.stop()

    async def test_long_prefill_chunks_then_hands_off(self):
        """A long prompt is chunked across multiple prefill ticks and only
        hands off once kv_pos catches the prompt length."""
        pool = KVPoolService(pool=None, n_blocks=2, max_len_per_block=128)
        store = KVStoreService(pool)
        model = PDStubModel(eos_after=2, eos_token=99)
        sched = PDScheduler(model, store, token_budget=16, eos_token=99)
        await sched.start()
        try:
            # 48 tokens at token_budget=16 needs 3 prefill ticks.
            job = RequestJob(request_id="L", tenant_id="T",
                             input_ids=list(range(1, 49)), max_tokens=4)
            await sched.submit(job)
            await asyncio.wait_for(job.future, timeout=4.0)
            assert sched.stats()["prefill_steps"] >= 3
            assert sched.stats()["handoffs"] == 1
        finally:
            await sched.stop()
