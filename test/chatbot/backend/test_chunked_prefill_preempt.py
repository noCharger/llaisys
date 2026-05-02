"""Chunked prefill, hash-chain prefix match, and preemption recovery."""
from __future__ import annotations

import asyncio
import pytest

from chatbot.backend.app.services.scheduler import (
    ClipperScheduler, RequestJob, DONE,
)
from chatbot.backend.app.services.kv_pool_service import KVPoolService


class CountingModel:
    """Returns ints. Each call records what each job was fed so tests can
    inspect chunked prefill behaviour."""

    def __init__(self, eos_after: int = 4, eos_token: int = 99):
        self.eos_after = eos_after
        self.eos_token = eos_token
        self.calls = 0
        self.feed_log = []  # list of [(request_id, chunk_len), ...] per call

    async def forward_batch(self, batch):
        self.calls += 1
        self.feed_log.append(
            [(j.request_id, len(j.pending_input_ids)) for j in batch]
        )
        out = []
        for j in batch:
            # Only count tokens that will actually be output (i.e., once the
            # job has fully prefilled). The scheduler filters mid-prefill
            # samples, so we can return any int safely.
            if len(j.output_tokens) + 1 >= self.eos_after:
                out.append(self.eos_token)
            else:
                out.append((hash(j.request_id) & 0x7FFF) + 1)  # avoid 0
        return out


@pytest.mark.anyio
class TestChunkedPrefill:
    async def test_long_prompt_chunks_under_budget(self):
        """An 80-token prompt with token_budget=32 should be chunked into
        3 ticks (32 + 32 + 16) before any output is produced."""
        pool = KVPoolService(pool=None, n_blocks=2, max_len_per_block=128)
        model = CountingModel(eos_after=2, eos_token=99)
        scheduler = ClipperScheduler(model, kv_pool_service=pool,
                                     eos_token=99, token_budget=32)
        scheduler.current_batch_size = 4.0
        await scheduler.start()
        try:
            j = RequestJob(request_id="L", tenant_id="T",
                           input_ids=list(range(1, 81)),  # 80 tokens
                           max_tokens=10)
            await scheduler.submit(j)

            # Drain output
            tokens = []
            while True:
                tok = await asyncio.wait_for(j.tokens_queue.get(), timeout=2.0)
                if tok is DONE:
                    break
                tokens.append(tok)

            # Inspect feed_log: first 2 ticks fed 32 tokens each (mid-prefill,
            # output discarded), 3rd tick fed 16 tokens and produced output[0].
            chunked_calls = [
                lengths for log in model.feed_log
                for (rid, lengths) in log if rid == "L"
            ]
            # First three calls correspond to prefill tail-chunks.
            assert chunked_calls[0] == 32, f"first chunk should be 32, got {chunked_calls[0]}"
            assert chunked_calls[1] == 32, f"second chunk should be 32, got {chunked_calls[1]}"
            assert chunked_calls[2] == 16, f"third chunk should be 16, got {chunked_calls[2]}"
            # After prefill completes, decode steps feed exactly 1 token each.
            assert all(c == 1 for c in chunked_calls[3:]), \
                f"decode steps should feed 1 token each, got {chunked_calls[3:]}"

            assert tokens[-1] == 99, "last token should be EOS"
        finally:
            await scheduler.stop()

    async def test_mixed_prefill_decode_in_one_tick(self):
        """A short prefill (≤budget − decode_demand) and a decode job share
        the same Execute step (Sarathi-Serve combined batching)."""
        pool = KVPoolService(pool=None, n_blocks=4, max_len_per_block=64)
        model = CountingModel(eos_after=3, eos_token=99)
        scheduler = ClipperScheduler(model, kv_pool_service=pool,
                                     eos_token=99, token_budget=64)
        scheduler.current_batch_size = 4.0
        await scheduler.start()
        try:
            # First job runs to completion to leave us in a known state.
            j_warm = RequestJob(request_id="W", tenant_id="T",
                                input_ids=[1], max_tokens=10)
            await scheduler.submit(j_warm)
            await asyncio.wait_for(j_warm.future, timeout=2.0)

            # Now submit two jobs at once: a short prefill + a job that's
            # essentially a decode-only equivalent with a single-token prompt.
            j_pf = RequestJob(request_id="P", tenant_id="T",
                              input_ids=list(range(1, 9)),  # 8 tokens
                              max_tokens=5)
            j_dc = RequestJob(request_id="D", tenant_id="T",
                              input_ids=[42], max_tokens=5)
            await scheduler.submit(j_pf)
            await scheduler.submit(j_dc)

            await asyncio.wait_for(asyncio.gather(j_pf.future, j_dc.future),
                                   timeout=3.0)

            # Find the call that batched both jobs together.
            mixed = [
                log for log in model.feed_log
                if {"P", "D"} <= {rid for (rid, _) in log}
            ]
            assert mixed, "P and D should have been packed in at least one tick"
            first = mixed[0]
            feeds = dict(first)
            # Prefill chunk feeds all 8 tokens; decode feeds 1.
            assert feeds["P"] == 8
            assert feeds["D"] == 1
        finally:
            await scheduler.stop()


@pytest.mark.anyio
class TestHashChainPrefix:
    async def test_same_tenant_resubmit_reuses_prefix(self):
        """After job 1 completes, job 2 in the same tenant with overlapping
        prefix of at least 16 tokens should observe matched_prefix_len > 0
        via the chain-hash index."""
        pool = KVPoolService(pool=None, n_blocks=2, max_len_per_block=128)

        # 32-token shared prefix.
        prefix = list(range(1000, 1032))

        # Simulate: tenant submits, generates, releases, then resubmits.
        bid1, m1 = pool.acquire("tenantX", prefix + [1])
        assert m1 == 0
        # Pretend prefill completed; commit the full history the way the
        # scheduler does during cleanup.
        pool.commit("tenantX", bid1, len(prefix) + 1, prefix + [1, 2, 3, 4])
        pool.release("tenantX", bid1)

        bid2, m2 = pool.acquire("tenantX", prefix + [99, 100])
        assert m2 == 32, f"expected 32-token chain match, got {m2}"
        assert bid2 == bid1, f"should reuse same block id, got {bid2}"

    async def test_cross_tenant_prefix_sharing(self):
        """Two tenants with the same token sequence share the same pages.

        This is safe because identical tokens produce identical KV under a
        deterministic forward, so cross-tenant sharing is pure memoization.
        Tenant isolation is enforced at auth, quota, and request routing
        layers, not by partitioning the prefix index.
        """
        pool = KVPoolService(pool=None, n_blocks=2, max_len_per_block=64)
        prefix = list(range(2000, 2032))
        bid1, _ = pool.acquire("tenantA", prefix)
        pool.commit("tenantA", bid1, len(prefix), prefix)
        pool.release("tenantA", bid1)

        # Tenant B with the SAME tokens MUST hit the cache.
        bid2, m2 = pool.acquire("tenantB", prefix)
        assert m2 == 32, f"cross-tenant should share prefix, got match={m2}"
        assert bid2 == bid1, "should reuse the same physical block"


@pytest.mark.anyio
class TestPreemption:
    async def test_pool_full_triggers_preemption(self):
        """current_batch_size=2 but pool size=1. When both jobs are eligible
        to run, the second pool.acquire fails and preemption evicts the
        first. Both eventually complete; the evicted job re-enters via the
        queue and recovers its prefix through the chain index."""
        pool = KVPoolService(pool=None, n_blocks=1, max_len_per_block=64)
        # eos_after high so neither job naturally finishes before preemption
        # has a chance to fire; max_tokens caps total output.
        model = CountingModel(eos_after=100, eos_token=99)
        scheduler = ClipperScheduler(model, kv_pool_service=pool,
                                     eos_token=99, token_budget=32,
                                     max_preemptions_per_job=5)
        scheduler.current_batch_size = 2.0  # batch wants 2 jobs concurrently
        await scheduler.start()
        try:
            j1 = RequestJob(request_id="J1", tenant_id="T",
                            input_ids=list(range(1, 5)),
                            max_tokens=4)
            j2 = RequestJob(request_id="J2", tenant_id="T",
                            input_ids=list(range(100, 104)),
                            max_tokens=4)
            await scheduler.submit(j1)
            await asyncio.sleep(0.02)  # let j1 enter active
            await scheduler.submit(j2)

            r1, r2 = await asyncio.wait_for(
                asyncio.gather(j1.future, j2.future), timeout=5.0)

            assert r1["finish_reason"] in ("stop", "length")
            assert r2["finish_reason"] in ("stop", "length")
            assert scheduler.preempt_count >= 1, \
                f"expected ≥1 preempt with pool=1 batch=2, got {scheduler.preempt_count}"
        finally:
            await scheduler.stop()

    async def test_preempt_bounded_by_max_preemptions(self):
        """If a job has been preempted max times, the next pool exhaustion
        cannot evict it. The new job is rejected instead of thrashing."""
        pool = KVPoolService(pool=None, n_blocks=1, max_len_per_block=64)
        model = CountingModel(eos_after=100, eos_token=99)  # never finishes
        scheduler = ClipperScheduler(model, kv_pool_service=pool,
                                     eos_token=99, token_budget=32,
                                     max_preemptions_per_job=0)  # cannot preempt
        scheduler.current_batch_size = 1.0
        await scheduler.start()
        try:
            j1 = RequestJob(request_id="A", tenant_id="T",
                            input_ids=[1, 2, 3], max_tokens=50)
            await scheduler.submit(j1)
            await asyncio.sleep(0.05)

            j2 = RequestJob(request_id="B", tenant_id="T",
                            input_ids=[10, 11, 12], max_tokens=50)
            await scheduler.submit(j2)

            # j2 must be rejected because preemption is disabled.
            with pytest.raises(RuntimeError, match="exhausted"):
                await asyncio.wait_for(j2.future, timeout=2.0)

            # Cancel j1 to clean up.
            j1.future.cancel()
            await asyncio.sleep(0.1)
        finally:
            await scheduler.stop()
