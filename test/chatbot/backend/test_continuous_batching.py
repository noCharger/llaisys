"""End-to-end integration test for the continuous-batching scheduler."""
from __future__ import annotations

import asyncio
import pytest

from chatbot.backend.app.services.scheduler import (
    ClipperScheduler, RequestJob, DONE,
)
from chatbot.backend.app.services.kv_pool_service import KVPoolService


class IntModelService:
    """Returns ints. Each request gets N decode steps before EOS."""

    def __init__(self, eos_after: int = 4, eos_token: int = 99):
        self.eos_after = eos_after
        self.eos_token = eos_token
        self.calls = 0

    async def forward_batch(self, batch):
        self.calls += 1
        out = []
        for j in batch:
            if j.tokens_generated + 1 >= self.eos_after:
                out.append(self.eos_token)
            else:
                # Emit a deterministic token derived from request id so we can
                # assert per-job streams stay separate.
                out.append(hash(j.request_id) & 0xFFFF)
        return out


@pytest.mark.anyio
class TestContinuousBatching:
    async def test_two_concurrent_requests_stream_independently(self):
        pool = KVPoolService(pool=None, n_blocks=4, max_len_per_block=32)
        model = IntModelService(eos_after=3, eos_token=99)
        scheduler = ClipperScheduler(model, kv_pool_service=pool,
                                     eos_token=99)
        scheduler.current_batch_size = 4.0

        await scheduler.start()
        try:
            j1 = RequestJob(request_id="A", tenant_id="tenantA",
                            input_ids=[1, 2, 3], max_tokens=10)
            j2 = RequestJob(request_id="B", tenant_id="tenantB",
                            input_ids=[4, 5], max_tokens=10)
            await scheduler.submit(j1)
            await scheduler.submit(j2)

            # Drain each token stream concurrently.
            async def drain(job):
                tokens = []
                while True:
                    tok = await asyncio.wait_for(job.tokens_queue.get(), timeout=2.0)
                    if tok is DONE:
                        break
                    tokens.append(tok)
                return tokens

            toks_a, toks_b = await asyncio.gather(drain(j1), drain(j2))

            assert len(toks_a) > 0, "stream A produced no tokens"
            assert len(toks_b) > 0, "stream B produced no tokens"
            # Last token of each must be EOS (99).
            assert toks_a[-1] == 99
            assert toks_b[-1] == 99
            # Streams use distinct hash-derived tokens for the non-EOS part.
            non_eos_a = [t for t in toks_a if t != 99]
            non_eos_b = [t for t in toks_b if t != 99]
            if non_eos_a and non_eos_b:
                assert non_eos_a[0] != non_eos_b[0], \
                    "streams A and B should not share token ids"

            # Pool blocks must have been released back.
            usage = pool.usage()
            assert usage["in_use"] == 0, f"blocks leaked: {usage}"
        finally:
            await scheduler.stop()

    async def test_tenant_isolation_in_pool(self):
        pool = KVPoolService(pool=None, n_blocks=2, max_len_per_block=16)

        # Tenant A acquires both blocks, fills with prefix.
        a1, _ = pool.acquire("tenantA", [1, 2, 3])
        pool.commit("tenantA", a1, 3, [1, 2, 3])
        a2, _ = pool.acquire("tenantA", [4])
        assert a1 >= 0 and a2 >= 0

        # Tenant B requests one — must fail because both are in_use.
        b, _ = pool.acquire("tenantB", [99])
        assert b == -1

        pool.release("tenantA", a1)

        # Tenant B can now acquire — block was wiped during cross-tenant takeover.
        b, matched = pool.acquire("tenantB", [99])
        assert b >= 0
        assert matched == 0, "B must not see A's prefix"

    async def test_block_released_on_finish(self):
        pool = KVPoolService(pool=None, n_blocks=1, max_len_per_block=16)
        model = IntModelService(eos_after=2, eos_token=99)
        scheduler = ClipperScheduler(model, kv_pool_service=pool,
                                     eos_token=99)
        scheduler.current_batch_size = 1.0
        await scheduler.start()
        try:
            j = RequestJob(request_id="X", tenant_id="t",
                           input_ids=[1], max_tokens=4)
            await scheduler.submit(j)
            await asyncio.wait_for(j.future, timeout=2.0)
            # After finish + cleanup, the only block must be free again.
            usage = pool.usage()
            assert usage["in_use"] == 0
        finally:
            await scheduler.stop()
