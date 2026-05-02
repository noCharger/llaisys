"""PagedAttention KV pool: lifecycle, prefix sharing, quota, cross-tenant wipe."""
import sys
import os
import ctypes

sys.path.insert(0, os.path.dirname(__file__))

import llaisys
from llaisys.libllaisys import LIB_LLAISYS, LlaisysQwen2Meta, DataType, DeviceType
from llaisys.models.qwen2 import PagedKVCache


def _build_stub_model(nlayer=2, nkvh=2, dh=4, dtype=DataType.F32):
    meta = LlaisysQwen2Meta()
    meta.dtype = dtype
    meta.nlayer = nlayer
    meta.hs = 8
    meta.nh = 2
    meta.nkvh = nkvh
    meta.dh = dh
    meta.di = 16
    meta.maxseq = 64
    meta.voc = 32
    meta.epsilon = 1e-5
    meta.theta = 10000.0
    meta.end_token = 0
    device_ids = (ctypes.c_int * 1)(0)
    model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
        ctypes.byref(meta), DeviceType.CPU, device_ids, 1)
    assert model_ptr, "stub model allocation failed"
    return model_ptr, meta


class _StubModel:
    def __init__(self, **kw):
        self.model, self.meta = _build_stub_model(**kw)

    def __del__(self):
        if getattr(self, "model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None


def _read_page_first_word(pool: PagedKVCache, page_id: int, layer: int = 0):
    """Return the first 4 floats of K[layer] for `page_id`."""
    t = pool.page_k(page_id, layer)
    assert t is not None
    n = 4
    buf = (ctypes.c_float * n)()
    api = llaisys.RuntimeAPI(DeviceType.CPU)
    api.memcpy_sync(buf, LIB_LLAISYS.tensorGetData(t),
                    n * ctypes.sizeof(ctypes.c_float),
                    llaisys.MemcpyKind.D2D)
    LIB_LLAISYS.tensorDestroy(t)  # we own the slice tensor
    return list(buf)


def test_basic_lifecycle():
    print("   acquire / append / release lifecycle")
    m = _StubModel()
    pool = PagedKVCache(m, n_pages=4, page_size=16, max_pages_per_request=8)
    assert pool.n_pages == 4
    assert pool.page_size == 16
    assert pool.global_pages_free == 4
    # Override the default 25% quota; this tiny pool needs full access.
    pool.set_tenant_quota("tenantA", 0, 4, 4)

    bid, matched = pool.acquire("tenantA", prefix_tokens=[1, 2, 3])
    assert bid >= 0 and matched == 0
    # No tokens written yet, so no pages allocated.
    assert pool.page_table(bid) == []
    assert pool.tenant_pages_used("tenantA") == 0

    # Append 5 tokens needs ceil(5/16) = 1 page.
    slots = pool.append(bid, 5)
    assert len(slots) == 5
    assert pool.tenant_pages_used("tenantA") == 1
    pt = pool.page_table(bid)
    assert len(pt) == 1
    # All 5 slots should map to that single page at offsets 0..4.
    for i, slot in enumerate(slots):
        page_id, offset = PagedKVCache.unpack_slot(slot)
        assert page_id == pt[0]
        assert offset == i

    # Append 30 more advances cursor 5..35 across 3 pages of size 16.
    slots2 = pool.append(bid, 30)
    pt = pool.page_table(bid)
    assert len(pt) == 3, f"expected 3 pages after 35 tokens, got {len(pt)}"
    assert pool.tenant_pages_used("tenantA") == 3
    assert pool.block_pos(bid) == 35

    pool.release(bid)
    # Released pages stay tagged with tenant; tenant_pages_used unchanged
    # because they sit in tenant.lru_free and still count against quota.
    assert pool.tenant_pages_used("tenantA") == 3


def test_prefix_share_increments_ref_count():
    print("   intra-tenant prefix sharing via chain hash + Commit")
    m = _StubModel()
    pool = PagedKVCache(m, n_pages=8, page_size=16, max_pages_per_request=8)
    pool.set_tenant_quota("T", 0, 8, 8)

    # Build a 32-token prefix and let session 1 commit it.
    prefix = list(range(1000, 1032))
    bid1, m1 = pool.acquire("T", prefix_tokens=prefix)
    assert m1 == 0
    pool.append(bid1, 32)
    pool.commit(bid1, new_pos=32, tokens=prefix)
    pages_after_session1 = pool.page_table(bid1)
    assert len(pages_after_session1) == 2
    pool.release(bid1)

    # Session 2 with same tenant and same prefix should match both pages.
    bid2, m2 = pool.acquire("T", prefix_tokens=prefix + [99, 100])
    assert m2 == 32, f"expected 32-token chain match, got {m2}"
    pages_session2 = pool.page_table(bid2)
    assert pages_session2 == pages_after_session1, \
        f"expected same pages reused, got {pages_session2} vs {pages_after_session1}"
    # tenant_pages_used should NOT increase since we share existing pages.
    assert pool.tenant_pages_used("T") == 2, \
        f"expected 2 pages in use, got {pool.tenant_pages_used('T')}"


def test_quota_enforcement():
    print("   max_pages quota rejects over-allocation")
    m = _StubModel()
    pool = PagedKVCache(m, n_pages=8, page_size=16, max_pages_per_request=4)
    pool.set_tenant_quota("X", reservation_floor=0, max_pages=2, burst_pages=0)

    bid, _ = pool.acquire("X", prefix_tokens=[])
    # Append 32 tokens fills 2 pages, exactly within quota.
    pool.append(bid, 32)
    assert pool.tenant_pages_used("X") == 2

    # 1 more token would need a 3rd page and must fail at max_pages=2.
    try:
        pool.append(bid, 1)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "expected quota exhaustion exception"
    pool.release(bid)


def test_cross_tenant_wipe():
    print("   cross-tenant takeover zero-wipes the page")
    m = _StubModel()
    pool = PagedKVCache(m, n_pages=2, page_size=16, max_pages_per_request=4)
    # Default tenant quotas would round 25% of n_pages down to 0, so set
    # explicit quotas large enough to exercise both tenants.
    pool.set_tenant_quota("A", 0, 2, 2)
    pool.set_tenant_quota("B", 0, 2, 2)

    # Tenant A claims both pages; poison page 0's K layer 0 with 7's.
    bid_a, _ = pool.acquire("A", prefix_tokens=[])
    pool.append(bid_a, 32)  # 2 pages
    pages_a = pool.page_table(bid_a)
    poison = (ctypes.c_float * 4)(7.0, 7.0, 7.0, 7.0)
    api = llaisys.RuntimeAPI(DeviceType.CPU)
    t = pool.page_k(pages_a[0], 0)
    api.memcpy_sync(LIB_LLAISYS.tensorGetData(t), poison, 16,
                    llaisys.MemcpyKind.D2D)
    LIB_LLAISYS.tensorDestroy(t)

    pool.release(bid_a)
    # No global free pages now. A holds both via lru_free, so B must trigger
    # cross-tenant eviction. A has reservation_floor=0, so eviction is allowed
    # and must zero-wipe before handing the page to B.
    bid_b, _ = pool.acquire("B", prefix_tokens=[])
    pool.append(bid_b, 16)  # 1 page
    pages_b = pool.page_table(bid_b)
    # The page B got must read all zeros.
    vals = _read_page_first_word(pool, pages_b[0])
    assert all(abs(v) < 1e-9 for v in vals), \
        f"cross-tenant page {pages_b[0]} not wiped: {vals}"


def test_cross_tenant_prefix_sharing():
    """Two tenants submitting the same prefix share the same physical pages.

    The chain-hash index is global, not partitioned by tenant. Identical
    input tokens produce bit-identical KV under a deterministic forward,
    so cross-tenant sharing is safe. Tenant isolation is enforced at auth,
    quota, and request routing layers, not by forbidding the cache hit.
    """
    print("   cross-tenant prefix match via global chain index")
    m = _StubModel()
    pool = PagedKVCache(m, n_pages=4, page_size=16, max_pages_per_request=4)
    pool.set_tenant_quota("A", 0, 4, 4)
    pool.set_tenant_quota("B", 0, 4, 4)

    # Tenant A writes a 32-token prefix and commits.
    prefix = list(range(500, 532))
    bid_a, m1 = pool.acquire("A", prefix_tokens=prefix)
    assert m1 == 0
    pool.append(bid_a, 32)
    pool.commit(bid_a, new_pos=32, tokens=prefix)
    pages_a = pool.page_table(bid_a)
    assert len(pages_a) == 2
    pool.release(bid_a)

    # Tenant B with the SAME prefix MUST hit the cache.
    bid_b, m2 = pool.acquire("B", prefix_tokens=prefix + [99, 100])
    assert m2 == 32, f"expected 32-token chain match, got {m2}"
    pages_b = pool.page_table(bid_b)
    assert pages_b == pages_a, \
        f"expected B to reuse A's pages, got {pages_b} vs {pages_a}"


def test_global_free_list_consistency():
    print("   global free list shrinks/grows symmetrically")
    m = _StubModel()
    pool = PagedKVCache(m, n_pages=4, page_size=16, max_pages_per_request=4)
    pool.set_tenant_quota("Z", 0, 4, 4)
    assert pool.global_pages_free == 4

    bid, _ = pool.acquire("Z", prefix_tokens=[])
    pool.append(bid, 16)  # 1 page
    assert pool.global_pages_free == 3
    pool.append(bid, 16)  # 1 more page
    assert pool.global_pages_free == 2
    pool.release(bid)
    # Pages are in tenant Z's lru_free, NOT global. Global is unchanged.
    assert pool.global_pages_free == 2
    assert pool.tenant_pages_used("Z") == 2


def _registered_tests():
    return [
        test_basic_lifecycle,
        test_prefix_share_increments_ref_count,
        test_quota_enforcement,
        test_cross_tenant_wipe,
        test_cross_tenant_prefix_sharing,
        test_global_free_list_consistency,
    ]


if __name__ == "__main__":
    print("Testing PagedKVCache")
    test_basic_lifecycle()
    test_prefix_share_increments_ref_count()
    test_quota_enforcement()
    test_cross_tenant_wipe()
    test_cross_tenant_prefix_sharing()
    test_global_free_list_consistency()
    print("\033[92mTest passed!\033[0m\n")
