"""Per-tenant quota propagation from TenantManager into the paged KV pool."""
import sys
import os
import ctypes
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + "/src")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
import llaisys
from llaisys.libllaisys import LIB_LLAISYS, LlaisysQwen2Meta, DataType, DeviceType
from llaisys.models.qwen2 import PagedKVCache

from chatbot.backend.app.models.tenant import TenantCreate, Quota
from chatbot.backend.app.services.tenant_manager import TenantManager
from chatbot.backend.app.services.kv_pool_service import KVPoolService


def _build_stub_model(nkvh=2, dh=4, dtype=DataType.F32, nlayer=2):
    meta = LlaisysQwen2Meta()
    meta.dtype = dtype
    meta.nlayer = nlayer
    meta.hs = 8; meta.nh = 2; meta.nkvh = nkvh; meta.dh = dh
    meta.di = 16; meta.maxseq = 64; meta.voc = 32
    meta.epsilon = 1e-5; meta.theta = 10000.0; meta.end_token = 0
    device_ids = (ctypes.c_int * 1)(0)
    return LIB_LLAISYS.llaisysQwen2ModelCreate(
        ctypes.byref(meta), DeviceType.CPU, device_ids, 1)


class _StubModel:
    def __init__(self): self.model = _build_stub_model()
    def __del__(self):
        if getattr(self, "model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)


@pytest.mark.anyio
class TestPagedQuota:
    async def test_quota_pushed_into_pool_on_first_acquire(self):
        m = _StubModel()
        paged = PagedKVCache(m, n_pages=8, page_size=16, max_pages_per_request=4)

        tm = TenantManager()
        # Tenant A has a hard quota of 2 pages. B has 3.
        tenant_a = await tm.create_tenant(TenantCreate(
            name="A",
            quotas=Quota(kv_pages_max=2, kv_pages_burst=0, kv_pages_reservation_floor=0),
        ))
        tenant_b = await tm.create_tenant(TenantCreate(
            name="B",
            quotas=Quota(kv_pages_max=3, kv_pages_burst=0, kv_pages_reservation_floor=0),
        ))

        svc = KVPoolService(paged_pool=paged, tenant_manager=tm)
        assert svc.is_paged

        bid_a, _ = svc.acquire(tenant_a.id, [])
        # Append 32 tokens = 2 pages → fits A's quota.
        svc.append(tenant_a.id, bid_a, 32)
        assert paged.tenant_pages_used(tenant_a.id) == 2

        # 1 more page → A's quota exhausted.
        with pytest.raises(RuntimeError):
            svc.append(tenant_a.id, bid_a, 1)

        # B's quota is independent.
        bid_b, _ = svc.acquire(tenant_b.id, [])
        svc.append(tenant_b.id, bid_b, 32)  # 2 pages, well within 3
        assert paged.tenant_pages_used(tenant_b.id) == 2

    async def test_release_validates_tenant(self):
        m = _StubModel()
        paged = PagedKVCache(m, n_pages=4, page_size=16, max_pages_per_request=4)
        tm = TenantManager()
        tenant_a = await tm.create_tenant(TenantCreate(
            name="A",
            quotas=Quota(kv_pages_max=4, kv_pages_burst=0, kv_pages_reservation_floor=0),
        ))
        tenant_b = await tm.create_tenant(TenantCreate(
            name="B",
            quotas=Quota(kv_pages_max=4, kv_pages_burst=0, kv_pages_reservation_floor=0),
        ))

        svc = KVPoolService(paged_pool=paged, tenant_manager=tm)
        bid_a, _ = svc.acquire(tenant_a.id, [])

        # Tenant B trying to release A's block must be rejected (logged and skipped).
        svc.release(tenant_b.id, bid_a)  # logs warning, no-op
        # A's release still works.
        svc.release(tenant_a.id, bid_a)

    async def test_quota_zero_means_pool_default(self):
        m = _StubModel()
        paged = PagedKVCache(m, n_pages=8, page_size=16, max_pages_per_request=4)
        tm = TenantManager()
        # All zeros → service should NOT push anything; C pool default applies
        # (default = max_pages = max(1, n_pages/4) = 2).
        tenant = await tm.create_tenant(TenantCreate(
            name="C",
            quotas=Quota(),  # defaults: kv_pages_* = 0
        ))
        svc = KVPoolService(paged_pool=paged, tenant_manager=tm)
        bid, _ = svc.acquire(tenant.id, [])
        # 32 tokens = 2 pages, within default of 2.
        svc.append(tenant.id, bid, 32)
        assert paged.tenant_pages_used(tenant.id) == 2
        # 3rd page hits the pool default cap.
        with pytest.raises(RuntimeError):
            svc.append(tenant.id, bid, 1)
