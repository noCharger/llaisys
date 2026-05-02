"""Paged-KV scatter: tokens land in the expected (page, offset) slots."""
import sys
import os
import ctypes

sys.path.insert(0, os.path.dirname(__file__))

import torch
import llaisys
from llaisys.libllaisys import LIB_LLAISYS, LlaisysQwen2Meta, DataType, DeviceType
from llaisys.models.qwen2 import PagedKVCache


def _build_stub_model(nkvh=2, dh=4, dtype=DataType.F32, nlayer=2):
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
    model = LIB_LLAISYS.llaisysQwen2ModelCreate(
        ctypes.byref(meta), DeviceType.CPU, device_ids, 1)
    assert model
    return model, meta


class _StubModel:
    def __init__(self, **kw):
        self.model, self.meta = _build_stub_model(**kw)

    def __del__(self):
        if getattr(self, "model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None


def test_scatter_kv_round_trip():
    print("   scatter K/V into pages and read back")
    nkvh, dh, page_size = 2, 4, 16
    m = _StubModel(nkvh=nkvh, dh=dh)
    pool = PagedKVCache(m, n_pages=4, page_size=page_size, max_pages_per_request=4)
    pool.set_tenant_quota("T", 0, 4, 4)

    n_tokens = 20  # 1 full page + 1 partial (4 tokens)
    bid, _ = pool.acquire("T", prefix_tokens=[])
    slots = pool.append(bid, n_tokens)
    pages = pool.page_table(bid)
    assert len(pages) == 2

    # Build a deterministic K_new and V_new — token i row k is i*100 + k.
    K = torch.zeros((n_tokens, nkvh, dh), dtype=torch.float32)
    V = torch.zeros((n_tokens, nkvh, dh), dtype=torch.float32)
    for i in range(n_tokens):
        for h in range(nkvh):
            for d in range(dh):
                K[i, h, d] = float(i * 100 + h * 10 + d)
                V[i, h, d] = float(-i * 100 - h * 10 - d)

    k_lai = llaisys.Tensor((n_tokens, nkvh, dh), dtype=DataType.F32, device=DeviceType.CPU)
    v_lai = llaisys.Tensor((n_tokens, nkvh, dh), dtype=DataType.F32, device=DeviceType.CPU)
    api = llaisys.RuntimeAPI(DeviceType.CPU)
    api.memcpy_sync(k_lai.data_ptr(), K.contiguous().data_ptr(),
                    K.numel() * K.element_size(), llaisys.MemcpyKind.D2D)
    api.memcpy_sync(v_lai.data_ptr(), V.contiguous().data_ptr(),
                    V.numel() * V.element_size(), llaisys.MemcpyKind.D2D)

    # Scatter into layer 0 (the pool has nlayer=2; we just exercise one).
    pool.scatter_kv(layer=0, k_new=k_lai, v_new=v_lai, slot_mapping=slots)

    # Read each page back and check the slots we wrote.
    for token_idx in range(n_tokens):
        slot = slots[token_idx]
        page_id, offset = PagedKVCache.unpack_slot(slot)
        assert page_id in pages, f"slot points to page {page_id} not in {pages}"
        assert offset == token_idx % page_size

        # Page tensor shape [page_size, nkvh, dh]
        page_k = pool.page_k(page_id, layer=0)
        page_v = pool.page_v(page_id, layer=0)

        readback = torch.zeros((page_size, nkvh, dh), dtype=torch.float32)
        api.memcpy_sync(readback.data_ptr(), LIB_LLAISYS.tensorGetData(page_k),
                        readback.numel() * readback.element_size(),
                        llaisys.MemcpyKind.D2D)
        for h in range(nkvh):
            for d in range(dh):
                got = readback[offset, h, d].item()
                want = float(token_idx * 100 + h * 10 + d)
                assert abs(got - want) < 1e-6, \
                    f"K mismatch at token {token_idx}, h={h}, d={d}: got {got}, want {want}"

        readback_v = torch.zeros((page_size, nkvh, dh), dtype=torch.float32)
        api.memcpy_sync(readback_v.data_ptr(), LIB_LLAISYS.tensorGetData(page_v),
                        readback_v.numel() * readback_v.element_size(),
                        llaisys.MemcpyKind.D2D)
        for h in range(nkvh):
            for d in range(dh):
                got = readback_v[offset, h, d].item()
                want = float(-token_idx * 100 - h * 10 - d)
                assert abs(got - want) < 1e-6, \
                    f"V mismatch at token {token_idx}, h={h}, d={d}"

        LIB_LLAISYS.tensorDestroy(page_k)
        LIB_LLAISYS.tensorDestroy(page_v)


def test_scatter_does_not_clobber_other_pages():
    print("   scatter only touches pages in slot_mapping, leaves others unchanged")
    nkvh, dh, page_size = 2, 4, 16
    m = _StubModel(nkvh=nkvh, dh=dh)
    pool = PagedKVCache(m, n_pages=4, page_size=page_size, max_pages_per_request=4)
    pool.set_tenant_quota("T", 0, 4, 4)

    # Acquire & write into pages 0 and 1.
    bid_a, _ = pool.acquire("T", [])
    slots_a = pool.append(bid_a, 16)  # 1 page
    K_a = torch.full((16, nkvh, dh), 5.0, dtype=torch.float32)
    V_a = torch.full((16, nkvh, dh), -5.0, dtype=torch.float32)
    k_la = llaisys.Tensor((16, nkvh, dh), dtype=DataType.F32, device=DeviceType.CPU)
    v_la = llaisys.Tensor((16, nkvh, dh), dtype=DataType.F32, device=DeviceType.CPU)
    api = llaisys.RuntimeAPI(DeviceType.CPU)
    api.memcpy_sync(k_la.data_ptr(), K_a.data_ptr(), K_a.numel()*4, llaisys.MemcpyKind.D2D)
    api.memcpy_sync(v_la.data_ptr(), V_a.data_ptr(), V_a.numel()*4, llaisys.MemcpyKind.D2D)
    pool.scatter_kv(0, k_la, v_la, slots_a)

    bid_b, _ = pool.acquire("T", [])
    slots_b = pool.append(bid_b, 16)  # 1 page
    K_b = torch.full((16, nkvh, dh), 9.0, dtype=torch.float32)
    V_b = torch.full((16, nkvh, dh), -9.0, dtype=torch.float32)
    k_lb = llaisys.Tensor((16, nkvh, dh), dtype=DataType.F32, device=DeviceType.CPU)
    v_lb = llaisys.Tensor((16, nkvh, dh), dtype=DataType.F32, device=DeviceType.CPU)
    api.memcpy_sync(k_lb.data_ptr(), K_b.data_ptr(), K_b.numel()*4, llaisys.MemcpyKind.D2D)
    api.memcpy_sync(v_lb.data_ptr(), V_b.data_ptr(), V_b.numel()*4, llaisys.MemcpyKind.D2D)
    pool.scatter_kv(0, k_lb, v_lb, slots_b)

    # Read page A's contents back — must still be 5.0 (not clobbered by B's scatter).
    page_a = pool.page_table(bid_a)[0]
    page_a_k = pool.page_k(page_a, 0)
    rb = torch.zeros((16, nkvh, dh), dtype=torch.float32)
    api.memcpy_sync(rb.data_ptr(), LIB_LLAISYS.tensorGetData(page_a_k),
                    rb.numel()*4, llaisys.MemcpyKind.D2D)
    LIB_LLAISYS.tensorDestroy(page_a_k)
    assert torch.allclose(rb, K_a), "page A was clobbered by page B's scatter"


if __name__ == "__main__":
    print("Testing PagedKVCache.scatter_kv")
    test_scatter_kv_round_trip()
    test_scatter_does_not_clobber_other_pages()
    print("\033[92mTest passed!\033[0m\n")
