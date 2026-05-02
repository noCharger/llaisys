"""forward_batch_paged self-consistency: valid ids, determinism, batch parity."""
import sys
import os
import ctypes

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import llaisys
from llaisys.libllaisys import LIB_LLAISYS, LlaisysQwen2Meta, DataType, DeviceType
from llaisys.models.qwen2 import PagedKVCache


def _build_stub_model(nlayer=2, hs=16, nh=4, nkvh=2, di=32, voc=37, maxseq=64,
                      dtype=DataType.F32, seed=0):
    meta = LlaisysQwen2Meta()
    meta.dtype = dtype
    meta.nlayer = nlayer; meta.hs = hs; meta.nh = nh; meta.nkvh = nkvh
    meta.dh = hs // nh; meta.di = di
    meta.maxseq = maxseq; meta.voc = voc
    meta.epsilon = 1e-5; meta.theta = 10000.0; meta.end_token = -1
    device_ids = (ctypes.c_int * 1)(0)
    model = LIB_LLAISYS.llaisysQwen2ModelCreate(
        ctypes.byref(meta), DeviceType.CPU, device_ids, 1)
    weights = LIB_LLAISYS.llaisysQwen2ModelWeights(model).contents
    rng = np.random.default_rng(seed)
    def fill(t, shape):
        data = rng.standard_normal(shape).astype(np.float32) * 0.1
        LIB_LLAISYS.tensorLoad(t, data.ctypes.data_as(ctypes.c_void_p))
    fill(weights.in_embed, (voc, hs)); fill(weights.out_embed, (voc, hs))
    fill(weights.out_norm_w, (hs,))
    for L in range(nlayer):
        fill(weights.attn_norm_w[L], (hs,))
        fill(weights.attn_q_w[L], (nh*meta.dh, hs)); fill(weights.attn_q_b[L], (nh*meta.dh,))
        fill(weights.attn_k_w[L], (nkvh*meta.dh, hs)); fill(weights.attn_k_b[L], (nkvh*meta.dh,))
        fill(weights.attn_v_w[L], (nkvh*meta.dh, hs)); fill(weights.attn_v_b[L], (nkvh*meta.dh,))
        fill(weights.attn_o_w[L], (hs, nh*meta.dh))
        fill(weights.mlp_norm_w[L], (hs,))
        fill(weights.mlp_gate_w[L], (di, hs))
        fill(weights.mlp_up_w[L], (di, hs))
        fill(weights.mlp_down_w[L], (hs, di))
    return model, meta


class _Holder:
    def __init__(self, model, meta):
        self.model = model
        self.meta = meta
    def __del__(self):
        if getattr(self, "model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None


def _run_paged(model, meta, input_ids, tenant="ten"):
    class _M:
        def __init__(self, m): self.model = m
    qmodel = _M(model)
    paged = PagedKVCache(qmodel, n_pages=4, page_size=16, max_pages_per_request=4)
    paged.set_tenant_quota(tenant, 0, 4, 4)
    bid, _ = paged.acquire(tenant, input_ids)
    slots = paged.append(bid, len(input_ids))

    cu = (ctypes.c_int32 * 2)(0, len(input_ids))
    bids = (ctypes.c_int32 * 1)(bid)
    tok_arr = (ctypes.c_int64 * len(input_ids))(*input_ids)
    slot_arr = (ctypes.c_int32 * len(input_ids))(*slots)
    temps = (ctypes.c_float * 1)(1.0)
    ps = (ctypes.c_float * 1)(1.0)
    ks = (ctypes.c_int * 1)(1)
    out = (ctypes.c_int64 * 1)()
    rc = LIB_LLAISYS.llaisysQwen2ModelForwardBatchPaged(
        model, paged._pool, tok_arr, cu, bids, slot_arr,
        temps, ps, ks, ctypes.c_int32(1), out)
    assert rc == 0, f"paged forward returned {rc}"
    return int(out[0])


def test_paged_greedy_is_deterministic():
    print("   forward_batch_paged is deterministic under greedy decode")
    model, meta = _build_stub_model(seed=1)
    h = _Holder(model, meta)
    input_ids = [3, 7, 11, 4, 9]
    a = _run_paged(model, meta, input_ids, tenant="X")
    b = _run_paged(model, meta, input_ids, tenant="Y")
    assert 0 <= a < meta.voc
    assert a == b, f"non-determinism: {a} vs {b}"


def test_paged_two_request_independence():
    print("   batch=2 paged forward: each request matches its own batch=1 output")
    model, meta = _build_stub_model(seed=7)
    h = _Holder(model, meta)
    req_a = [2, 5, 9]
    req_b = [1, 4, 7, 10, 13]
    ref_a = _run_paged(model, meta, req_a, tenant="A")
    ref_b = _run_paged(model, meta, req_b, tenant="B")

    # Now run them packed together.
    class _M:
        def __init__(self, m): self.model = m
    qmodel = _M(model)
    paged = PagedKVCache(qmodel, n_pages=8, page_size=16, max_pages_per_request=4)
    paged.set_tenant_quota("Pa", 0, 4, 4)
    paged.set_tenant_quota("Pb", 0, 4, 4)
    bid_a, _ = paged.acquire("Pa", req_a)
    bid_b, _ = paged.acquire("Pb", req_b)
    slots_a = paged.append(bid_a, len(req_a))
    slots_b = paged.append(bid_b, len(req_b))

    packed = req_a + req_b
    cu = (ctypes.c_int32 * 3)(0, len(req_a), len(req_a) + len(req_b))
    bids = (ctypes.c_int32 * 2)(bid_a, bid_b)
    tok_arr = (ctypes.c_int64 * len(packed))(*packed)
    slot_arr = (ctypes.c_int32 * len(packed))(*(slots_a + slots_b))
    temps = (ctypes.c_float * 2)(1.0, 1.0)
    ps = (ctypes.c_float * 2)(1.0, 1.0)
    ks = (ctypes.c_int * 2)(1, 1)
    out = (ctypes.c_int64 * 2)()
    rc = LIB_LLAISYS.llaisysQwen2ModelForwardBatchPaged(
        model, paged._pool, tok_arr, cu, bids, slot_arr,
        temps, ps, ks, ctypes.c_int32(2), out)
    assert rc == 0
    assert int(out[0]) == ref_a, f"req A broken: batched={out[0]} solo={ref_a}"
    assert int(out[1]) == ref_b, f"req B broken: batched={out[1]} solo={ref_b}"


if __name__ == "__main__":
    print("Testing forward_batch_paged self-consistency")
    test_paged_greedy_is_deterministic()
    test_paged_two_request_independence()
    print("\033[92mTest passed!\033[0m\n")
