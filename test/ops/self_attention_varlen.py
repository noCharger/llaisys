import sys
import os
from typing import List, Tuple

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from test_utils import (
    random_tensor,
    check_equal,
    llaisys_dtype,
    llaisys_device,
)


def _make_packed(
    request_shapes: List[Tuple[int, int]],  # list of (q_len, kv_len)
    nh: int,
    nkvh: int,
    hd: int,
    dtype_name: str,
    device_name: str,
):
    """Build packed Q + per-request K/V blocks + cu_seqlens_q.

    Also returns the equivalent torch tensors so the reference output can be
    computed by running the single-sequence kernel B times.
    """
    q_torch_list = []
    k_torch_list = []
    v_torch_list = []
    q_lai_list = []
    k_lai_list = []
    v_lai_list = []
    out_torch_list = []
    out_lai_list = []

    for q_len, kv_len in request_shapes:
        # Per-request reference tensors. We build them independently so the
        # single-sequence reference path consumes contiguous KV.
        q_t, q_l = random_tensor((q_len, nh, hd), dtype_name, device_name)
        k_t, k_l = random_tensor((kv_len, nkvh, hd), dtype_name, device_name)
        v_t, v_l = random_tensor((kv_len, nkvh, hd), dtype_name, device_name)
        out_t, out_l = random_tensor((q_len, nh, hd), dtype_name, device_name)

        q_torch_list.append(q_t)
        k_torch_list.append(k_t)
        v_torch_list.append(v_t)
        out_torch_list.append(out_t)

        q_lai_list.append(q_l)
        k_lai_list.append(k_l)
        v_lai_list.append(v_l)
        out_lai_list.append(out_l)

    return (
        q_torch_list,
        k_torch_list,
        v_torch_list,
        out_torch_list,
        q_lai_list,
        k_lai_list,
        v_lai_list,
        out_lai_list,
    )


def _pack_along_seqlen(tensors_lai, total_q, nh, hd, dtype_name, device_name):
    """Concatenate per-request llaisys tensors along dim 0 into a single
    contiguous packed tensor. Uses host-side numpy/torch concat, then copies
    back into a fresh llaisys tensor."""
    # Concat in torch (already on the right device) for efficiency
    parts = []
    for t in tensors_lai:
        # Convert llaisys tensor to torch view via memcpy
        shape = t.shape()
        torch_t = torch.empty(
            shape,
            dtype=_torch_dtype(dtype_name),
            device=_torch_device(device_name),
        )
        api = llaisys.RuntimeAPI(llaisys_device(device_name))
        api.memcpy_sync(
            torch_t.data_ptr(),
            t.data_ptr(),
            torch_t.numel() * torch_t.element_size(),
            llaisys.MemcpyKind.D2D,
        )
        parts.append(torch_t)
    packed_torch = torch.cat(parts, dim=0).contiguous()

    packed_lai = llaisys.Tensor(
        (total_q, nh, hd),
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
    )
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        packed_lai.data_ptr(),
        packed_torch.data_ptr(),
        packed_torch.numel() * packed_torch.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return packed_lai, packed_torch


def _torch_dtype(name):
    return {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _torch_device(name):
    return torch.device("cpu" if name == "cpu" else f"cuda")


def test_varlen(request_shapes, nh, nkvh, hd, dtype_name, atol, rtol, device_name):
    print(
        f"   batch={len(request_shapes)} shapes={request_shapes} "
        f"nh={nh} nkvh={nkvh} hd={hd} dtype<{dtype_name}>"
    )

    (
        q_torch_list,
        k_torch_list,
        v_torch_list,
        out_torch_list,
        q_lai_list,
        k_lai_list,
        v_lai_list,
        _out_lai_list,
    ) = _make_packed(request_shapes, nh, nkvh, hd, dtype_name, device_name)

    scale = 1.0 / (hd**0.5)

    # 1. Reference: run single-sequence self_attention for each request and
    # collect the outputs back into a packed buffer.
    ref_outs_lai = []
    for q_l, k_l, v_l, q_t in zip(q_lai_list, k_lai_list, v_lai_list, q_torch_list):
        ref_out = llaisys.Tensor(
            q_t.shape,
            dtype=llaisys_dtype(dtype_name),
            device=llaisys_device(device_name),
        )
        llaisys.Ops.self_attention(ref_out, q_l, k_l, v_l, scale)
        ref_outs_lai.append(ref_out)

    total_q = sum(s[0] for s in request_shapes)

    # 2. Pack q's and known reference outputs into single packed tensors.
    packed_q, _ = _pack_along_seqlen(q_lai_list, total_q, nh, hd, dtype_name, device_name)
    ref_packed, ref_packed_torch = _pack_along_seqlen(
        ref_outs_lai, total_q, nh, hd, dtype_name, device_name
    )

    # 3. Allocate fresh packed output for varlen call.
    packed_out = llaisys.Tensor(
        (total_q, nh, hd),
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
    )

    # 4. Build cu_seqlens_q
    cu_q = [0]
    for q_len, _ in request_shapes:
        cu_q.append(cu_q[-1] + q_len)

    # 5. Call varlen op with K/V blocks coming straight from per-request tensors.
    llaisys.Ops.self_attention_varlen(
        packed_out,
        packed_q,
        k_lai_list,
        v_lai_list,
        cu_q,
        scale,
    )

    # 6. Compare packed_out to ref_packed_torch.
    assert check_equal(packed_out, ref_packed_torch, atol=atol, rtol=rtol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()

    # (request_shapes, nh, nkvh, hd)
    test_cases = [
        # one decode-style request: q=1 token, kv=8 cached tokens
        ([(1, 8)], 4, 2, 8),
        # two prefill-only requests, identical lengths
        ([(4, 4), (4, 4)], 4, 2, 8),
        # mixed prefill/decode batch
        ([(5, 5), (1, 11), (3, 3)], 4, 2, 8),
        # single request that matches the existing self_attention test exactly
        ([(2, 2)], 1, 1, 4),
        # bigger heads
        ([(7, 13), (1, 9)], 8, 4, 16),
    ]
    test_dtypes = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.self_attention_varlen on {args.device}")
    for shapes, nh, nkvh, hd in test_cases:
        for dtype_name, atol, rtol in test_dtypes:
            test_varlen(shapes, nh, nkvh, hd, dtype_name, atol, rtol, args.device)

    print("\033[92mTest passed!\033[0m\n")
