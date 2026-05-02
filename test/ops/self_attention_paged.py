"""Paged self-attention parity vs. single-sequence self_attention reference."""
import sys
import os
import ctypes

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import llaisys
from test_utils import random_tensor, check_equal, llaisys_dtype, llaisys_device


def _build_paged_inputs(request_shapes, nh, nkvh, hd, page_size,
                       dtype_name, device_name):
    """Construct big_k, big_v with the per-request KV stuffed at known
    page positions, and return all the metadata needed by paged op.

    Returns: (big_k, big_v, q_packed, ref_outs_torch, block_tables,
              block_table_lens, cu_seqlens_q, kv_lens, packed_q_torch)
    """
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    n_requests = len(request_shapes)

    # Choose page_ids for each request — non-trivial mapping (page 5, 0, 3...)
    # to verify the paged op uses block_tables, not contiguous order.
    next_page = 0
    block_tables_flat = []
    block_table_lens = []
    kv_lens = []
    request_kv_pages = []  # per-request: list of (page_id, valid_offset_count)

    for q_len, kv_len in request_shapes:
        npages = (kv_len + page_size - 1) // page_size
        # Spread page ids non-contiguously: shuffle.
        pages = list(range(next_page, next_page + npages))
        # Reverse them to make sure the gather works for non-monotonic order.
        pages = list(reversed(pages))
        next_page += npages
        block_tables_flat.extend(pages)
        block_table_lens.append(npages)
        kv_lens.append(kv_len)
        request_kv_pages.append((pages, kv_len))

    n_total_pages = next_page + 1   # leave a buffer
    big_shape = (n_total_pages * page_size, nkvh, hd)

    big_k_torch = torch.zeros(big_shape, dtype=_torch_dtype(dtype_name))
    big_v_torch = torch.zeros(big_shape, dtype=_torch_dtype(dtype_name))

    # Generate per-request K/V; place them into big_k/big_v at the chosen pages.
    per_request_q = []
    per_request_kv_torch = []
    for r, (q_len, kv_len) in enumerate(request_shapes):
        # Random K/V for this request.
        k_t = torch.rand((kv_len, nkvh, hd), dtype=_torch_dtype(dtype_name))
        v_t = torch.rand((kv_len, nkvh, hd), dtype=_torch_dtype(dtype_name))
        q_t = torch.rand((q_len, nh, hd), dtype=_torch_dtype(dtype_name))

        per_request_kv_torch.append((k_t, v_t))
        per_request_q.append(q_t)

        pages, _ = request_kv_pages[r]
        copied = 0
        for p_idx, page_id in enumerate(pages):
            take = min(page_size, kv_len - copied)
            big_k_torch[page_id * page_size : page_id * page_size + take] = k_t[copied : copied + take]
            big_v_torch[page_id * page_size : page_id * page_size + take] = v_t[copied : copied + take]
            copied += take

    # Build packed Q.
    total_q = sum(q for q, _ in request_shapes)
    q_packed_torch = torch.cat(per_request_q, dim=0).contiguous()
    cu_seqlens_q = [0]
    for q_len, _ in request_shapes:
        cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)

    # Build llaisys tensors.
    big_k_lai = llaisys.Tensor(big_shape, dtype=llaisys_dtype(dtype_name),
                                device=llaisys_device(device_name))
    big_v_lai = llaisys.Tensor(big_shape, dtype=llaisys_dtype(dtype_name),
                                device=llaisys_device(device_name))
    q_lai = llaisys.Tensor((total_q, nh, hd), dtype=llaisys_dtype(dtype_name),
                            device=llaisys_device(device_name))
    api.memcpy_sync(big_k_lai.data_ptr(), big_k_torch.data_ptr(),
                    big_k_torch.numel() * big_k_torch.element_size(),
                    llaisys.MemcpyKind.D2D)
    api.memcpy_sync(big_v_lai.data_ptr(), big_v_torch.data_ptr(),
                    big_v_torch.numel() * big_v_torch.element_size(),
                    llaisys.MemcpyKind.D2D)
    api.memcpy_sync(q_lai.data_ptr(), q_packed_torch.data_ptr(),
                    q_packed_torch.numel() * q_packed_torch.element_size(),
                    llaisys.MemcpyKind.D2D)

    return (big_k_lai, big_v_lai, q_lai,
            per_request_kv_torch, per_request_q,
            block_tables_flat, block_table_lens, cu_seqlens_q, kv_lens)


def _torch_dtype(name):
    return {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[name]


def test_paged_attention(request_shapes, nh, nkvh, hd, page_size,
                          dtype_name, atol, rtol, device_name):
    print(f"   batch={len(request_shapes)} shapes={request_shapes} "
          f"page_size={page_size} dtype<{dtype_name}>")

    (big_k, big_v, q_packed, kv_torch_list, q_torch_list,
     block_tables, block_table_lens, cu_seqlens_q, kv_lens) = \
        _build_paged_inputs(request_shapes, nh, nkvh, hd, page_size,
                            dtype_name, device_name)

    scale = 1.0 / (hd ** 0.5)
    total_q = sum(q for q, _ in request_shapes)

    out = llaisys.Tensor((total_q, nh, hd), dtype=llaisys_dtype(dtype_name),
                          device=llaisys_device(device_name))

    llaisys.Ops.self_attention_paged(
        out, q_packed, big_k, big_v,
        block_tables, block_table_lens, cu_seqlens_q, kv_lens,
        page_size, scale,
    )

    # Reference: run single-sequence self_attention per request and stack.
    ref_pieces = []
    for r, ((q_len, kv_len), (k_t, v_t)) in enumerate(
            zip(request_shapes, kv_torch_list)):
        q_t = q_torch_list[r]
        # Use the existing varlen op as the oracle (it dispatches to single-seq).
        out_r_lai = llaisys.Tensor((q_len, nh, hd), dtype=llaisys_dtype(dtype_name),
                                     device=llaisys_device(device_name))
        # Build lai tensors for k_t, v_t, q_t.
        api = llaisys.RuntimeAPI(llaisys_device(device_name))
        q_r_lai = llaisys.Tensor((q_len, nh, hd), dtype=llaisys_dtype(dtype_name),
                                   device=llaisys_device(device_name))
        k_r_lai = llaisys.Tensor((kv_len, nkvh, hd), dtype=llaisys_dtype(dtype_name),
                                   device=llaisys_device(device_name))
        v_r_lai = llaisys.Tensor((kv_len, nkvh, hd), dtype=llaisys_dtype(dtype_name),
                                   device=llaisys_device(device_name))
        api.memcpy_sync(q_r_lai.data_ptr(), q_t.contiguous().data_ptr(),
                        q_t.numel() * q_t.element_size(), llaisys.MemcpyKind.D2D)
        api.memcpy_sync(k_r_lai.data_ptr(), k_t.contiguous().data_ptr(),
                        k_t.numel() * k_t.element_size(), llaisys.MemcpyKind.D2D)
        api.memcpy_sync(v_r_lai.data_ptr(), v_t.contiguous().data_ptr(),
                        v_t.numel() * v_t.element_size(), llaisys.MemcpyKind.D2D)
        llaisys.Ops.self_attention(out_r_lai, q_r_lai, k_r_lai, v_r_lai, scale)

        # Pull back into torch.
        rb = torch.zeros((q_len, nh, hd), dtype=_torch_dtype(dtype_name))
        api.memcpy_sync(rb.data_ptr(), out_r_lai.data_ptr(),
                        rb.numel() * rb.element_size(), llaisys.MemcpyKind.D2D)
        ref_pieces.append(rb)

    ref = torch.cat(ref_pieces, dim=0)
    assert check_equal(out, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()

    test_cases = [
        # (request_shapes [(q,kv),...], nh, nkvh, hd, page_size)
        ([(1, 8)],          4, 2, 8,  16),    # decode-style, 1 page
        ([(4, 4)],          4, 2, 8,  16),    # one short prefill in one page
        ([(20, 20)],        4, 2, 8,  16),    # spans 2 pages
        ([(5, 5), (1, 17)], 4, 2, 8,  16),    # mixed prefill + decode, page boundary on req 2
        ([(7, 33), (1, 9)], 8, 4, 16, 16),    # bigger heads, multi-page request
    ]
    test_dtypes = [
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.self_attention_paged on {args.device}")
    for shapes, nh, nkvh, hd, ps in test_cases:
        for dt, atol, rtol in test_dtypes:
            test_paged_attention(shapes, nh, nkvh, hd, ps, dt, atol, rtol, args.device)
    print("\033[92mTest passed!\033[0m\n")
