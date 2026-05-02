import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from test_utils import (
    random_tensor,
    llaisys_dtype,
    llaisys_device,
    zero_tensor,
)


def test_argmax_batch(n, voc, dtype_name, device_name):
    print(f"   argmax_batch n={n} voc={voc} dtype<{dtype_name}>")
    logits_t, logits_l = random_tensor((n, voc), dtype_name, device_name)

    # Reference: per-row argmax via torch
    ref = torch.argmax(logits_t, dim=-1).to(torch.int64)

    out = llaisys.Tensor((n,), dtype=llaisys.DataType.I64,
                         device=llaisys_device(device_name))
    llaisys.Ops.argmax_batch(out, logits_l)

    # Copy result back
    got = torch.empty((n,), dtype=torch.int64,
                      device=ref.device)
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        got.data_ptr(),
        out.data_ptr(),
        n * 8,
        llaisys.MemcpyKind.D2D,
    )
    assert torch.equal(got, ref), f"argmax_batch mismatch: got={got} ref={ref}"


def test_random_sample_batch_smoke(n, voc, device_name):
    """Smoke test: random_sample_batch returns valid token ids per row.
    We can't compare exact values because of RNG, but we check:
      - all results are in [0, voc)
      - results match per-row single random_sample calls when seeded the same way
        (impossible because the C++ side reseeds, so just check bounds)
    """
    print(f"   random_sample_batch n={n} voc={voc} (smoke)")
    logits_t, logits_l = random_tensor((n, voc), "f32", device_name)
    out = llaisys.Tensor((n,), dtype=llaisys.DataType.I64,
                         device=llaisys_device(device_name))

    temps = [1.0] * n
    top_ps = [0.9] * n
    top_ks = [50] * n

    llaisys.Ops.random_sample_batch(out, logits_l, temps, top_ps, top_ks)

    got = torch.empty((n,), dtype=torch.int64)
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        got.data_ptr(),
        out.data_ptr(),
        n * 8,
        llaisys.MemcpyKind.D2D,
    )
    assert ((got >= 0) & (got < voc)).all(), f"out of range: {got}"


def test_random_sample_batch_argmax_equiv(n, voc, device_name):
    """When top_k=1, random_sample_batch must match argmax_batch."""
    print(f"   random_sample_batch top_k=1 vs argmax n={n} voc={voc}")
    logits_t, logits_l = random_tensor((n, voc), "f32", device_name)

    sampled = llaisys.Tensor((n,), dtype=llaisys.DataType.I64,
                             device=llaisys_device(device_name))
    llaisys.Ops.random_sample_batch(
        sampled, logits_l, [1.0]*n, [1.0]*n, [1]*n,
    )

    argmaxed = llaisys.Tensor((n,), dtype=llaisys.DataType.I64,
                              device=llaisys_device(device_name))
    llaisys.Ops.argmax_batch(argmaxed, logits_l)

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    s = torch.empty((n,), dtype=torch.int64); a = torch.empty((n,), dtype=torch.int64)
    api.memcpy_sync(s.data_ptr(), sampled.data_ptr(), n*8, llaisys.MemcpyKind.D2D)
    api.memcpy_sync(a.data_ptr(), argmaxed.data_ptr(), n*8, llaisys.MemcpyKind.D2D)
    assert torch.equal(s, a), f"top_k=1 mismatch: sampled={s} argmaxed={a}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()

    print(f"Testing batched samplers on {args.device}")

    for dtype_name in ["f32", "f16", "bf16"]:
        for n, voc in [(1, 32), (4, 1024), (8, 50257)]:
            test_argmax_batch(n, voc, dtype_name, args.device)

    for n, voc in [(1, 32), (4, 1024), (8, 50257)]:
        test_random_sample_batch_smoke(n, voc, args.device)
        test_random_sample_batch_argmax_equiv(n, voc, args.device)

    print("\033[92mTest passed!\033[0m\n")
