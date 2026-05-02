from .libllaisys import LIB_LLAISYS
from .libllaisys.tensor import llaisysTensor_t
from .tensor import Tensor
from ctypes import c_float, c_int, c_int32, POINTER


class Ops:
    @staticmethod
    def add(c: Tensor, a: Tensor, b: Tensor):
        LIB_LLAISYS.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())

    @staticmethod
    def argmax(max_idx: Tensor, max_val: Tensor, vals: Tensor):
        LIB_LLAISYS.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())

    @staticmethod
    def embedding(out: Tensor, index: Tensor, weight: Tensor):
        LIB_LLAISYS.llaisysEmbedding(
            out.lib_tensor(), index.lib_tensor(), weight.lib_tensor()
        )

    @staticmethod
    def linear(out: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
        LIB_LLAISYS.llaisysLinear(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), bias.lib_tensor()
        )

    @staticmethod
    def rearrange(out: Tensor, inp: Tensor):
        LIB_LLAISYS.llaisysRearrange(out.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def rms_norm(out: Tensor, inp: Tensor, weight: Tensor, eps: float):
        LIB_LLAISYS.llaisysRmsNorm(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
        )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        LIB_LLAISYS.llaisysROPE(
            out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
        )

    @staticmethod
    def self_attention(attn_val: Tensor, q: Tensor, k: Tensor, v: Tensor, scale: float):
        LIB_LLAISYS.llaisysSelfAttention(
            attn_val.lib_tensor(),
            q.lib_tensor(),
            k.lib_tensor(),
            v.lib_tensor(),
            c_float(scale),
        )

    @staticmethod
    def self_attention_varlen(
        attn_val: Tensor,
        q: Tensor,
        k_blocks: list,
        v_blocks: list,
        cu_seqlens_q: list,
        scale: float,
    ):
        batch = len(k_blocks)
        assert len(v_blocks) == batch, "k_blocks and v_blocks must be same length"
        assert len(cu_seqlens_q) == batch + 1, "cu_seqlens_q must be batch+1"

        k_arr = (llaisysTensor_t * batch)(*[t.lib_tensor() for t in k_blocks])
        v_arr = (llaisysTensor_t * batch)(*[t.lib_tensor() for t in v_blocks])
        cu_arr = (c_int32 * (batch + 1))(*cu_seqlens_q)

        LIB_LLAISYS.llaisysSelfAttentionVarlen(
            attn_val.lib_tensor(),
            q.lib_tensor(),
            k_arr,
            v_arr,
            cu_arr,
            c_int32(batch),
            c_float(scale),
        )

    @staticmethod
    def self_attention_paged(
        attn_val: Tensor,
        q: Tensor,
        big_k: Tensor,
        big_v: Tensor,
        block_tables: list,           # flat list of page ids per request, concatenated
        block_table_lens: list,       # [batch]
        cu_seqlens_q: list,           # [batch+1]
        kv_lens: list,                # [batch]
        page_size: int,
        scale: float,
    ):
        batch = len(block_table_lens)
        assert len(cu_seqlens_q) == batch + 1
        assert len(kv_lens) == batch
        assert len(block_tables) == sum(block_table_lens)

        bt_arr = (c_int32 * len(block_tables))(*block_tables)
        btl_arr = (c_int32 * batch)(*block_table_lens)
        cu_arr = (c_int32 * (batch + 1))(*cu_seqlens_q)
        kv_arr = (c_int32 * batch)(*kv_lens)

        LIB_LLAISYS.llaisysSelfAttentionPaged(
            attn_val.lib_tensor(),
            q.lib_tensor(),
            big_k.lib_tensor(),
            big_v.lib_tensor(),
            bt_arr, btl_arr, cu_arr, kv_arr,
            c_int32(batch), c_int32(page_size), c_float(scale),
        )

    @staticmethod
    def swiglu(out: Tensor, gate: Tensor, up: Tensor):
        LIB_LLAISYS.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())

    @staticmethod
    def random_sample(out_token: Tensor, logits: Tensor, temp: float, top_p: float, top_k: int):
        LIB_LLAISYS.llaisysRandomSample(
            out_token.lib_tensor(),
            logits.lib_tensor(),
            c_float(temp),
            c_float(top_p),
            c_int(top_k)
        )

    @staticmethod
    def random_sample_batch(
        out_tokens: Tensor,
        logits: Tensor,
        temps: list,
        top_ps: list,
        top_ks: list,
    ):
        n = len(temps)
        assert len(top_ps) == n and len(top_ks) == n, "param arrays must be same length"
        temp_arr = (c_float * n)(*temps)
        top_p_arr = (c_float * n)(*top_ps)
        top_k_arr = (c_int * n)(*top_ks)
        LIB_LLAISYS.llaisysRandomSampleBatch(
            out_tokens.lib_tensor(),
            logits.lib_tensor(),
            temp_arr,
            top_p_arr,
            top_k_arr,
        )

    @staticmethod
    def argmax_batch(out_indices: Tensor, logits: Tensor):
        LIB_LLAISYS.llaisysArgmaxBatch(
            out_indices.lib_tensor(),
            logits.lib_tensor(),
        )
