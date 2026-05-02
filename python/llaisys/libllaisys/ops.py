from .tensor import llaisysTensor_t
from ctypes import c_float, c_int, c_int32, POINTER

def load_ops(lib):
    lib.llaisysAdd.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysAdd.restype = None

    lib.llaisysArgmax.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysArgmax.restype = None

    lib.llaisysEmbedding.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysEmbedding.restype = None

    lib.llaisysLinear.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysLinear.restype = None

    lib.llaisysRearrange.argtypes = [llaisysTensor_t, llaisysTensor_t]
    lib.llaisysRearrange.restype = None

    lib.llaisysRmsNorm.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysRmsNorm.restype = None

    lib.llaisysROPE.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysROPE.restype = None

    lib.llaisysSelfAttention.argtypes = [
        llaisysTensor_t,  # attn_val
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k
        llaisysTensor_t,  # v
        c_float    # scale
    ]
    lib.llaisysSelfAttention.restype = None

    lib.llaisysSelfAttentionVarlen.argtypes = [
        llaisysTensor_t,            # attn_val (packed [total_q, nh, dh])
        llaisysTensor_t,            # q       (packed [total_q, nh, dh])
        POINTER(llaisysTensor_t),   # k_blocks: array of batch tensors
        POINTER(llaisysTensor_t),   # v_blocks: array of batch tensors
        POINTER(c_int32),           # cu_seqlens_q: int32[batch+1]
        c_int32,                    # batch
        c_float,                    # scale
    ]
    lib.llaisysSelfAttentionVarlen.restype = None

    lib.llaisysSelfAttentionPaged.argtypes = [
        llaisysTensor_t,            # attn_val [total_q, nh, dh]
        llaisysTensor_t,            # q
        llaisysTensor_t,            # big_k [n_pages*page_size, nkvh, dh]
        llaisysTensor_t,            # big_v
        POINTER(c_int32),           # block_tables (flat, sum of npages_r)
        POINTER(c_int32),           # block_table_lens [batch]
        POINTER(c_int32),           # cu_seqlens_q [batch+1]
        POINTER(c_int32),           # kv_lens [batch]
        c_int32,                    # batch
        c_int32,                    # page_size
        c_float,                    # scale
    ]
    lib.llaisysSelfAttentionPaged.restype = None

    lib.llaisysSwiGLU.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysSwiGLU.restype = None

    lib.llaisysRandomSample.argtypes = [
        llaisysTensor_t, # out_token
        llaisysTensor_t, # logits
        c_float, # temp
        c_float, # top_p
        c_int    # top_k
    ]
    lib.llaisysRandomSample.restype = None

    lib.llaisysRandomSampleBatch.argtypes = [
        llaisysTensor_t,    # out_tokens [N] i64
        llaisysTensor_t,    # logits [N, voc]
        POINTER(c_float),   # temps [N]
        POINTER(c_float),   # top_ps [N]
        POINTER(c_int),     # top_ks [N]
    ]
    lib.llaisysRandomSampleBatch.restype = None

    lib.llaisysArgmaxBatch.argtypes = [
        llaisysTensor_t,    # out_indices [N] i64
        llaisysTensor_t,    # logits [N, voc]
    ]
    lib.llaisysArgmaxBatch.restype = None
