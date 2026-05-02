#pragma once

#include "../../tensor/tensor.hpp"

#include <cstdint>

namespace llaisys::ops {

// Paged variable-length self-attention.
//
// big_k/big_v are the pool's [n_pages*page_size, nkvh, dh] buffers; each
// request's KV is gathered via its block_table (a slice of the flat int32
// `block_tables` array of total length `sum(block_table_lens)`).
// kv_lens[r] is total valid tokens (last page may be partial).
// Causal mask: query i within request r attends to KV [0, kv_lens[r] - q_len_r + i].
void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t big_k,
                          tensor_t big_v,
                          const int32_t *block_tables,
                          const int32_t *block_table_lens,
                          const int32_t *cu_seqlens_q,
                          const int32_t *kv_lens,
                          int32_t batch,
                          int32_t page_size,
                          float scale);

} // namespace llaisys::ops
