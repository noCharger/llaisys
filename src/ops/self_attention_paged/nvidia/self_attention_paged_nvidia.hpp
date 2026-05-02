#pragma once

#include "../../../tensor/tensor.hpp"

#include <cstdint>

namespace llaisys::ops::nvidia {

// CUDA path: gather KV from pages into a contiguous tmp, then dispatch the
// existing per-request strided-batched attention kernel.
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

} // namespace llaisys::ops::nvidia
