#pragma once

#include "../../../tensor/tensor.hpp"

#include <cstdint>
#include <vector>

namespace llaisys::ops::nvidia {

// CUDA path: per-request loop reusing a single scratch buffer; each request
// runs cublasGemmStridedBatchedEx over heads (same as single-sequence kernel).
void self_attention_varlen(tensor_t attn_val,
                           tensor_t q,
                           const std::vector<tensor_t> &k_blocks,
                           const std::vector<tensor_t> &v_blocks,
                           const int32_t *cu_seqlens_q,
                           int32_t batch,
                           float scale);

} // namespace llaisys::ops::nvidia
