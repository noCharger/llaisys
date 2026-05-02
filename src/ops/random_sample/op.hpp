#ifndef OPS_RANDOM_SAMPLE_OP_HPP
#define OPS_RANDOM_SAMPLE_OP_HPP

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    void random_sample(tensor_t out_token, tensor_t logits, float temp, float top_p, int top_k);

    // Per-row sampler: out_tokens [N] i64, logits [N, voc].
    // temps/top_ps/top_ks are length-N host arrays of per-request params.
    void random_sample_batch(tensor_t out_tokens,
                             tensor_t logits,
                             const float *temps,
                             const float *top_ps,
                             const int *top_ks);
}

#endif // OPS_RANDOM_SAMPLE_OP_HPP
