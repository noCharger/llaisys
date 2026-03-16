#ifndef OPS_RANDOM_SAMPLE_OP_HPP
#define OPS_RANDOM_SAMPLE_OP_HPP

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
    void random_sample(tensor_t out_token, tensor_t logits, tensor_t workspace, float temp, float top_p, int top_k);
}

#endif // OPS_RANDOM_SAMPLE_OP_HPP
