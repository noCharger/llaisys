#pragma once
#include "tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
}
