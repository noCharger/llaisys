#pragma once
#include "llaisys/tensor.h"

namespace llaisys::ops::nvidia {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
}
