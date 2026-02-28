#pragma once
#include "llaisys/tensor.h"

namespace llaisys::ops::nvidia {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
}
