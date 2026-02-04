#pragma once

namespace llaisys::ops::common {

template <typename T1, typename T2>
inline bool allContiguous(const T1& t1, const T2& t2) {
    return t1->isContiguous() && t2->isContiguous();
}

template <typename T1, typename T2, typename T3>
inline bool allContiguous(const T1& t1, const T2& t2, const T3& t3) {
    return t1->isContiguous() && t2->isContiguous() && t3->isContiguous();
}

// Shared conversion helpers to reduce duplication across ops
template <typename T>
inline float to_float(T v) {
    return static_cast<float>(v);
}

template <>
inline float to_float<float>(float v) { return v; }

template <>
inline float to_float<llaisys::fp16_t>(llaisys::fp16_t v) {
    return llaisys::utils::_f16_to_f32(v);
}

template <>
inline float to_float<llaisys::bf16_t>(llaisys::bf16_t v) {
    return llaisys::utils::_bf16_to_f32(v);
}

template <typename T>
inline T from_float(float v) {
    return static_cast<T>(v);
}

template <>
inline float from_float<float>(float v) { return v; }

template <>
inline llaisys::fp16_t from_float<llaisys::fp16_t>(float v) {
    return llaisys::utils::_f32_to_f16(v);
}

template <>
inline llaisys::bf16_t from_float<llaisys::bf16_t>(float v) {
    return llaisys::utils::_f32_to_bf16(v);
}

} // namespace llaisys::ops::common