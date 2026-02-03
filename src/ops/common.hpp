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

} // namespace llaisys::ops::common