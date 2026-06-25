#pragma once

#include <cstdint>
#include <tuple>

namespace dorado::secondary {

template <typename T>
struct IntervalGeneric {
    T start{};
    T end{};

    T length() const { return end - start; }
};

template <typename T>
bool operator==(const IntervalGeneric<T>& a, const IntervalGeneric<T>& b) {
    return (a.start == b.start) && (a.end == b.end);
}

template <typename T>
bool operator<(const IntervalGeneric<T>& a, const IntervalGeneric<T>& b) {
    return std::tie(a.start, a.end) < std::tie(b.start, b.end);
}

using Interval = IntervalGeneric<int32_t>;
using Interval64 = IntervalGeneric<int64_t>;

}  // namespace dorado::secondary
