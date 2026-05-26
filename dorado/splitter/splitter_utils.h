#pragma once

#include "splitter/ReadSplitter.h"

#include <ATen/core/TensorBody.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

namespace dorado::splitter {

template <class FilterF>
auto filter_ranges(const PosRanges& ranges, FilterF filter_f) {
    PosRanges filtered;
    std::copy_if(ranges.begin(), ranges.end(), std::back_inserter(filtered), filter_f);
    return filtered;
}

//merges overlapping ranges and ranges separated by merge_dist or less
//ranges supposed to be sorted by start coordinate
PosRanges merge_ranges(const PosRanges& ranges, uint64_t merge_dist);

SimplexReadPtr subread(const SimplexRead& read,
                       std::optional<PosRange> seq_range,
                       PosRange signal_range);

namespace detail {
// Convert a _Half into signed value that has the same ordering (for fast comparisons).
inline constexpr int16_t fast_half_comparable(const c10::Half& val) {
    // Decompose into parts.
    const bool s = val.x & 0x8000;
    const int16_t em = val.x & 0x7FFF;
    // Rebuild as an int16_t.
    return s ? -em : em;
}
}  // namespace detail

template <typename T>
SampleRanges<T> detect_pore_signal(const at::Tensor& signal,
                                   T threshold,
                                   uint64_t cluster_dist,
                                   uint64_t ignore_prefix,
                                   uint64_t ignore_spikes_threshold);

}  // namespace dorado::splitter
