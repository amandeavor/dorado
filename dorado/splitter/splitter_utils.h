#pragma once

#include "splitter/ReadSplitter.h"

#include <ATen/core/TensorBody.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

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
                                   uint64_t ignore_spikes_threshold) {
    SampleRanges<T> clusters;

    const auto pore_a = signal.accessor<T, 1>();
    const int64_t pore_a_size = pore_a.size(0);

    const auto over_threshold = [threshold](const T& val) {
    // ARM64 has native _Half support but x64 doesn't and has to convert a c10::Half
    // to a single precision float to compare them, so use a fast comparison since we
    // likely won't need the single precision float afterwards.
#if !defined(__aarch64__)
        if constexpr (std::is_same_v<T, c10::Half>) {
            return detail::fast_half_comparable(val) > detail::fast_half_comparable(threshold);
        }
#endif
        return val > threshold;
    };

    int64_t idx = ignore_prefix;
    while (idx < pore_a_size) {
        int64_t cl_start = -1;

        // Most of the time is spent looking for the start of a peak, so make that hot loop tight.
        for (; idx < pore_a_size; idx++) {
            const T sample = pore_a[idx];
            if (over_threshold(sample)) {
                cl_start = idx;
                break;
            }
        }
        if (cl_start == -1) {
            // No peak found.
            break;
        }

        // Read until end of the peak.
        T cl_max = std::numeric_limits<T>::min();
        int64_t cl_argmax = -1;
        for (; idx < pore_a_size; idx++) {
            const T sample = pore_a[idx];
            if (!over_threshold(sample)) {
                break;
            }
            if (sample >= cl_max) {
                cl_max = sample;
                cl_argmax = idx;
            }
        }
        const int64_t cl_end = idx;

        // report cluster
        assert(cl_start < pore_a_size && cl_end <= pore_a_size);
        clusters.push_back(SampleRange(cl_start, cl_end, cl_argmax, cl_max));
    }

    // merge clusters
    SampleRanges<T> merged_clusters;
    for (auto&& cluster : clusters) {
        if (cluster.end_sample - cluster.start_sample < ignore_spikes_threshold) {
            // discard spurious clusters
            continue;
        }

        if (merged_clusters.empty()) {
            merged_clusters.push_back(std::move(cluster));
            continue;
        }

        auto& last_cluster = merged_clusters.back();
        if (cluster.start_sample - last_cluster.end_sample > cluster_dist) {
            // new cluster is too far away to merge, just accept it
            merged_clusters.push_back(std::move(cluster));
        } else {
            // extend previous cluster and update metadata
            last_cluster.end_sample = cluster.end_sample;
            if (cluster.max_val >= last_cluster.max_val) {
                last_cluster.max_val = cluster.max_val;
                last_cluster.argmax_sample = cluster.argmax_sample;
            }
        }
    }

    return merged_clusters;
}

}  // namespace dorado::splitter
