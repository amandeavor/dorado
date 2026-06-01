#include "read_pipeline/base/chunk.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dorado::utils {

std::vector<std::size_t> generate_chunks(const std::size_t num_samples,
                                         const std::size_t chunk_size,
                                         const std::size_t stride,
                                         const std::size_t overlap) {
    if (num_samples == 0) {
        throw std::runtime_error("utils::generate_chunks: empty read");
    }
    if (stride == 0) {
        throw std::logic_error("utils::generate_chunks: invalid stride " + std::to_string(stride));
    }
    if ((chunk_size == 0) || ((chunk_size % stride) != 0) || (chunk_size <= overlap)) {
        throw std::logic_error("utils::generate_chunks: invalid chunk size " +
                               std::to_string(chunk_size) + " with overlap " +
                               std::to_string(overlap) + " and stride " + std::to_string(stride));
    }
    if ((overlap % stride) != 0) {
        throw std::logic_error("utils::generate_chunks: invalid overlap " +
                               std::to_string(overlap) + " with stride " + std::to_string(stride));
    }

    std::vector<std::size_t> offsets;
    offsets.emplace_back(0);

    std::size_t offset = 0;
    std::size_t last_offset = (num_samples > chunk_size) ? (num_samples - chunk_size) : 0;
    if (const std::size_t misalignment = last_offset % stride; misalignment != 0) {
        // Move last chunk start to the next stride boundary, we'll zero pad any excess samples required.
        last_offset += stride - misalignment;
    }
    const std::size_t chunk_step = chunk_size - overlap;
    while ((offset + chunk_size) < num_samples) {
        offset = std::min(offset + chunk_step, last_offset);
        offsets.emplace_back(offset);
    }

    return offsets;
}

std::vector<std::pair<std::size_t, std::size_t>> generate_variable_chunks(
        const std::size_t num_samples,
        const std::size_t chunk_size,
        const std::size_t stride,
        const std::size_t overlap) {
    if (num_samples == 0) {
        throw std::runtime_error("utils::generate_variable_chunks: empty read");
    }
    if (stride == 0) {
        throw std::logic_error("utils::generate_variable_chunks: invalid stride " +
                               std::to_string(stride));
    }
    if ((chunk_size == 0) || ((chunk_size % stride) != 0) || (chunk_size == stride) ||
        (chunk_size <= overlap)) {
        throw std::logic_error("utils::generate_variable_chunks: invalid chunk size " +
                               std::to_string(chunk_size) + " with overlap " +
                               std::to_string(overlap) + " and stride " + std::to_string(stride));
    }
    if (((overlap % stride) != 0) || ((stride != 1) && (overlap == 0))) {
        throw std::logic_error("utils::generate_variable_chunks: invalid overlap " +
                               std::to_string(overlap) + " with stride " + std::to_string(stride));
    }

    const std::size_t num_chunks =
            1 + ((num_samples > chunk_size) ? std::ceil((num_samples - chunk_size) /
                                                        static_cast<double>(chunk_size - overlap))
                                            : 0);

    const std::size_t num_samples_with_overlaps = num_samples + ((num_chunks - 1) * overlap);
    const std::size_t num_longer_chunks = num_samples_with_overlaps % num_chunks;
    const std::size_t adjusted_chunk_size = num_samples_with_overlaps / num_chunks;

    assert(adjusted_chunk_size <= chunk_size);
    assert(num_samples_with_overlaps == ((num_longer_chunks * (adjusted_chunk_size + 1)) +
                                         ((num_chunks - num_longer_chunks) * adjusted_chunk_size)));

    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    for (std::size_t i = 0, chunk_start = 0; i < num_chunks; ++i) {
        intervals.emplace_back(chunk_start,
                               chunk_start + adjusted_chunk_size + (i < num_longer_chunks));
        chunk_start = intervals.back().second - overlap;
    }

    assert(intervals.back().second == num_samples);

    // Adjust interval positions wrt stride.
    for (std::size_t i = 1; i < num_chunks; ++i) {
        if (const std::size_t misalignment = intervals[i].first % stride; misalignment != 0) {
            intervals[i].first += stride - misalignment;
        }
    }
    for (std::size_t i = 0; i < (num_chunks - 1); ++i) {
        if (const std::size_t misalignment = intervals[i].second % stride; misalignment != 0) {
            intervals[i].second -= misalignment;
        }
    }

    return intervals;
}

std::vector<std::pair<std::size_t, std::size_t>> generate_variable_chunks_tx(const std::size_t num_samples,
                                                                             const std::size_t max_chunk_size,
                                                                             const std::size_t stride,
                                                                             const std::size_t chunk_size_granularity,
                                                                             const std::size_t overlap) {
    if (num_samples == 0) {
        throw std::runtime_error("utils::generate_chunks: empty read");
    }
    if (stride == 0) {
        throw std::logic_error("utils::generate_chunks: invalid stride " + std::to_string(stride));
    }
    if ((chunk_size_granularity == 0) || ((chunk_size_granularity % stride) != 0)) {
        throw std::logic_error("utils::generate_chunks: invalid chunk_size_granularity " + std::to_string(chunk_size_granularity)
                             + " with stride " + std::to_string(stride));
    }
    if ((max_chunk_size == 0) || ((max_chunk_size % stride) != 0) || (max_chunk_size <= overlap) ||
        ((max_chunk_size % chunk_size_granularity) != 0)) {
        throw std::logic_error("utils::generate_chunks: invalid chunk size " +
                               std::to_string(max_chunk_size) + " with overlap " +
                               std::to_string(overlap) + " and stride " + std::to_string(stride) +
                               " and chunk_size_granularity " + std::to_string(chunk_size_granularity));
    }

    std::vector<std::pair<std::size_t, std::size_t>> intervals;
    std::size_t offset = 0;
    const std::size_t chunk_step = max_chunk_size - overlap;

    while ((offset + max_chunk_size) < num_samples) {
        intervals.emplace_back(offset,
                               offset + max_chunk_size);
        offset += chunk_step;
    }

    // This will get padded to batch_size_granularity in BasecallerNode::basecall_worker_thread
    if (offset != num_samples) {
        intervals.emplace_back(offset, num_samples);
    }

    return intervals;
}

}  // namespace dorado::utils