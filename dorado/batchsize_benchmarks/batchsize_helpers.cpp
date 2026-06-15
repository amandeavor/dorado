#include "batchsize_helpers.h"

#include "SpeedEntry.h"
#include "compiled_timings.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <ranges>
#include <stdexcept>

namespace dorado::batchsize_benchmarks {

namespace {

// Fallback for platforms that don't support std::ranges::to() yet.
#if !defined(_LIBCPP_VERSION)
template <template <typename...> typename Container, std::ranges::range Range>
auto to(Range && range) {
    using T = std::ranges::range_value_t<Range>;
    return Container<T>{range.begin(), range.end()};
}
#endif

}  // namespace

int pick_best_batch_size(std::span<const SpeedEntry> speeds,
                         uint64_t memory_limit,
                         float time_penalty) {
    if (speeds.empty()) {
        throw std::logic_error("Empty span passed to batch size selection");
    } else if (!compiled_cache::is_sorted(speeds)) {
        throw std::runtime_error("Speeds must be sorted");
    }

    // Filter entries to those that fit in memory.
    const auto is_below_memory_limit = [memory_limit](const SpeedEntry & entry) {
        return entry.memory_used <= memory_limit;
    };
    auto entries_below_memory_limit =
            to<std::vector>(speeds | std::views::filter(is_below_memory_limit));
    if (entries_below_memory_limit.empty()) {
        throw std::runtime_error(
                fmt::format("No entries remaining after applying memory_limit ({})", memory_limit));
    }

    // Sort the entries so that we can search through them for the first "best" entry.
    const auto compare_better = [](const SpeedEntry & lhs, const SpeedEntry & rhs) {
        // The order of these represents priority, ie basecall_speed is the most important.
        return std::tie(lhs.basecall_speed, lhs.memory_used, lhs.batch_size) <
               std::tie(rhs.basecall_speed, rhs.memory_used, rhs.batch_size);
    };
    std::ranges::sort(entries_below_memory_limit, compare_better);

    // The fastest speed will be at the back.
    const auto fastest_entry = entries_below_memory_limit.back();

    // Apply the time penalty.
    // TODO: the existing time_penalty code seems odd since 3 is half of 1 is half of 0.
    // TODO: would |speed * (1 - speed_penalty)| be better?
    const double threshold_speed = fastest_entry.basecall_speed / (1.0 + time_penalty);
    const auto under_threshold = [](const SpeedEntry & entry, double speed) {
        return entry.basecall_speed < speed;
    };
    const auto first_over_threshold =
            std::lower_bound(entries_below_memory_limit.begin(), entries_below_memory_limit.end(),
                             threshold_speed, under_threshold);
    if (first_over_threshold == entries_below_memory_limit.end()) {
        // This should be impossible since the fastest speed should be over the threshold.
        throw std::logic_error("Error in batch size selection");
    }

    spdlog::debug("Fastest capped+limited batch size is {} @ {}", first_over_threshold->batch_size,
                  first_over_threshold->basecall_speed);
    return first_over_threshold->batch_size;
}

}  // namespace dorado::batchsize_benchmarks
