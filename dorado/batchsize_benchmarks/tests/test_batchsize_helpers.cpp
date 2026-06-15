#include "SpeedEntry.h"
#include "batchsize_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <limits>
#include <stdexcept>

#define CUT_TAG "[batchsize_benchmarks]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace dorado::batchsize_benchmarks;

namespace {

constexpr uint64_t NO_MEMORY_LIMIT = std::numeric_limits<uint64_t>::max();
constexpr float NO_TIME_PENALTY = 0;
// Make a time_penalty that represents a factor of the best speed, ie a penalty of 0.25 will look for >=75% best speed.
double penalty_factor(double scale) { return 1.0 / (1.0 - scale) - 1.0; }

DEFINE_TEST("Invalid input") {
    // Can't select a best input from nothing.
    std::span<const SpeedEntry> empty;
    CATCH_CHECK_THROWS_AS(pick_best_batch_size(empty, NO_MEMORY_LIMIT, NO_TIME_PENALTY),
                          std::logic_error);
}

DEFINE_TEST("Basic test") {
    const SpeedEntry entries[]{
            {.batch_size = 1, .basecall_speed = 2, .memory_used = 3},
            {.batch_size = 4, .basecall_speed = 5, .memory_used = 6},
            {.batch_size = 7, .basecall_speed = 8, .memory_used = 9},
    };
    const std::span span{entries};

    CATCH_CHECK(pick_best_batch_size(span.subspan<0, 1>(), NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 1);
    CATCH_CHECK(pick_best_batch_size(span.subspan<0, 2>(), NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 4);
    CATCH_CHECK(pick_best_batch_size(span.subspan<0, 3>(), NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 7);
    CATCH_CHECK(pick_best_batch_size(span.subspan<1, 1>(), NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 4);
    CATCH_CHECK(pick_best_batch_size(span.subspan<1, 2>(), NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 7);
    CATCH_CHECK(pick_best_batch_size(span.subspan<2, 1>(), NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 7);
}

DEFINE_TEST("Entries with the same speed picks smallest memory usage") {
    const SpeedEntry entries[]{
            {.batch_size = 1, .basecall_speed = 5, .memory_used = 10},
            {.batch_size = 2, .basecall_speed = 5, .memory_used = 9},
            {.batch_size = 3, .basecall_speed = 5, .memory_used = 11},
    };

    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 2);
}

DEFINE_TEST("memory_limit is enforced") {
    const SpeedEntry entries[]{
            {.batch_size = 1, .basecall_speed = 10, .memory_used = 100},
            {.batch_size = 2, .basecall_speed = 20, .memory_used = 200},
            {.batch_size = 4, .basecall_speed = 40, .memory_used = 400},
    };

    CATCH_CHECK(pick_best_batch_size(entries, 500, NO_TIME_PENALTY) == 4);

    CATCH_CHECK(pick_best_batch_size(entries, 401, NO_TIME_PENALTY) == 4);
    CATCH_CHECK(pick_best_batch_size(entries, 400, NO_TIME_PENALTY) == 4);
    CATCH_CHECK(pick_best_batch_size(entries, 399, NO_TIME_PENALTY) == 2);

    CATCH_CHECK(pick_best_batch_size(entries, 201, NO_TIME_PENALTY) == 2);
    CATCH_CHECK(pick_best_batch_size(entries, 200, NO_TIME_PENALTY) == 2);
    CATCH_CHECK(pick_best_batch_size(entries, 199, NO_TIME_PENALTY) == 1);

    CATCH_CHECK(pick_best_batch_size(entries, 101, NO_TIME_PENALTY) == 1);
    CATCH_CHECK(pick_best_batch_size(entries, 100, NO_TIME_PENALTY) == 1);
    CATCH_CHECK_THROWS_WITH(pick_best_batch_size(entries, 10, NO_TIME_PENALTY),
                            "No entries remaining after applying memory_limit (10)");
}

DEFINE_TEST("time_penalty is enforced") {
    const SpeedEntry entries[]{
            {.batch_size = 1, .basecall_speed = 10, .memory_used = 11},  //  25% of best (1 - 0.75)
            {.batch_size = 2, .basecall_speed = 20, .memory_used = 12},  //  50% of best (1 - 0.50)
            {.batch_size = 3, .basecall_speed = 40, .memory_used = 13},  // 100% of best (1 - 0.00)
    };

    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0)) == 3);
    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.01)) == 3);

    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.49)) == 3);
    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.50)) == 2);
    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.51)) == 2);

    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.74)) == 2);
    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.75)) == 1);
    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.76)) == 1);

    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, penalty_factor(0.99)) == 1);
}

DEFINE_TEST("memory_limit and time_penalty is enforced") {
    const SpeedEntry entries[]{
            {.batch_size = 1, .basecall_speed = 10, .memory_used = 100},  // 50% of batch_size=2
            {.batch_size = 2, .basecall_speed = 20, .memory_used = 200},  // first under 300 memory
            {.batch_size = 3, .basecall_speed = 40, .memory_used = 400},
    };

    CATCH_CHECK(pick_best_batch_size(entries, 300, penalty_factor(0)) == 2);
    // Penalty is applied after the memory limit is applied, not in total.
    CATCH_CHECK(pick_best_batch_size(entries, 300, penalty_factor(0.5)) == 1);
}

DEFINE_TEST("time_penalty picks first entry over penalty, by batch size not speed") {
    constexpr float time_penalty = 0.32f;

    // Seen in real world data with fastest=854:
    // 854 / (1 + 0.32) = 647
    // So first batch size over is 709, but first speed over is 648.
    const SpeedEntry entries[]{
            {.batch_size = 10, .basecall_speed = 643, .memory_used = 102},
            {.batch_size = 20, .basecall_speed = 599, .memory_used = 106},
            {.batch_size = 30, .basecall_speed = 709, .memory_used = 110},  // first, by batch size
            {.batch_size = 40, .basecall_speed = 642, .memory_used = 114},
            {.batch_size = 50, .basecall_speed = 741, .memory_used = 117},
            {.batch_size = 60, .basecall_speed = 687, .memory_used = 122},
            {.batch_size = 70, .basecall_speed = 769, .memory_used = 125},
            {.batch_size = 80, .basecall_speed = 689, .memory_used = 130},
            {.batch_size = 90, .basecall_speed = 793, .memory_used = 133},
            {.batch_size = 100, .basecall_speed = 753, .memory_used = 137},
            {.batch_size = 110, .basecall_speed = 818, .memory_used = 141},
            {.batch_size = 120, .basecall_speed = 656, .memory_used = 156},
            {.batch_size = 130, .basecall_speed = 648, .memory_used = 161},  // first, by speed
            {.batch_size = 140, .basecall_speed = 663, .memory_used = 164},
            {.batch_size = 150, .basecall_speed = 678, .memory_used = 168},
            {.batch_size = 160, .basecall_speed = 854, .memory_used = 281},  // best, no penalty
    };

    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, NO_TIME_PENALTY) == 160);
    CATCH_CHECK(pick_best_batch_size(entries, NO_MEMORY_LIMIT, time_penalty) == 30);
}

}  // namespace
