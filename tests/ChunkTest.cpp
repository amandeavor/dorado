#include "read_pipeline/base/chunk.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <random>
#include <span>

#define TEST_GROUP "[utils]"

CATCH_TEST_CASE("Test generate_chunks", TEST_GROUP) {
    CATCH_SECTION("Invalid input") {
        auto [num_samples, chunk_size, stride, overlap] =
                GENERATE(table<size_t, size_t, size_t, size_t>({
                        {0, 9996, 6, 498},       // num_samples == 0
                        {12345, 0, 6, 498},      // chunk_size == 0
                        {12345, 9996, 0, 498},   // stride == 0
                        {12345, 9996, 10, 498},  // (chunk_size % stride) != 0
                        {12345, 9996, 7, 498},   // (overlap % stride) != 0
                        {12345, 9996, 6, 9996},  // chunk_size <= overlap
                        {12345, 9996, 6, 9997},  // chunk_size <= overlap
                }));

        CATCH_CAPTURE(num_samples, chunk_size, stride, overlap);

        CATCH_CHECK_THROWS(
                dorado::utils::generate_chunks(num_samples, chunk_size, stride, overlap));
    }

    CATCH_SECTION("Valid input") {
        auto [num_samples, chunk_size, stride, overlap, expected] =
                GENERATE(table<size_t, size_t, size_t, size_t, std::vector<size_t>>({
                        {9996 / 2, 9996, 6, 498, {0}},
                        {9996, 9996, 6, 498, {0}},
                        {9996 + 1, 9996, 6, 498, {0, 6}},
                        {9996 + (9996 / 2), 9996, 6, 498, {0, 4998}},
                        {(2 * 9996) + (9996 / 2), 9996, 1, 0, {0, 9996, 14994}},
                        {3 * 9996, 9996, 6, 498, {0, 9498, 18996, 19992}},
                }));

        CATCH_CAPTURE(num_samples, chunk_size, stride, overlap);

        CATCH_CHECK(dorado::utils::generate_chunks(num_samples, chunk_size, stride, overlap) ==
                    expected);
    }

    CATCH_SECTION("Valid chunks") {
        std::mt19937 generator(42);
        std::uniform_int_distribution<> distribution(1024, 2097152);

        const auto validate_chunks = [&](const std::size_t chunk_size, const std::size_t stride,
                                         const std::size_t overlap) {
            std::vector<std::size_t> reads;
            reads.reserve(16);
            for (int i = 0; i < 16; ++i) {
                reads.emplace_back(distribution(generator));
            }
            for (const std::size_t num_samples : reads) {
                std::vector<std::size_t> offsets;
                CATCH_REQUIRE_NOTHROW(offsets = dorado::utils::generate_chunks(
                                              num_samples, chunk_size, stride, overlap));
                CATCH_REQUIRE_FALSE(std::empty(offsets));

                CATCH_REQUIRE(offsets.front() == 0);

                for (std::size_t i = 1; i < (std::size(offsets) - 1); ++i) {
                    CATCH_REQUIRE((offsets[i] % stride) == 0);
                    CATCH_REQUIRE(offsets[i] == (i * (chunk_size - overlap)));
                }

                CATCH_REQUIRE((offsets.back() % stride) == 0);
                CATCH_REQUIRE(offsets.back() < num_samples);
                if (std::size(offsets) > 1) {
                    CATCH_REQUIRE((num_samples - offsets.back()) >= (chunk_size - stride));
                    CATCH_REQUIRE((num_samples - offsets.back()) <= chunk_size);
                }
            }
        };

        validate_chunks(9996, 6, 498);
        validate_chunks(9996, 7, 497);
        validate_chunks(9996, 12, 492);
        validate_chunks(9996, 17, 510);
        validate_chunks(555, 5, 25);
        validate_chunks(83, 1, 13);
        validate_chunks(123, 1, 0);
    }
}

CATCH_TEST_CASE("Test generate_variable_chunks", TEST_GROUP) {
    using Interval = std::pair<std::size_t, std::size_t>;

    CATCH_SECTION("Invalid input") {
        auto [num_samples, chunk_size, stride, overlap] =
                GENERATE(table<size_t, size_t, size_t, size_t>({
                        {0, 9996, 6, 498},       // num_samples == 0
                        {12345, 0, 6, 498},      // chunk_size == 0
                        {12345, 9996, 0, 498},   // stride == 0
                        {12345, 9996, 10, 498},  // (chunk_size % stride) != 0
                        {12345, 6, 6, 498},      // chunk_size == stride
                        {12345, 9996, 7, 498},   // (overlap % stride) != 0
                        {12345, 9996, 7, 0},     // (stride != 1) && (overlap == 0)
                        {12345, 9996, 6, 9996},  // chunk_size <= overlap
                        {12345, 9996, 6, 9997},  // chunk_size <= overlap
                }));

        CATCH_CAPTURE(num_samples, chunk_size, stride, overlap);

        CATCH_CHECK_THROWS(
                dorado::utils::generate_variable_chunks(num_samples, chunk_size, stride, overlap));
    }

    CATCH_SECTION("Valid input") {
        auto [num_samples, chunk_size, stride, overlap,
              expected] = GENERATE(table<size_t, size_t, size_t, size_t, std::vector<Interval>>({
                {9996 / 2, 9996, 6, 498, {{0, 4998}}},
                {9996, 9996, 6, 498, {{0, 9996}}},
                {9996 + 1, 9996, 6, 498, {{0, 5244}, {4752, 9997}}},
                {9996 + (9996 / 2), 9996, 6, 498, {{0, 7746}, {7248, 14994}}},
                {(2 * 9996) + (9996 / 2), 9996, 1, 0, {{0, 8330}, {8330, 16660}, {16660, 24990}}},
                {3 * 9996,
                 9996,
                 6,
                 498,
                 {{0, 7866}, {7374, 15240}, {14748, 22614}, {22122, 29988}}},
        }));

        CATCH_CAPTURE(num_samples, chunk_size, stride, overlap);

        CATCH_CHECK(dorado::utils::generate_variable_chunks(num_samples, chunk_size, stride,
                                                            overlap) == expected);
    }

    CATCH_SECTION("Valid chunks") {
        std::mt19937 generator(42);
        std::uniform_int_distribution<> distribution(1024, 2097152);

        const auto validate_chunks = [&](const std::size_t chunk_size, const std::size_t stride,
                                         const std::size_t overlap) {
            std::vector<std::size_t> reads;
            reads.reserve(16);
            for (int i = 0; i < 16; ++i) {
                reads.emplace_back(distribution(generator));
            }
            for (const std::size_t num_samples : reads) {
                std::vector<Interval> intervals;
                CATCH_REQUIRE_NOTHROW(intervals = dorado::utils::generate_variable_chunks(
                                              num_samples, chunk_size, stride, overlap));
                CATCH_REQUIRE_FALSE(std::empty(intervals));

                CATCH_REQUIRE(intervals.front().first == 0);
                for (std::size_t i = 1; i < std::size(intervals); ++i) {
                    CATCH_REQUIRE((intervals[i].first % stride) == 0);
                }

                for (std::size_t i = 0; i < std::size(intervals); ++i) {
                    CATCH_REQUIRE((intervals[i].second - intervals[i].first) > 0);
                    CATCH_REQUIRE((intervals[i].second - intervals[i].first) <= chunk_size);
                }
                for (std::size_t i = 1; i < std::size(intervals); ++i) {
                    CATCH_REQUIRE((intervals[i - 1].second - intervals[i].first) <= overlap);
                }

                for (std::size_t i = 0; i < (std::size(intervals) - 1); ++i) {
                    CATCH_REQUIRE((intervals[i].second % stride) == 0);
                }
                CATCH_REQUIRE(intervals.back().second == num_samples);
            }
        };

        validate_chunks(9996, 6, 498);
        validate_chunks(9996, 7, 497);
        validate_chunks(9996, 12, 492);
        validate_chunks(9996, 17, 510);
        validate_chunks(555, 5, 25);
        validate_chunks(83, 1, 13);
        validate_chunks(123, 1, 0);
    }
}

CATCH_TEST_CASE("Test generate_variable_chunks_tx", TEST_GROUP) {
    using Interval = std::pair<std::size_t, std::size_t>;

    CATCH_SECTION("Invalid input") {
        auto [num_samples, max_chunk_size, stride, chunk_size_granularity,
              overlap] = GENERATE(table<size_t, size_t, size_t, size_t, size_t>({
                {0, 12288, 12, 768, 498},          // num_samples == 0
                {12345, 0, 12, 768, 498},          // max_chunk_size == 0
                {12345, 12288, 0, 768, 498},       // stride == 0
                {12345, 12288, 12, 0, 498},        // chunk_size_granularity == 0
                {12345, 12288, 12, 768 + 1, 498},  // (chunk_size_granularity % stride) != 0
                {12345, 12288 + 1, 12, 1, 498},    // (max_chunk_size % stride) != 0
                {12345, 12300, 12, 768, 498},      // (max_chunk_size % chunk_size_granularity) != 0
                {12345, 12288, 12, 768, 12288},    // max_chunk_size <= overlap
                {12345, 12288, 12, 768, 12288 + 1},  // max_chunk_size <= overlap
        }));

        CATCH_CAPTURE(num_samples, max_chunk_size, stride, chunk_size_granularity, overlap);

        CATCH_CHECK_THROWS(dorado::utils::generate_variable_chunks_tx(
                num_samples, max_chunk_size, stride, chunk_size_granularity, overlap));
    }

    CATCH_SECTION("Valid input") {
        auto [num_samples, max_chunk_size, stride, chunk_size_granularity, overlap, expected] =
                GENERATE(table<size_t, size_t, size_t, size_t, size_t, std::vector<Interval>>({
                        {384, 12288, 12, 768, 498, {{0, 384}}},
                        {12288 - 1, 12288, 12, 768, 498, {{0, 12287}}},
                        {12288 + 1, 12288, 12, 768, 498, {{0, 12288}, {12288 - 498, 12288 + 1}}},
                        {12288 + (12288 / 2), 12288, 12, 768, 498, {{0, 12288}, {11790, 18432}}},
                        {(2 * 12288) + (12288 / 2),
                         12288,
                         12,
                         768,
                         0,
                         {{0, 12288}, {12288, 24576}, {24576, 30720}}},
                        {3 * 12288,
                         12288,
                         12,
                         768,
                         498,
                         {{0, 12288}, {11790, 24078}, {23580, 35868}, {35370, 36864}}},
                }));

        CATCH_CAPTURE(num_samples, max_chunk_size, stride, chunk_size_granularity, overlap);

        CATCH_CHECK(dorado::utils::generate_variable_chunks_tx(num_samples, max_chunk_size, stride,
                                                               chunk_size_granularity,
                                                               overlap) == expected);
    }

    CATCH_SECTION("Valid chunks") {
        std::mt19937 generator(42);
        std::uniform_int_distribution<> distribution(60, 2097152);

        const auto validate_chunks = [&](const std::size_t max_chunk_size, const std::size_t stride,
                                         const std::size_t chunk_size_granularity,
                                         const std::size_t overlap) {
            std::vector<std::size_t> reads;
            reads.reserve(16);
            for (int i = 0; i < 16; ++i) {
                reads.emplace_back(distribution(generator));
            }

            for (const std::size_t num_samples : reads) {
                std::vector<Interval> intervals;
                CATCH_REQUIRE_NOTHROW(intervals = dorado::utils::generate_variable_chunks_tx(
                                              num_samples, max_chunk_size, stride,
                                              chunk_size_granularity, overlap));

                CATCH_REQUIRE_FALSE(std::empty(intervals));
                CATCH_REQUIRE(intervals.front().first == 0);
                CATCH_REQUIRE(intervals.back().second == num_samples);

                for (std::size_t i = 0; i < std::size(intervals); ++i) {
                    const auto [start, end] = intervals[i];

                    CATCH_REQUIRE(start < end);
                    CATCH_REQUIRE(end <= num_samples);
                    CATCH_REQUIRE((end - start) <= max_chunk_size);

                    if (i + 1 < std::size(intervals)) {
                        CATCH_REQUIRE((end - start) == max_chunk_size);
                        CATCH_REQUIRE(((end - start) % chunk_size_granularity) == 0);
                        CATCH_REQUIRE((end % stride) == 0);
                    }
                }

                for (std::size_t i = 1; i < std::size(intervals); ++i) {
                    CATCH_REQUIRE(intervals[i].first ==
                                  intervals[i - 1].first + (max_chunk_size - overlap));

                    CATCH_REQUIRE(intervals[i].first <= intervals[i - 1].second);
                    CATCH_REQUIRE((intervals[i - 1].second - intervals[i].first) == overlap);
                }
            }
        };

        validate_chunks(9996, 6, 12, 498);
        validate_chunks(9996, 7, 7, 497);
        validate_chunks(9996, 12, 12, 492);
        validate_chunks(9996, 17, 17, 510);
        validate_chunks(555, 5, 15, 25);
        validate_chunks(83, 1, 1, 13);
        validate_chunks(123, 1, 3, 0);
        validate_chunks(1920, 12, 192, 600);
    }
}