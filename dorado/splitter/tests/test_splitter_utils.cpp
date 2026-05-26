#include "splitter_utils.h"

#include <ATen/ops/from_blob.h>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <limits>
#include <random>
#include <set>

#define CUT_TAG "[splitter_utils]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)
#define DEFINE_TEMPLATE_TEST(name, ...) \
    CATCH_TEMPLATE_TEST_CASE(CUT_TAG " " name, CUT_TAG, __VA_ARGS__)

using namespace dorado::splitter;

namespace {

template <typename T>
void check_equal(const SampleRanges<T> &lhs, const SampleRanges<T> &rhs) {
    CATCH_REQUIRE(lhs.size() == rhs.size());
    for (std::size_t i = 0; i < lhs.size(); i++) {
        CATCH_CHECK(lhs[i].start_sample == rhs[i].start_sample);
        CATCH_CHECK(lhs[i].end_sample == rhs[i].end_sample);
        CATCH_CHECK(lhs[i].argmax_sample == rhs[i].argmax_sample);
        CATCH_CHECK(lhs[i].max_val == rhs[i].max_val);
    }
}

template <typename T>
constexpr at::ScalarType get_dtype() {
    if constexpr (std::is_same_v<T, int16_t>) {
        return at::kShort;
    } else if constexpr (std::is_same_v<T, c10::Half>) {
        return at::kHalf;
    } else if constexpr (std::is_same_v<T, float>) {
        return at::kFloat;
    } else {
        static_assert(std::is_same_v<T, void>);
    }
}

DEFINE_TEMPLATE_TEST("detect_pore_signal() smoke test", int16_t, float, c10::Half) {
    constexpr auto dtype = get_dtype<TestType>();
    const auto options = at::TensorOptions().dtype(dtype);

    struct {
        const char *name;
        std::vector<TestType> input;
        TestType threshold;
        uint64_t cluster_dist;
        uint64_t ignore_prefix;
        uint64_t ignore_spikes_threshold;
        SampleRanges<TestType> expected;
    } tests[] = {
            {
                    .name = "basic",
                    .input = {4, 5, 3, 2, 4},
                    .threshold = 0,
                    .cluster_dist = 0,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(0, 5, 1, 5),
                            },
            },

            // .threshold
            {
                    .name = "threshold single",
                    .input = {1, 4, 3, 1, 10, 12, 11, 13, 4, 2, 5, 3},
                    .threshold = 5,
                    .cluster_dist = 0,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(4, 8, 7, 13),
                            },
            },
            {
                    .name = "threshold split",
                    .input = {11, 14, 13, 11, 0, 2, 1, 3, 14, 12, 15, 13},
                    .threshold = 5,
                    .cluster_dist = 0,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(0, 4, 1, 14),
                                    SampleRange<TestType>(8, 12, 10, 15),
                            },
            },

            // .cluster_dist
            {
                    .name = "clustering=1",
                    .input = {10, 10, -10, -10, 10, 10},
                    .threshold = 0,
                    .cluster_dist = 1,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(0, 2, 1, 10),
                                    SampleRange<TestType>(4, 6, 5, 10),
                            },
            },
            {
                    .name = "clustering=2",
                    .input = {10, 10, -10, -10, 10, 10},
                    .threshold = 0,
                    .cluster_dist = 2,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(0, 6, 5, 10),
                            },
            },

            // .ignore_prefix
            {
                    .name = "prefix",
                    .input = {1, 4, 3, 2, 1},
                    .threshold = 0,
                    .cluster_dist = 0,
                    .ignore_prefix = 2,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(2, 5, 2, 3),
                            },
            },

            // .ignore_spikes_threshold
            {
                    .name = "spike_threshold=1",
                    .input = {-1, -1, 5, -1, -1},
                    .threshold = 0,
                    .cluster_dist = 0,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 1,
                    .expected =
                            {
                                    SampleRange<TestType>(2, 3, 2, 5),
                            },
            },
            {
                    .name = "spike_threshold=2",
                    .input = {-1, -1, 5, -1, -1},
                    .threshold = 0,
                    .cluster_dist = 0,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 2,
                    .expected = {},
            },

            // repeated argmax
            {
                    .name = "argmax",
                    .input = {1, 1, 2, 2, 1, 1},
                    .threshold = 0,
                    .cluster_dist = 0,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(0, 6, 3, 2),
                            },
            },

            // long input
            {
                    .name = "long input",
                    .input = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6},
                    .threshold = 5,
                    .cluster_dist = 0,
                    .ignore_prefix = 0,
                    .ignore_spikes_threshold = 0,
                    .expected =
                            {
                                    SampleRange<TestType>(5, 6, 5, 9),
                                    SampleRange<TestType>(7, 8, 7, 6),
                                    SampleRange<TestType>(11, 15, 14, 9),
                                    SampleRange<TestType>(18, 19, 18, 8),
                                    SampleRange<TestType>(20, 21, 20, 6),
                            },
            },
    };

    for (auto &test : tests) {
        CATCH_CAPTURE(test.name);
        const auto signal = at::from_blob(std::data(test.input), std::size(test.input), options);
        auto peaks = detect_pore_signal<TestType>(signal, test.threshold, test.cluster_dist,
                                                  test.ignore_prefix, test.ignore_spikes_threshold);
        check_equal(test.expected, peaks);
    }
}

DEFINE_TEMPLATE_TEST("detect_pore_signal() big input, spikes", int16_t, float, c10::Half) {
    constexpr auto dtype = get_dtype<TestType>();
    const auto options = at::TensorOptions().dtype(dtype);

    const std::size_t max_size = 100;
    const float range = 20;  // using range as threshold
    std::minstd_rand rng;
    std::uniform_real_distribution<float> dist(-range, range);

    for (std::size_t spike_idx_1 = 0; spike_idx_1 < max_size; spike_idx_1++) {
        for (std::size_t spike_idx_2 = 0; spike_idx_2 < max_size; spike_idx_2++) {
            CATCH_CAPTURE(spike_idx_1, spike_idx_2);

            // Add both spikes. It's intentional for both to sometimes overlap.
            std::vector<TestType> input(max_size);
            std::generate(input.begin(), input.end(),
                          [&] { return static_cast<TestType>(dist(rng)); });
            input[spike_idx_1] = range + 1;
            input[spike_idx_2] = range + 1;

            const auto signal = at::from_blob(std::data(input), std::size(input), options);
            const auto peaks = detect_pore_signal<TestType>(signal, range, 0, 0, 0);

            SampleRanges<TestType> expected;
            if (spike_idx_1 == spike_idx_2) {
                // Both the same spike.
                const std::size_t idx = spike_idx_1;
                expected.push_back(SampleRange<TestType>(idx, idx + 1, idx, input[idx]));
            } else if (spike_idx_1 == spike_idx_2 + 1 || spike_idx_2 == spike_idx_1 + 1) {
                // Close enough that they combine together.
                const std::size_t start = std::min(spike_idx_1, spike_idx_2);
                const std::size_t end = std::max(spike_idx_1, spike_idx_2);
                expected.push_back(SampleRange<TestType>(start, end + 1, end, input[end]));
            } else {
                // 2 unique spikes.
                std::array<std::size_t, 2> spikes{spike_idx_1, spike_idx_2};
                std::sort(spikes.begin(), spikes.end());
                for (std::size_t i : spikes) {
                    expected.push_back(SampleRange<TestType>(i, i + 1, i, input[i]));
                }
            }
            check_equal(expected, peaks);
        }
    }
}

DEFINE_TEST("fast_half_comparable()") {
    using dorado::splitter::detail::fast_half_comparable;
    using limits = std::numeric_limits<c10::Half>;

    // Pick some arbitrary inputs.
    std::set<c10::Half> inputs;
    for (int i = -32; i <= 32; i++) {
        inputs.insert(i / 1.f);
        inputs.insert(i / 2.f);
        inputs.insert(i / 3.f);
    }
    for (c10::Half v : {limits::max(), limits::min(), limits::lowest(), -limits::lowest()}) {
        inputs.insert(v / 1.f);
        inputs.insert(v / 2.f);
        inputs.insert(v / 3.f);
    }

    // Test them all against each other.
    for (float i : inputs) {
        for (float j : inputs) {
            CATCH_CAPTURE(i, j);
            const bool expected = i < j;
            const bool result = fast_half_comparable(i) < fast_half_comparable(j);
            CATCH_CHECK(expected == result);
        }
    }
}

}  // namespace
