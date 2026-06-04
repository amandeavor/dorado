#include "BenchmarkCache.h"
#include "SpeedEntry.h"
#include "TestUtils.h"
#include "entries_equal.h"

#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[batchsize_benchmarks]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using namespace dorado::batchsize_benchmarks;

namespace {

DEFINE_TEST("Compiled cache works") {
    // This test relies on the "test.csv" timings being baked in.
    const SpeedEntry expected_0_0[]{
            {1, 2, 3},
            {4, 5, 6},
    };
    const SpeedEntry expected_0_1[]{
            {7, 8, 9},
            {10, 11, 12},
    };
    const SpeedEntry expected_1_0[]{
            {13, 14, 15},
            {16, 17, 18},
            {19, 20, 21},
    };
    const SpeedEntry expected_1_2[]{
            {22, 23, 24},
    };

    BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
        CATCH_CHECK(tests::entries_equal(proxy.get_timings("fake gpu 0", "fake model 0"),
                                         expected_0_0));
        CATCH_CHECK(tests::entries_equal(proxy.get_timings("fake gpu 0", "fake model 1"),
                                         expected_0_1));
        CATCH_CHECK(tests::entries_equal(proxy.get_timings("fake gpu 1", "fake model 0"),
                                         expected_1_0));
        CATCH_CHECK(tests::entries_equal(proxy.get_timings("fake gpu 1", "fake model 2"),
                                         expected_1_2));

        // "Missing" entries should be empty.
        CATCH_CHECK(proxy.get_timings("fake gpu 0", "fake model 2").empty());
        CATCH_CHECK(proxy.get_timings("fake gpu 1", "fake model 1").empty());
    });

    // Test that the runtime cache is prioritized over the compiled one.
    // We do this here rather than in another test so that there isn't an
    // ordering issue when it comes to running the tests.
    {
        std::vector<SpeedEntry> entries{
                {100, 200, 300},
        };
        BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
            proxy.add_timings("fake gpu 0", "fake model 0", entries);
            CATCH_CHECK(
                    tests::entries_equal(proxy.get_timings("fake gpu 0", "fake model 0"), entries));
        });
    }
}

DEFINE_TEST("Runtime cache works") {
    const std::string gpu_name = "test gpu";
    const std::string model_name = "test model";

    // There shouldn't exist any entries yet.
    BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
        CATCH_CHECK(proxy.get_timings(gpu_name, model_name).empty());
    });

    // Add entries to the cache and check that they match.
    {
        std::vector<SpeedEntry> entries{
                // Intentionally not sorted.
                {4, 5, 6},
                {7, 8, 9},
                {1, 2, 3},
        };
        BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
            proxy.add_timings(gpu_name, model_name, entries);
        });

        // The cache should sort them when they're added, so do the same here.
        tests::entries_sort(entries);
        BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
            CATCH_CHECK(tests::entries_equal(proxy.get_timings(gpu_name, model_name), entries));
        });
    }

    // Replace the runtime values with a different set.
    {
        std::vector<SpeedEntry> entries = {
                // Intentionally not sorted.
                {400, 500, 600},
                {100, 200, 300},
        };
        BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
            proxy.add_timings(gpu_name, model_name, entries);
        });

        // Check that we get back the new entries.
        tests::entries_sort(entries);
        BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
            CATCH_CHECK(tests::entries_equal(proxy.get_timings(gpu_name, model_name), entries));
        });
    }
}

DEFINE_TEST("Saving and loading runtime cache works") {
    const TempDir runtime_cache_dir = make_temp_dir("runtime_cache_dir");
    const auto runtime_cache = runtime_cache_dir.m_path / "cache.csv";

    const std::string gpu_name = "saving test gpu";
    const std::string model_name = "saving test model";
    const std::vector<SpeedEntry> entries_saved{
            {1, 2, 3},
            {4, 5, 6},
    };
    const std::vector<SpeedEntry> entries_replaced{
            {7, 8, 9},
    };

    BenchmarkCache::with_lock([&](BenchmarkCache::CacheProxy proxy) {
        // Add the entries we'll load back.
        proxy.add_timings(gpu_name, model_name, entries_saved);
        CATCH_CHECK(tests::entries_equal(proxy.get_timings(gpu_name, model_name), entries_saved));

        // Save them to a file
        CATCH_CHECK(proxy.export_to_file(runtime_cache));

        // Replace the entries.
        proxy.add_timings(gpu_name, model_name, entries_replaced);
        CATCH_CHECK(
                tests::entries_equal(proxy.get_timings(gpu_name, model_name), entries_replaced));

        // Load them back from the file.
        CATCH_CHECK(proxy.load_from_file(runtime_cache));

        // Check that they match the original.
        CATCH_CHECK(tests::entries_equal(proxy.get_timings(gpu_name, model_name), entries_saved));
    });
}

DEFINE_TEST("GPU name alias works") {
    BenchmarkCache::with_lock([](BenchmarkCache::CacheProxy proxy) {
        const std::string_view model_name = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0";
        const auto aliased = proxy.get_timings("NVIDIA A100-PCIE-40GB", model_name);
        const auto expected = proxy.get_timings("NVIDIA A100 80GB PCIe", model_name);
        CATCH_CHECK_FALSE(expected.empty());
        CATCH_CHECK(tests::entries_equal(expected, aliased));
    });
}

}  // namespace
