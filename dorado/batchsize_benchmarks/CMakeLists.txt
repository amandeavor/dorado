dorado_add_library(dorado_benchmarks
    NAME
        dorado_benchmarks
    PUBLIC_DIR
        batchsize_benchmarks
    SOURCES_PUBLIC
        batchsize_benchmarks.h
    SOURCES_PRIVATE
        batchsize_benchmarks.cpp
        batchsize_helpers.cpp
        batchsize_helpers.h
        BenchmarkCache.cpp
        BenchmarkCache.h
        BenchmarkSink.cpp
        BenchmarkSink.h
        compiled_timings.h
        csv_helpers.cpp
        csv_helpers.h
        SpeedEntry.h
    SOURCES_TEST
        entries_equal.h
        test_batchsize_helpers.cpp
        test_cache.cpp
        test_csv.cpp
    DEPENDS_PUBLIC
        # nothing
    DEPENDS_PRIVATE
        dorado_api
        dorado_basecall
        dorado_config
        dorado_io_lib
        dorado_read_pipeline
        dorado_utils
        spdlog::spdlog
)

# Bake the benchmark timings.
include(compile_benchmarks.cmake)
set(compiled_timings_cpp "${CMAKE_CURRENT_BINARY_DIR}/compiled_timings.cpp")
dorado_compile_benchmarks(
    FILES
        "timings/NVIDIA A100 80GB PCIe.csv"
        "timings/NVIDIA GeForce RTX 5070 Ti.csv"
        "timings/NVIDIA GeForce RTX 5090.csv"
        "timings/NVIDIA H100 NVL.csv"
        "timings/NVIDIA H100 PCIe.csv"
        "timings/NVIDIA L40S.csv"
        "timings/NVIDIA RTX 5000 Ada Generation.csv"
        "timings/NVIDIA RTX A6000.csv"
        "timings/NVIDIA RTX PRO 5000 Blackwell.csv"
        "timings/NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.csv"
        "timings/NVIDIA RTX PRO 6000 Blackwell Server Edition.csv"
        "timings/NVIDIA RTX PRO 6000 Blackwell Workstation Edition.csv"
        "timings/Orin.csv"
        "timings/Quadro GV100.csv"
        "timings/Tesla V100-PCIE-16GB.csv"
        "timings/test.csv"
    OUTPUT
        ${compiled_timings_cpp}
)
target_sources(dorado_benchmarks PRIVATE ${compiled_timings_cpp})
