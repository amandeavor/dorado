#include "utils/parameters.h"

#include <algorithm>
#include <thread>

namespace dorado::utils {

ThreadAllocations default_thread_allocations(int num_devices,
                                             int num_modbase_threads,
                                             bool enable_aligner,
                                             bool enable_barcoder,
                                             bool adapter_trimming,
                                             bool enable_polya) {
    const int max_threads = std::thread::hardware_concurrency();
    ThreadAllocations allocs;
    allocs.writer_threads = num_devices * 2;
    allocs.read_converter_threads = num_devices * 2;
    allocs.read_filter_threads = num_devices * 2;
    allocs.modbase_threads = num_devices * num_modbase_threads;
    allocs.scaler_node_threads = num_devices * 2;
    allocs.splitter_node_threads = num_devices * 2;
    allocs.loader_threads = num_devices * 4;
    allocs.polya_threads = enable_polya ? num_devices * 4 : 0;
    const int total_threads_used = allocs.writer_threads + allocs.read_converter_threads +
                                   allocs.read_filter_threads + allocs.modbase_threads +
                                   allocs.scaler_node_threads + allocs.loader_threads +
                                   allocs.splitter_node_threads + allocs.polya_threads;
    const int remaining_threads = max_threads - total_threads_used;

    // Divide up remaining threads between the active optional nodes.
    const int number_enabled = static_cast<int>(enable_aligner) +
                               static_cast<int>(enable_barcoder) +
                               static_cast<int>(adapter_trimming);
    if (number_enabled > 0) {
        const int thread_split = std::max(num_devices * 5, remaining_threads / number_enabled);
        allocs.aligner_threads = enable_aligner ? thread_split : 0;
        allocs.barcoder_threads = enable_barcoder ? thread_split : 0;
        allocs.adapter_threads = adapter_trimming ? thread_split : 0;
    }
    return allocs;
};

}  // namespace dorado::utils
