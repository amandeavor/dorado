#include "MPSCaller.h"

#include "MetalCallerTask.h"
#include "basecall/crf_utils.h"
#include "decode/beam_search.h"
#include "model/TxModel.h"
#include "torch_utils/metal_utils.h"
#include "utils/memory_utils.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <mutex>

using namespace dorado::utils;
using torch::indexing::Slice;

namespace {

// Types to use for different parts of the pipeline.
constexpr at::ScalarType scores_dtype = at::kHalf;
constexpr at::ScalarType posts_dtype = at::kFloat;
at::TensorOptions get_crf_options() {
    return at::TensorOptions().device(at::kMPS).dtype(at::kHalf);
}

CREATE_POINT_OF_INTEREST_ID(MPSCaller);

}  // namespace

namespace dorado::basecall {

using namespace config;

MPSCaller::MPSCaller(const BasecallModelConfig &model_config) : MetalCaller(model_config) {
    ScopedAutoReleasePool autorelease_pool;

    if (!model_config.is_tx_model()) {
        throw std::logic_error("MPSCaller got invalid model config");
    }

    // Our metal builds assume shared memory, so it's safe to check host.
    if (auto total_mem = utils::total_host_memory_GB(); total_mem < 16) {
        spdlog::warn(
                "Less than 16GB of memory available: {}GB detected. "
                "This is below minimum spec and may cause issues",
                total_mem);
    }

    // Create MPS objects.
    {
        m_device = get_mtl_device();
        m_command_queue = NS::TransferPtr(m_device->newCommandQueue());
        m_decode_complete_event = NS::TransferPtr(m_device->newSharedEvent());

        m_bwd_scan_float_cps = make_cps(m_device.get(), "backward_scan_float", {}, std::nullopt);
        m_fwd_scan_add_softmax_float_cps =
                make_cps(m_device.get(), "forward_scan_add_softmax_float", {}, std::nullopt);
    }

    // Cache common options.
    {
        m_states = std::pow(m_config.tx->crf.n_base, model_config.state_len);

        m_decoder_options = decode::DecoderOptions();
        m_decoder_options.q_shift = model_config.qbias;
        m_decoder_options.q_scale = model_config.qscale;

        if (m_decoder_options.blank_score != m_config.tx->crf.blank_score) {
            spdlog::warn("Transformer model config does not have the expected blank score");
        }
    }

    // Set configuration sizes.
    {
        assert(model_config.has_normalised_basecaller_params());

        m_in_chunk_size = model_config.basecaller.chunk_size();
        // Chunk size after decimation via convolution stride.
        m_out_chunk_size = m_in_chunk_size / model_config.stride;
        m_batch_size = model_config.basecaller.batch_size();

        if (m_batch_size == 0) {
            // Testing shows that a batch size of 16 is optimal in most cases, except
            // on low memory machines where 16 can perform a lot worse than 8.
            // TODO: replace with implementation of autobatch size calculation
            if (utils::total_host_memory_GB() < 16) {
                m_batch_size = 8;
            } else {
                m_batch_size = 16;
            }
        }
    }

    // Create tensors for scan/decode.
    {
        assert(m_out_chunk_size > 0);
        assert(m_batch_size > 0);
        assert(model_config.outsize > 0);

        const int T = m_out_chunk_size;
        const int C = model_config.outsize;
        const int Cs = m_states;
        const int N = m_batch_size;

        m_scores_TNC = torch::empty({T, N, C}, scores_dtype);
        m_posts_NTC = torch::empty({N, T + 1, Cs}, posts_dtype);
        m_bwd_NTC = torch::empty({N, T + 1, Cs});
    }

    // Load the model and get ready for calling.
    m_model = load_crf_model(model_config, get_crf_options());
    start_threads();
}

MPSCaller::~MPSCaller() = default;

at::Tensor MPSCaller::create_input_tensor() const {
    // NCT
    return at::zeros({m_batch_size, m_config.num_features, m_in_chunk_size}, at::kHalf);
}

int MPSCaller::get_max_safe_batch_size(float, const config::BasecallModelConfig &) {
    // TODO: better number here
    return 32;
}

int MPSCaller::get_batch_size_granularity() { return 8; }

bool MPSCaller::run_scan_kernels(MTL::CommandBuffer *const cb, int try_count) {
    POINT_OF_INTEREST_SCOPE(MPSCaller, run_scan_kernels, "try_count=%i", try_count);

    // ScanArgs expects scores TNC tensor sizes
    std::vector<int32_t> scan_args_{m_out_chunk_size, m_batch_size, m_states};
    auto scan_args = create_vec_buffer(m_device.get(), scan_args_);
    name_mtl_object(scan_args, "scan_kernel_args");

    // TODO: optimise grid size
    launch_kernel_no_wait(
            m_bwd_scan_float_cps.get(), cb,
            {scan_args.get(), mtl_for_tensor(m_scores_TNC), mtl_for_tensor(m_bwd_NTC)}, {},
            m_batch_size, m_states);

    launch_kernel_no_wait(m_fwd_scan_add_softmax_float_cps.get(), cb,
                          {scan_args.get(), mtl_for_tensor(m_scores_TNC), mtl_for_tensor(m_bwd_NTC),
                           mtl_for_tensor(m_posts_NTC)},
                          {}, m_batch_size, m_states);

    return run_command_buffer("linear/scan/softmax", cb, try_count);
}

bool MPSCaller::call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) {
    auto scores_TNC = m_model->forward(task.input->to(get_crf_options()))
                              .transpose(0, 1)
                              .contiguous()
                              .to(scores_dtype);

    MTL::CommandBuffer *const cb = next_command_buffer(m_command_queue.get(), try_count);
    if (m_decode_complete_event) {
        // wait for the previous decode task to complete - this acts as a mutex
        // previous scores are processed in the decode threads
        cb->encodeWait(m_decode_complete_event.get(), task.decode_complete_event_id - 1);
    }

    m_scores_TNC.index_put_({at::indexing::Ellipsis}, scores_TNC);

    std::lock_guard lock(inter_caller_mutex);
    return run_scan_kernels(cb, try_count);
}

DecodedData MPSCaller::decode(int chunk_idx) const {
    POINT_OF_INTEREST_SCOPE(MPSCaller, decode, "chunk_idx=%i", chunk_idx);

    // Not splitting batches in Tx impl so chunk idx should be in [0, N)
    assert(chunk_idx < m_batch_size);
    return decode::beam_search_decode(
            m_scores_TNC.index({Slice(), chunk_idx}), m_bwd_NTC[chunk_idx], m_posts_NTC[chunk_idx],
            m_decoder_options.beam_width, m_decoder_options.beam_cut, m_decoder_options.blank_score,
            m_decoder_options.q_shift, m_decoder_options.q_scale, 1.0f);
};

}  // namespace dorado::basecall
