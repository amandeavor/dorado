#pragma once

#include "basecall/MetalCaller.h"

namespace dorado::basecall {

namespace model {
struct TxModelImpl;
}  // namespace model

class MetalTxCaller final : public MetalCaller {
public:
    MetalTxCaller(const config::BasecallModelConfig &model_config);
    ~MetalTxCaller();

    at::Tensor create_input_tensor() const override;

    static int get_max_safe_batch_size(float memory_limit_fraction,
                                       const config::BasecallModelConfig &model_config);
    static int get_batch_size_granularity();

private:
    void load_tx_model(const config::BasecallModelConfig &model_config);
    bool run_scan_kernels(MTL::CommandBuffer *const cb, int try_count);
    DecodedData decode(int chunk_idx) const override;
    bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) override;

    std::unique_ptr<model::TxModelImpl> m_model;
    NS::SharedPtr<MTL::CommandQueue> m_command_queue;

    at::ScalarType m_scores_dtype = at::kHalf;
    at::ScalarType m_posts_dtype = at::kFloat;

    int m_in_chunk_size, m_out_chunk_size, m_batch_size, m_states;
    at::Tensor m_scores_TNC, m_posts_NTC, m_bwd_NTC;

    NS::SharedPtr<MTL::ComputePipelineState> m_bwd_scan_float_cps, m_fwd_scan_add_softmax_float_cps;
};

}  // namespace dorado::basecall
