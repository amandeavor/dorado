#pragma once

#include "AuxiliaryData.h"
#include "WorkingMemory.h"
#include "config/common.h"

#include <torch/nn.h>

#include <optional>
#include <vector>

namespace dorado::nn {

struct ConvStackImpl : torch::nn::Module {
    explicit ConvStackImpl(const std::vector<config::ConvParams> &layer_params);
#if DORADO_CUDA_BUILD
    void reserve_working_memory(WorkingMemory &wm,
                                const AuxiliaryData *aux /* = nullptr */,
                                std::optional<TensorLayout> output_layout);
    void run_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
    void run_koi_vcs_sup(at::Tensor x, AuxiliaryData const *aux);
#endif  // if DORADO_CUDA_BUILD

    at::Tensor forward(at::Tensor x);

    struct ConvLayer {
        explicit ConvLayer(const config::ConvParams &params);
        const config::ConvParams params;
        torch::nn::Conv1d conv{nullptr};
        
        // These VCS SUP variables must be outside #if DORADO_CUDA_BUILD because ConvStackImpl's constructor will run regardless
        // If non DORADO_CUDA_BUILD gets run, these get filled but never used, no probs
        int conv_layer_num;
        int num_working_blocks_per_min_chunksize;
        int next_layer_padding{0};  // Initialised to 0 for last vcs sup convolution that does not get filled AND outputs without padding
#if DORADO_CUDA_BUILD
        TensorLayout output_layout{TensorLayout::NTC};
        bool cutlass_conv{false};
        int output_T_padding{0};
        at::Tensor w_device;
        at::Tensor w_t_device;
        at::Tensor b_device;

        at::Tensor conv_load_lut;
        at::Tensor conv_store_lut;
        at::Tensor conv_output;

        void reserve_working_memory(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
        void run_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
        void run_koi_vcs_sup();
#endif  // if DORADO_CUDA_BUILD
    };

    std::vector<ConvLayer> layers;
};

TORCH_MODULE(ConvStack);

}  // namespace dorado::nn