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
    at::Tensor run_koi_vcs_tx(at::Tensor x, AuxiliaryData *aux);
#endif  // if DORADO_CUDA_BUILD

    at::Tensor forward(at::Tensor x);

    struct ConvLayer {
        explicit ConvLayer(const config::ConvParams &params);
        const config::ConvParams params;
        torch::nn::Conv1d conv{nullptr};
        
#if DORADO_CUDA_BUILD
        TensorLayout output_layout{TensorLayout::NTC};
        bool cutlass_conv{false};
        int output_T_padding{0};
        at::Tensor w_device;
        at::Tensor w_t_device;
        at::Tensor b_device;

        int M_max;                  // Logic explained in ConvStackImpl::ConvLayer::run_koi_vcs_tx
        at::Tensor conv_output;     // Gets intialised if (aux && aux->chunk_table.defined()), and is of shape (batch_size * chunk_size, params->size)
        int conv_layer_num;
        int next_layer_padding{0};  // Initialised to 0 for last vcs sup convolution that does not get filled AND outputs without padding
        int M_out;

        void reserve_working_memory(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
        void run_koi(WorkingMemory &wm, const AuxiliaryData *aux /* = nullptr */);
        at::Tensor run_koi_vcs_tx(at::Tensor &conv_input, AuxiliaryData *aux);
#endif  // if DORADO_CUDA_BUILD
    };

    std::vector<ConvLayer> layers;
};

TORCH_MODULE(ConvStack);

}  // namespace dorado::nn