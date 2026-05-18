#include "nn/ConvStack.h"

#include "config/common.h"
#include "nn/KoiUtils.h"
#include "torch_utils/gpu_profiling.h"

#if DORADO_CUDA_BUILD
extern "C" {
#include "koi.h"
}
#endif

#include <torch/nn.h>

namespace dorado::nn {

#if DORADO_CUDA_BUILD

namespace {

KoiActivation get_koi_activation(config::Activation act) {
    if (act == config::Activation::SWISH) {
        return KOI_SWISH;
    } else if (act == config::Activation::SWISH_CLAMP) {
        return KOI_SWISH_CLAMP;
    } else if (act == config::Activation::TANH) {
        return KOI_TANH;
    } else {
        throw std::logic_error("Unrecognised activation function id.");
    }
}

// We have three different LSTM code paths:
//
// - Quantized: This path is only available for narrow LSTM layers, C == 96 or C == 128. It
//   uses CuBLAS GEMM (or torch::matmul) for the FP16 input-hidden matmul, and a custom kernel
//   using the DP4A instruction for the Int8*Int8->Int32 hidden-hidden matmul, and FP16 gate
//   computation. DP4A is not available on compute arch 6.2 (TX2).
//
// - Cutlass: This path is only available for LSTM layers where C is a multiple of 128 between
//   256 and 1024. It is currently only available on compute arch 8.0 (A100) and 9.0 (H100).
//   It uses a custom kernel based on the Cutlass library which performs Tensor Core matmul using
//   either F16 or Int8 and fuses the gate computation. FP16 is used only for the first LSTM layer,
//   and only if the output activation of the last convolution is not tanh.
// TODO: Add Cutlass kernels for 7.0 (V100, FP16) and for GPUs with less shared memory (7.x, 8.x)
//
// - CuBLAS: Slowest. This is the fallback path when none of the other paths applies. It uses
//   CuBLAS GEMM (or torch::matmul) plus `host_lstm_step_f16` from Koi. Uses FP16 precision.
//
// Each path needs its input in a different memory layout. To avoid extra transpose/conversion
// steps, the last convolution writes output in a memory layout suitable to serve as working memory
// for the first LSTM layer. (The specific memory layouts are further explained below in
// `LSTMStackImpl::forward_[cublas|cutlass]`.)
//
// These are the possible memory layouts for in/out buffers in working memory:
// [where T = chunk size ("time"), N = batch size, C = layer size ("channels")]
//
// - NTC: A contiguous tensor of size [N, T, C], dtype torch::kF16
// - TNC: A contiguous tensor of size [T, N, C], dtype torch::kF16
// - CUTLASS_TNC_F16: a contiguous tensor of size [T + 3, N, C], dtype torch::kF16
// - CUTLASS_TNC_I8: a contiguous tensor of size [T + 3, N, C], dtype torch::kI8
// - CUBLAS_TN2C: a contiguous tensor of size [T + 1, N, 2, C], dtype torch::kF16
// - CUBLAS_TNC: a contiguous tensor of size [T + 3, N, C], dtype torch::kF16
//

TensorLayout get_koi_lstm_input_layout(const int layer_size,
                                       const int inner_dim,
                                       const config::Activation activation) {
    TensorLayout layout = TensorLayout::CUBLAS_TN2C;
    if (koi_can_use_quantised_lstm() && (layer_size == 96 || layer_size == 128)) {
        layout = TensorLayout::NTC;
    } else if (koi_can_use_cutlass() && layer_size <= 1024 && layer_size > 128 &&
               (layer_size % 128) == 0) {
        layout = (activation == config::Activation::TANH) ? TensorLayout::CUTLASS_TNC_I8
                                                          : TensorLayout::CUTLASS_TNC_F16;
    }

    // Apply override (Cutlass override can only be applied if conditions are met)
    const char *env_lstm_mode = std::getenv("DORADO_LSTM_MODE");
    if (env_lstm_mode != nullptr) {
        std::string lstm_mode_str(env_lstm_mode);
        if (lstm_mode_str == "CUBLAS_TN2C") {
            layout = TensorLayout::CUBLAS_TN2C;
        } else if (lstm_mode_str == "CUTLASS_TNC_I8" && layout == TensorLayout::CUTLASS_TNC_F16) {
            layout = TensorLayout::CUTLASS_TNC_I8;
        } else if (lstm_mode_str == "CUTLASS_TNC_F16" && layout == TensorLayout::CUTLASS_TNC_I8) {
            layout = TensorLayout::CUTLASS_TNC_F16;
        }
    }

    const bool flstm = (inner_dim > 0);
    if (flstm) {  // cannot be overriden
        if (koi_can_use_cutlass() && ((layer_size % 128) == 0)) {
            if (koi_can_run_flstm() && (layer_size == 1024) && (inner_dim == 128)) {
                layout = TensorLayout::CUTLASS_TNC_I8;
            } else {
                layout = TensorLayout::CUTLASS_TNC_F16;
            }
        } else {
            layout = TensorLayout::CUBLAS_TNC;
        }
    }

    return layout;
}

}  // namespace
#endif

ConvStackImpl::ConvStackImpl(const std::vector<config::ConvParams> &layer_params) {
    for (size_t i = 0; i < layer_params.size(); ++i) {
        auto &layer = layers.emplace_back(layer_params[i]);
        auto opts = torch::nn::Conv1dOptions(layer.params.insize, layer.params.size,
                                             layer.params.winlen)
                            .stride(layer.params.stride)
                            .padding(layer.params.winlen / 2);
        layer.conv = register_module(std::string("conv") + std::to_string(i + 1),
                                     torch::nn::Conv1d(opts));
#if CUDA_DORADO_BUILD
        // I can't instantiate conv_output here because Model creation happens before AuxiliaryData creation
        layer.conv_layer_num = i;
        // Last layer has default next_layer_padding = 0, intentionally, input to tx encoder is not padded
        if (i > 0) {
            layers[i-1].next_layer_padding = (layer.params.winlen / 2);
        }
#endif  // DORADO_CUDA_BUILD
    }
}

#if DORADO_CUDA_BUILD
void ConvStackImpl::reserve_working_memory(WorkingMemory &wm,
                                           const AuxiliaryData *const aux,
                                           std::optional<TensorLayout> output_layout) {
    if (std::empty(layers)) {
        throw std::runtime_error("Empty Koi convolution stack.");
    }
    auto &last = layers.back();
    last.output_layout =
            output_layout.has_value()
                    ? output_layout.value()
                    : get_koi_lstm_input_layout(last.params.size, last.params.inner_dim,
                                                last.params.activation);

    last.cutlass_conv = utils::get_dev_opt<bool>("cutlass_conv", true) &&
                        (last.output_layout == TensorLayout::CUTLASS_TNC_I8 ||
                         last.output_layout == TensorLayout::CUTLASS_TNC_F16);
    if (last.cutlass_conv && layers.size() >= 2) {
        layers[layers.size() - 2].output_T_padding = last.params.winlen / 2;
    }
    for (auto &layer : layers) {
        layer.reserve_working_memory(wm, aux);
    }
}

void ConvStackImpl::run_koi(WorkingMemory &wm, const AuxiliaryData *const aux) {
    for (auto &layer : layers) {
        layer.run_koi(wm, aux);
    }
}

at::Tensor ConvStackImpl::run_koi_vcs_tx(at::Tensor& x, AuxiliaryData const *aux) {
    for (auto &layer : layers) {
        x = layer.run_koi_vcs_tx(x, aux);
        aux->min_chunksize /= layer.params.stride;
    }
    return x;
}
#endif  // if DORADO_CUDA_BUILD

at::Tensor ConvStackImpl::forward(at::Tensor x) {
    // Input x is [N, C_in, T_in], contiguity optional
    for (auto &layer : layers) {
        utils::ScopedProfileRange spr("conv", 2);
        x = layer.conv(x);
        if (layer.params.activation == config::Activation::SWISH) {
            torch::silu_(x);
        } else if (layer.params.activation == config::Activation::SWISH_CLAMP) {
            torch::silu_(x).clamp_(c10::nullopt, 3.5f);
        } else if (layer.params.activation == config::Activation::TANH) {
            x.tanh_();
        } else {
            throw std::logic_error("Unrecognised activation function id.");
        }
    }
    // Output is [N, T_out, C_out], non-contiguous
    return x.transpose(1, 2);
}

ConvStackImpl::ConvLayer::ConvLayer(const config::ConvParams &conv_params) : params(conv_params) {}

#if DORADO_CUDA_BUILD
void ConvStackImpl::ConvLayer::reserve_working_memory(WorkingMemory &wm,
                                                      const AuxiliaryData *const aux) {
    assert(wm.layout == TensorLayout::NTC);
    const int T_in = wm.T;
    int T_out = T_in / params.stride;
    const int C_in = wm.C;
    const int C_out = params.size;
    if (output_layout == TensorLayout::NTC && C_out > 16) {
        wm.next_TC(T_out, params.winlen * C_in, TensorLayout::NTC);
    } else if (cutlass_conv) {
        if (aux) {
            wm.next_N(aux->N());
            T_out = aux->T_lstm();
        }
    } else if (output_layout != TensorLayout::NTC) {
        wm.next_TC(T_out, params.winlen * C_in, TensorLayout::TNC);
        if (output_layout == TensorLayout::CUTLASS_TNC_I8) {
            wm.next_TC(T_out, C_out, TensorLayout::TNC);
        }
    }
    wm.next_TC(T_out + 2 * output_T_padding, C_out, output_layout);
}

void ConvStackImpl::ConvLayer::run_koi(WorkingMemory &wm, const AuxiliaryData *const aux) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    utils::ScopedProfileRange spr("conv", 2);

    auto in = wm.current;
    assert(wm.layout == TensorLayout::NTC);
    const int padding = (params.winlen / 2);
    const int T_in = cutlass_conv ? wm.T - 2 * padding : wm.T;
    const int T_out = T_in / params.stride;
    const int C_in = wm.C;
    const int C_out = params.size;

    if (!w_device.defined()) {
        auto opts = in.options().dtype(torch::kF16);
        // conv->weight is [C_out, C_in, W], we want [W, C_in, C_out]
        w_device = conv->weight.permute({2, 1, 0}).contiguous().flatten(0, 1).to(opts);
        if (cutlass_conv) {
            if (aux) {
                w_t_device = w_device.clone();  // for convolution_postprocess
            }
            w_device = w_device.transpose(0, 1).contiguous();
        }
        b_device = conv->bias.to(opts);
    }

    if (output_layout == TensorLayout::NTC && C_out <= 16) {
        utils::ScopedProfileRange spr2("small conv", 3);
        auto out = wm.next_TC(T_out + 2 * output_T_padding, C_out, output_layout);
        out = out.narrow(1, output_T_padding, T_out);
        if (host_convolution_f16(stream, wm.N, C_in, C_out, T_in, params.winlen, params.stride,
                                 params.winlen / 2, int(out.stride(0)), in.data_ptr(),
                                 out.data_ptr(), w_device.data_ptr(), b_device.data_ptr(),
                                 get_koi_activation(params.activation))) {
            throw std::runtime_error(
                    std::string("Koi convolution (host_convolution_f16) failed with in size ") +
                    std::to_string(params.insize));
        }
        if (aux) {
            if (host_convolution_postprocess(stream, get_koi_activation(params.activation), KOI_F16,
                                             aux->chunk_sizes().size(), C_in, C_out,
                                             aux->device_chunk_intervals.data_ptr(), 1,
                                             params.winlen, params.stride, in.data_ptr(),
                                             out.data_ptr(), nullptr, w_device.data_ptr(),
                                             b_device.data_ptr())) {
                throw std::runtime_error(
                        std::string("Koi convolution (host_convolution_postprocess) failed with in "
                                    "size ") +
                        std::to_string(params.insize));
            }
        }
    } else if (cutlass_conv) {
        utils::ScopedProfileRange spr2("linear conv", 3);
        auto out_type = (output_layout == TensorLayout::CUTLASS_TNC_I8) ? KOI_I8 : KOI_F16;
        in.slice(1, 0, padding) = 0;
        in.slice(1, -padding, torch::indexing::None) = 0;
        at::Tensor out_ntc;
        void *out_layout = nullptr;
        if (aux) {
            wm.next_N(aux->N());
            wm.next_TC(aux->T_lstm(), C_out, output_layout);
            out_ntc = wm.current.narrow(0, 1, wm.T + 1);
            out_layout = aux->device_in_layout.data_ptr();
        } else {
            wm.next_TC(T_out, C_out, output_layout);
            out_ntc = wm.get_current_NTC_view();
        }
        const bool try_fast_conv = utils::get_dev_opt<bool>("koi_conv", true);
        int res = -1;
        if (try_fast_conv) {
            utils::ScopedProfileRange spr3("host_convolution", 4);
            const bool use_f32_accum = utils::get_dev_opt<bool>("koi_conv_f32", false);
            const int N = aux ? 1 : wm.N;
            res = host_convolution(stream, N, T_in, C_in, C_out, params.winlen, params.stride,
                                   padding, int(in.stride(0)), int(out_ntc.stride(0)),
                                   int(out_ntc.stride(1)),
                                   in.slice(1, padding, torch::indexing::None).data_ptr(),
                                   out_ntc.data_ptr(), out_layout, w_device.data_ptr(),
                                   b_device.data_ptr(), get_koi_activation(params.activation),
                                   KOI_F16, out_type, use_f32_accum ? KOI_F32 : KOI_F16);
        }
        if (res != KOI_SUCCESS) {
            utils::ScopedProfileRange spr3("host_linear", 4);
            res = host_linear(stream, KOI_F16, get_koi_activation(params.activation), out_type,
                              aux ? 1 : wm.N, T_out, C_in * params.winlen, C_out, int(in.stride(0)),
                              params.stride * C_in, int(out_ntc.stride(0)), int(out_ntc.stride(1)),
                              in.data_ptr(), w_device.data_ptr(), out_ntc.data_ptr(), out_layout,
                              nullptr, b_device.data_ptr());
        }
        if (res != KOI_SUCCESS) {
            throw std::runtime_error(
                    std::string("Koi convolution (host_linear) failed with in size ") +
                    std::to_string(params.insize));
        }
        if (aux) {
            utils::ScopedProfileRange spr3("host_convolution_postprocess", 4);
            res = host_convolution_postprocess(
                    stream, get_koi_activation(params.activation), out_type,
                    aux->chunk_sizes().size(), C_in, C_out, aux->device_chunk_intervals.data_ptr(),
                    1, params.winlen, params.stride, in.narrow(1, padding, T_in).data_ptr(),
                    out_ntc.data_ptr(), out_layout, w_t_device.data_ptr(), b_device.data_ptr());
            if (res != KOI_SUCCESS) {
                throw std::runtime_error(
                        std::string("Koi convolution (host_convolution_postprocess) failed with in "
                                    "size ") +
                        std::to_string(params.insize));
            }
        }
    } else {
        if (aux) {
            throw std::runtime_error(
                    "Koi convolution error: unsupported variable chunk sizes code path!");
        }
        utils::ScopedProfileRange spr2("window conv", 3);
        // The window tensor is either NTC or TNC, depending on whether the first two
        // dimensions of the output layout are NT or TN.
        bool is_NT = (output_layout == TensorLayout::NTC);
        wm.next_TC(T_out, params.winlen * C_in, is_NT ? TensorLayout::NTC : TensorLayout::TNC);
        auto window_mat = wm.get_current_NTC_view();
        auto res = host_window_ntwc_f16(stream, wm.N, T_in, C_in, params.winlen, params.stride,
                                        int(window_mat.stride(0)), int(window_mat.stride(1)),
                                        in.data_ptr(), window_mat.data_ptr());
        if (res != KOI_SUCCESS) {
            throw std::runtime_error(
                    std::string("Koi convolution (host_window_ntwc_f16) failed with in size ") +
                    std::to_string(params.insize));
        }

        auto mm_in = wm.current.flatten(0, 1);
        at::Tensor mm_out;
        if (output_layout == TensorLayout::NTC) {
            mm_out = wm.next_TC(T_out, C_out, output_layout);
        } else if (output_layout == TensorLayout::CUTLASS_TNC_I8) {
            mm_out = wm.next_TC(T_out, C_out, TensorLayout::TNC);
        } else {
            wm.next_TC(T_out, C_out, output_layout);
            mm_out = wm.get_current_NTC_view().transpose(0, 1);
        }

        mm_out = mm_out.view({-1, C_out});
        dorado::utils::matmul_f16(mm_in, w_device, mm_out);
        res = host_bias_activation_f16_inplace(
                stream, int(mm_out.size(0)), int(mm_out.size(1)), int(mm_out.stride(0)),
                mm_out.data_ptr(), b_device.data_ptr(), get_koi_activation(params.activation));

        if (res != KOI_SUCCESS) {
            throw std::runtime_error(
                    std::string("Koi convolution (host_bias_activation_f16_inplace) failed with in "
                                "size ") +
                    std::to_string(params.insize));
        }

        if (output_layout == TensorLayout::CUTLASS_TNC_I8) {
            wm.next_TC(T_out, C_out, output_layout);
            auto conv_out = wm.get_current_NTC_view().transpose(0, 1).view({-1, C_out});
            host_convert(stream, mm_out.data_ptr(), 0, int(mm_out.stride(0)), int(mm_out.stride(1)),
                         KOI_F16, conv_out.data_ptr(), 0, int(conv_out.stride(0)),
                         int(conv_out.stride(1)), KOI_I8, 1, int(conv_out.size(0)),
                         int(conv_out.size(1)));
        }
    }
}

at::Tensor ConvStackImpl::ConvLayer::run_koi_vcs_tx(at::Tensor &conv_input,
                                                    AuxiliaryData const *aux) {
    // Implementation supposes chunk_table remains unchanged since entering ConvStack
    // Eg: S | L            S = Start, L = Length
    //     0 | 768
    //   768 | 768*4
    // 768*5 | 768*2
    // Also expecting chunk_table.shape = (num_varlen_chunks, 2). Linearised = [[S, L], [S, L], [S, L], ...]
    // This can of course be changed

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    utils::ScopedProfileRange spr("conv", 2);
    auto opts_f16 = in.options().dtype(torch::kF16);
    auto opts_i32 = in.options().dtype(torch::kInt32);

    const int winlen = params.winlen;
    const int padding = (winlen / 2);
    const int C_in = params.insize;
    const int C_out = params.size;

    if (!w_device.defined()) {
        // conv->weight is [C_out, C_in, W], we want [C_out, W, C_in] and then further tiling
        w_device = conv->weight.transpose(1, 2).contiguous().to(opts_f16);

        // Special tiling for first Conv
        if (conv_layer_num == 0) {
            w_device = w_device.view({C_out / 2, 2, winlen * C_in}).transpose(1, 2).contiguous();
        }
        else {
            // Last layer needs to tile OUTSIZE, too big to fit in SMEM
            const int SMEM_N = (conv_layer_num == 4) ? 128 : C_out;
            w_device = w_device.view({C_out / SMEM_N, SMEM_N / 16, 2, 8, (winlen * C_in) / 16, 2, 8})
                               .permute({0, 4, 1, 2, 5, 3, 6}).contiguous();
        }
        b_device = conv->bias.to(opts_f16);
    }

    if (!conv_output.defined()) {
        // BasecallerNode.cpp adds varlen_chunks to batch so that varlen_chunks + max_padding_in_ConvLayers does not exceed batch_size * chunk_size
        // So it's safe to instantiate output tensors with M dimension = batch_size * chunk_size, even if actual input is less than that,
        // the final excessive M rows won't get filled this layer, nor loaded next layer, they just never get touched. HOWEVER, they are still needed 
        // so we have fixed M throughout layers and batches! This way, we just instantiate conv_output once per ConvLayer, and re-use it for all batches!
        // All of this so Torch doesn't reallocate output tensors with every batch, consuming all of GPU Mem.

        M_max = aux->N() * aux->T_in(); // aux->N() and aux->T_in() should NOT change throughout entire basecalling
                                        // Once determined by CudaCaller.cpp, should stay the same until the end of basecalling
        conv_output = torch::empty(({C_out / 8, M_max, 8}), opts_f16);  // Doesn't really need to be in Koi Layout, BUT, if you decide to change it,
                                                                        // you must also change TxModules.cpp, as it gets C from THIS layout!
    }

    koi_vcs_sup_fill_conv_load_store_lut(
        stream_ptr,
        aux->total_num_granularity,
        aux->chunk_table.data_ptr(),
        aux->conv_load_lut.data_ptr(),  // Load and Store LUTs make sense for them to be in AuxiliaryData.h, as their
        aux->conv_store_lut.data_ptr(), // shape depend on total_num_granularity, which can changes between batches
        conv_input.data_ptr(),
        M_max,
        C_in,
        aux->chunk_size_granularity,    // This value gets updated according to ConvLayers stride in ConvStackImpl::run_koi_vcs_tx
        stride,
        padding,
        next_layer_padding
    );

    if (conv_layer_num == 0) {
        koi_vcs_sup_cnn1(
            stream,
            conv_input.data_ptr(),
            w_device.data_ptr(),
            conv_output.data_ptr(),
            w_bias.data_ptr(),
            aux->conv_load_lut.data_ptr(),
            aux->conv_store_lut.data_ptr(),
            aux->total_num_granularity,
            M_max   // M dimension of output tensor
        );
    }
    else {
        koi_vcs_sup_cnn(
            stream,
            conv_input.data_ptr(),
            w_device.data_ptr(),
            conv_output.data_ptr(),
            w_bias.data_ptr(),
            aux->conv_load_lut.data_ptr(),
            aux->conv_store_lut.data_ptr(),
            aux->total_num_granularity,
            M_max,              // M dimension of input tensor
            M_max               // M dimension of output tensor, both same given VCS Tx strategy!
            winlen,
            C_in,
            C_out,
            padding,
            stride
        );
    }

    return conv_output;
}

#endif  // if DORADO_CUDA_BUILD

}  // namespace dorado::nn