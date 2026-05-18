#include "nn/AuxiliaryData.h"

#include "nn/KoiThreads.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>

#include <numeric>
#include <stdexcept>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAStream.h>

extern "C" {
#include "koi.h"
}
#endif

namespace dorado {
namespace nn {

AuxiliaryData::AuxiliaryData(at::Tensor workspace,
                             const std::int32_t batch_size,
                             const std::int32_t chunk_size,
                             const std::int32_t stride,
                             const std::int32_t chunk_size_granularity_,
                             const std::span<const std::int32_t> chunk_sizes)
        : workspace_(std::move(workspace)),
          N_(batch_size),
          T_in_(chunk_size),
          T_out_(chunk_size / stride),
          T_lstm_(1 + T_out_ + 1),
          stride_(stride),
          chunk_size_granularity(chunk_size_granularity_) {
    T_lstm_ += T_lstm_ & 1;  // needs to be even for easier LUT creation

    total_num_varlen_chunks = std::ssize(chunk_sizes);
    chunk_table_.resize(2 * total_num_varlen_chunks);
    chunk_intervals_.resize(2 * total_num_varlen_chunks);
    int i = 0;
    total_num_granularity = 0;
    for (const std::int32_t& cs : chunk_sizes) {
        int cs_blocks = cs / chunk_size_granularity;
        chunk_table_[(2 * i) + 0] = total_num_granularity;
        chunk_table_[(2 * i) + 1] = cs_blocks;
        chunk_intervals_[(2 * i) + 0] = total_num_granularity;
        chunk_intervals_[(2 * i) + 1] = total_num_granularity + cs_blocks;
        total_num_granularity += cs_blocks;
        ++i;
    }
}

void AuxiliaryData::restore_convolution_auxiliary_data() {
#if DORADO_CUDA_BUILD
    if (!device_chunk_intervals.defined()) {
        throw std::runtime_error("AuxiliaryData error: undefined chunk intervals!");
    }
    device_chunk_intervals.mul_(stride_);
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

void AuxiliaryData::create_lstm_auxiliary_data([[maybe_unused]] const at::Device& device,
                                               [[maybe_unused]] KoiThreads& thread_pool) {
#if DORADO_CUDA_BUILD
    if (device_in_layout.defined()) {
        return;
    }

    auto options = at::TensorOptions().device(device).dtype(at::kInt);
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    const std::int32_t chunk_sum =
            std::accumulate(std::cbegin(chunk_sizes_), std::cend(chunk_sizes_), 0);

    device_chunk_intervals =
            at::from_blob(std::data(chunk_intervals_),
                          {static_cast<std::int32_t>(std::size(chunk_intervals_))}, options);

    device_in_layout = at::empty({chunk_sum}, options);
    device_out_layout = at::empty({N_ * (T_lstm_ + 1)}, options);
    device_fwd_encoding = at::empty({N_ * T_lstm_}, options);
    device_bwd_encoding = at::empty({N_ * (T_lstm_ + 1)}, options);

    constexpr std::int32_t SUBBATCH_SIZE{32};

    const int status = host_lstm_preprocess(
            stream.stream(), N_, std::data(chunk_sizes_), std::size(chunk_sizes_), SUBBATCH_SIZE,
            T_lstm_, workspace_.data_ptr<std::int32_t>(), workspace_.size(0), thread_pool.get(),
            nullptr, device_out_layout.data_ptr<std::int32_t>(),
            device_fwd_encoding.data_ptr<std::int32_t>(), device_in_layout.data_ptr<std::int32_t>(),
            nullptr, device_bwd_encoding.data_ptr<std::int32_t>());

    if (status != KOI_SUCCESS) {
        throw std::runtime_error("RNN auxiliary data creation failed.");
    }
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

void AuxiliaryData::create_shared_auxiliary_data([[maybe_unused]] const at::Device& device) {
#if DORADO_CUDA_BUILD
    if (device_chunk_table.defined()) {
        return;
    }

    device_chunk_table = at::from_blob(std::data(chunk_table_),
                                       {static_cast<std::int32_t>(std::size(chunk_table_))},
                                       at::TensorOptions().device(device).dtype(at::kInt));
#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

void AuxiliaryData::create_tx_auxiliary_data([[maybe_unused]] const at::Device& device) {
#if DORADO_CUDA_BUILD
    if (conv_load_lut.defined() || conv_store_lut.defined() || qkv_rope_lut.defined()) {
        return;
    }

    auto i32_opts = at::TensorOptions().device(device).dtype(at::kInt32);
    conv_load_lut = at::empty({total_num_granularity}, i32_opts);
    conv_store_lut = at::empty({total_num_granularity}, i32_opts);
    qkv_rope_lut = at::empty({total_num_granularity}, i32_opts);

    // conv_load_lut and conv_store_lut get used on every ConvLayer
    // qkv_rope_lut does get filled once for a given batch, but its doesn't get filled here
    // because it would involve hardcoding TxEncoder granularity here, which wouldn't be
    // the end of the world tbh but decided to fill qkv_rope_lut in TxModules.cpp

#else
    throw std::runtime_error("AuxiliaryData error: unsupported code path!");
#endif
}

}  // namespace nn
}  // namespace dorado
