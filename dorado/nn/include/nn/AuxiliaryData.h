#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>

#include <cstdint>
#include <span>
#include <vector>

namespace dorado::nn {

class KoiThreads;

class AuxiliaryData {
public:
    AuxiliaryData(at::Tensor workspace,
                  std::int32_t batch_size,
                  std::int32_t chunk_size,
                  std::int32_t stride,
                  std::int32_t chunk_size_granularity_,
                  std::span<const std::int32_t> chunk_sizes);

    std::int32_t N() const { return N_; }
    std::int32_t T_in() const { return T_in_; }
    std::int32_t T_out() const { return T_out_; }
    std::int32_t T_lstm() const { return T_lstm_; }
    std::int32_t NT_in() const { return chunk_intervals_.back(); }
    std::int32_t NT_out() const { return NT_in() / stride_; }
    std::int32_t NT_in_max() const { return N_ * T_in_; }
    std::int32_t NT_out_max() const { return NT_in_max() / stride_; }

    void restore_convolution_auxiliary_data();

    at::Tensor device_chunk_intervals;

    void create_auxiliary_data(const c10::Device& device, KoiThreads& thread_pool, bool is_lstm_model);

    at::Tensor device_in_layout;
    at::Tensor device_out_layout;
    at::Tensor device_fwd_encoding;
    at::Tensor device_bwd_encoding;

    std::span<const std::int32_t> chunk_sizes() const {  return chunk_sizes_; }

    at::Tensor device_chunk_intervals;
    // ! Need to make sure device_chunk_table is used in Decoder !
    at::Tensor device_chunk_table;  // Torch Tensor of shape (total_num_varlen_chunks, 2). Column 0 is chunk start. Column 1 is chunk_length
    at::Tensor conv_load_lut;       // Torch Tensor of shape total_num_granularity
    at::Tensor conv_store_lut;      // Torch Tensor of shape total_num_granularity
    at::Tensor qkv_rope_lut;
    int chunk_size_granularity;  // This should be chunk_size_granularity() from BasecallModelConfig.h
    int total_num_granularity;    // This is entire input length divided by chunk_size_granularity
    int total_num_varlen_chunks;  // Amount of varlen chunks in batch

private:
    at::Tensor workspace_;
    std::int32_t N_{0};
    std::int32_t T_in_{0};
    std::int32_t T_out_{0};
    std::int32_t T_lstm_{0};
    std::int32_t stride_{0};
    std::vector<std::int32_t> chunk_sizes_;
    std::vector<std::int32_t> chunk_table_;
    std::vector<std::int32_t> chunk_intervals_;
};

}  // namespace dorado::nn
