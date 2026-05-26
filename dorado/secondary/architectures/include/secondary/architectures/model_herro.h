#pragma once

#include <ATen/Tensor.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>

#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

namespace dorado::secondary {

class HerroEncoderLayerImpl final : public torch::nn::Module {
public:
    HerroEncoderLayerImpl();

    at::Tensor forward(const at::Tensor& src, const at::Tensor& src_key_padding_mask);

private:
    torch::nn::MultiheadAttention m_self_attn{nullptr};
    torch::nn::Linear m_linear1{nullptr};
    torch::nn::Dropout m_dropout{nullptr};
    torch::nn::Linear m_linear2{nullptr};
    torch::nn::LayerNorm m_norm1{nullptr};
    torch::nn::LayerNorm m_norm2{nullptr};
    torch::nn::Dropout m_dropout1{nullptr};
    torch::nn::Dropout m_dropout2{nullptr};
};
TORCH_MODULE(HerroEncoderLayer);

class HerroEncoderImpl final : public torch::nn::Module {
public:
    HerroEncoderImpl();

    at::Tensor forward(at::Tensor src, const at::Tensor& src_key_padding_mask);

private:
    torch::nn::ModuleList m_layers{nullptr};
};
TORCH_MODULE(HerroEncoder);

class HerroTransformerImpl final : public torch::nn::Module {
public:
    HerroTransformerImpl();

    at::Tensor forward(at::Tensor x,
                       const std::vector<at::Tensor>& target_positions,
                       const at::Tensor& lengths);

private:
    at::Tensor pad_selected_positions(const at::Tensor& x,
                                      const std::vector<at::Tensor>& target_positions);

    torch::nn::Sequential m_context_read{nullptr};
    torch::nn::Sequential m_context_pos{nullptr};
    HerroEncoder m_encoder{nullptr};
    torch::nn::LayerNorm m_ln{nullptr};
    at::Tensor m_padded_buffer{nullptr};
};
TORCH_MODULE(HerroTransformer);

class ModelHerro final : public torch::nn::Module {
public:
    std::tuple<at::Tensor, at::Tensor> forward(const at::Tensor& bases,
                                               const at::Tensor& qualities,
                                               const at::Tensor& lengths,
                                               const std::vector<at::Tensor>& target_positions);

private:
    friend std::shared_ptr<ModelHerro> load_model_herro(const std::filesystem::path& model_path);

    ModelHerro();

    torch::nn::Embedding m_embedding{nullptr};
    HerroTransformer m_qn{nullptr};
    torch::nn::Linear m_fc2{nullptr};
    torch::nn::Linear m_bases_fc{nullptr};
};

std::shared_ptr<ModelHerro> load_model_herro(const std::filesystem::path& model_path);

}  // namespace dorado::secondary
