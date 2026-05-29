#include "secondary/architectures/model_herro.h"

#include "utils/container_utils.h"

#include <ATen/TensorIndexing.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace dorado::secondary {
namespace {

constexpr int64_t BASE_ALPHABET_SIZE = 12;
constexpr int64_t BASE_EMBEDDING_SIZE = 6;
constexpr int64_t INPUT_CHANNELS = BASE_EMBEDDING_SIZE + 1;
constexpr int64_t READ_CONV_CHANNELS = 128;
constexpr int64_t MODEL_DIM = 256;
constexpr int64_t NUM_HEADS = 8;
constexpr int64_t FF_DIM = 2048;
constexpr int64_t NUM_ENCODER_LAYERS = 4;
constexpr int64_t INFO_CLASSES = 1;
constexpr int64_t BASE_CLASSES = 5;
constexpr double DROPOUT = 0.1;
constexpr double LAYER_NORM_EPS = 1.0e-5;

at::Tensor create_padding_mask(const at::Tensor& lengths, const int64_t max_length) {
    const at::Tensor positions =
            at::arange(max_length, lengths.options()).expand({lengths.size(0), max_length});
    return positions.ge(lengths.unsqueeze(-1));
}

void copy_tensor(const std::string& name,
                 const at::Tensor& src,
                 torch::OrderedDict<std::string, at::Tensor>& params,
                 torch::OrderedDict<std::string, at::Tensor>& buffers) {
    if (params.contains(name)) {
        params[name].copy_(src);
    } else if (buffers.contains(name)) {
        buffers[name].copy_(src);
    } else {
        throw std::runtime_error("Unexpected tensor in Herro weights: " + name);
    }
}

void load_torchscript_weights(ModelHerro& model, const std::filesystem::path& model_path) {
    spdlog::debug("Loading Herro model weights from file: {}", model_path.string());

    at::InferenceMode infer_guard;

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path.string(), torch::kCPU);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model weights from " + model_path.string() +
                                 " with error: " + e.what());
    }

    torch::OrderedDict<std::string, at::Tensor> params = model.named_parameters(true);
    torch::OrderedDict<std::string, at::Tensor> buffers = model.named_buffers(true);

    std::unordered_set<std::string> expected;
    for (const auto& param : params) {
        expected.emplace(param.key());
    }
    for (const auto& buffer : buffers) {
        expected.emplace(buffer.key());
    }

    std::unordered_set<std::string> loaded;
    for (const auto& param : module.named_parameters(true)) {
        copy_tensor(param.name, param.value, params, buffers);
        loaded.emplace(param.name);
    }
    for (const auto& buffer : module.named_buffers(true)) {
        copy_tensor(buffer.name, buffer.value, params, buffers);
        loaded.emplace(buffer.name);
    }

    std::vector<std::string> missing;
    for (const std::string& name : expected) {
        if (loaded.count(name) == 0) {
            missing.push_back(name);
        }
    }
    if (!missing.empty()) {
        throw std::runtime_error("Cannot load Herro weights: missing tensors in weights file: " +
                                 utils::print_container_as_string(missing, ", ", true));
    }
}

}  // namespace

HerroEncoderLayerImpl::HerroEncoderLayerImpl()
        : m_self_attn(torch::nn::MultiheadAttentionOptions(MODEL_DIM, NUM_HEADS).dropout(DROPOUT)),
          m_linear1(MODEL_DIM, FF_DIM),
          m_dropout(DROPOUT),
          m_linear2(FF_DIM, MODEL_DIM),
          m_norm1(torch::nn::LayerNormOptions({MODEL_DIM}).eps(LAYER_NORM_EPS)),
          m_norm2(torch::nn::LayerNormOptions({MODEL_DIM}).eps(LAYER_NORM_EPS)),
          m_dropout1(DROPOUT),
          m_dropout2(DROPOUT) {
    register_module("self_attn", m_self_attn);
    register_module("linear1", m_linear1);
    register_module("dropout", m_dropout);
    register_module("linear2", m_linear2);
    register_module("norm1", m_norm1);
    register_module("norm2", m_norm2);
    register_module("dropout1", m_dropout1);
    register_module("dropout2", m_dropout2);
}

at::Tensor HerroEncoderLayerImpl::forward(const at::Tensor& src,
                                          const at::Tensor& src_key_padding_mask) {
    at::Tensor x = m_norm1->forward(src);
    x = x.transpose(0, 1);
    const at::Tensor attn = std::get<0>(
            m_self_attn->forward(x, x, x, src_key_padding_mask, false /* need_weights */));
    x = src + m_dropout1->forward(attn.transpose(0, 1));

    at::Tensor y = m_norm2->forward(x);
    y = m_linear2->forward(m_dropout->forward(at::gelu(m_linear1->forward(y))));
    return x + m_dropout2->forward(y);
}

HerroEncoderImpl::HerroEncoderImpl() : m_layers(torch::nn::ModuleList()) {
    for (int64_t i = 0; i < NUM_ENCODER_LAYERS; ++i) {
        m_layers->push_back(HerroEncoderLayer());
    }
    register_module("layers", m_layers);
}

at::Tensor HerroEncoderImpl::forward(at::Tensor src, const at::Tensor& src_key_padding_mask) {
    for (const auto& layer : *m_layers) {
        src = layer->as<HerroEncoderLayer>()->forward(src, src_key_padding_mask);
    }
    return src;
}

HerroTransformerImpl::HerroTransformerImpl()
        : m_context_read(torch::nn::Sequential(
                  torch::nn::Conv2d(
                          torch::nn::Conv2dOptions(INPUT_CHANNELS, READ_CONV_CHANNELS, {17, 1})
                                  .padding({8, 0})
                                  .bias(false)),
                  torch::nn::BatchNorm2d(READ_CONV_CHANNELS),
                  torch::nn::ReLU())),
          m_context_pos(torch::nn::Sequential(
                  torch::nn::Conv2d(torch::nn::Conv2dOptions(READ_CONV_CHANNELS, MODEL_DIM, {1, 31})
                                            .bias(false)),
                  torch::nn::BatchNorm2d(MODEL_DIM),
                  torch::nn::ReLU())),
          m_encoder(HerroEncoder()),
          m_ln(torch::nn::LayerNormOptions({MODEL_DIM}).eps(LAYER_NORM_EPS)) {
    register_module("context_read", m_context_read);
    register_module("context_pos", m_context_pos);
    register_module("encoder", m_encoder);
    register_module("ln", m_ln);
}

at::Tensor HerroTransformerImpl::pad_selected_positions(
        const at::Tensor& x,
        const std::vector<at::Tensor>& target_positions) {
    int64_t max_length = 0;
    for (const at::Tensor& positions : target_positions) {
        max_length = std::max(max_length, positions.size(0));
    }

    const int64_t batch_size = std::ssize(target_positions);
    const int64_t required_numel = batch_size * max_length * MODEL_DIM;
    const bool needs_buffer = !m_padded_buffer.defined() ||
                              (m_padded_buffer.numel() < required_numel) ||
                              (m_padded_buffer.device() != x.device()) ||
                              (m_padded_buffer.scalar_type() != x.scalar_type());

    if (needs_buffer) {
        // Add some additional capacity to reduce re-allocations for small
        // increases in required numel
        constexpr float CAPACITY_FACTOR = 1.1f;
        const int64_t numel = static_cast<int64_t>(required_numel * CAPACITY_FACTOR);
        m_padded_buffer = at::empty({numel}, x.options());
    }

    at::Tensor padded =
            m_padded_buffer.narrow(0, 0, required_numel).view({batch_size, max_length, MODEL_DIM});
    padded.zero_();

    for (size_t i = 0; i < target_positions.size(); ++i) {
        using at::indexing::Slice;
        const at::Tensor selected =
                x.select(0, static_cast<int64_t>(i)).index({target_positions[i]});
        padded.index({static_cast<int64_t>(i), Slice(0, selected.size(0)), Slice()})
                .copy_(selected);
    }
    return padded;
}

at::Tensor HerroTransformerImpl::forward(at::Tensor x,
                                         const std::vector<at::Tensor>& target_positions,
                                         const at::Tensor& lengths) {
    x = m_context_read->forward(x);
    x = m_context_pos->forward(x);
    x = x.squeeze(-1).transpose(1, 2);

    x = pad_selected_positions(x, target_positions);
    const at::Tensor padding_mask = create_padding_mask(lengths, x.size(1));
    x = m_encoder->forward(x, padding_mask);
    x = x.index({padding_mask.logical_not()});
    return m_ln->forward(x);
}

ModelHerro::ModelHerro()
        : m_embedding(torch::nn::EmbeddingOptions(BASE_ALPHABET_SIZE, BASE_EMBEDDING_SIZE)),
          m_qn(HerroTransformer()),
          m_fc2(MODEL_DIM, INFO_CLASSES),
          m_bases_fc(MODEL_DIM, BASE_CLASSES) {
    register_module("embedding", m_embedding);
    register_module("qn", m_qn);
    register_module("fc2", m_fc2);
    register_module("bases_fc", m_bases_fc);
}

std::tuple<at::Tensor, at::Tensor> ModelHerro::forward(
        const at::Tensor& bases,
        const at::Tensor& qualities,
        const at::Tensor& lengths,
        const std::vector<at::Tensor>& target_positions) {
    const at::Tensor bases_embeds = m_embedding->forward(bases);
    at::Tensor x = at::cat({bases_embeds, qualities.unsqueeze(-1)}, -1);
    x = x.permute({0, 3, 1, 2});
    x = m_qn->forward(x, target_positions, lengths);
    return {m_fc2->forward(x).squeeze(-1), m_bases_fc->forward(x)};
}

std::shared_ptr<ModelHerro> load_model_herro(const std::filesystem::path& model_path) {
    std::shared_ptr<ModelHerro> model(new ModelHerro());
    load_torchscript_weights(*model, model_path);
    return model;
}

}  // namespace dorado::secondary
