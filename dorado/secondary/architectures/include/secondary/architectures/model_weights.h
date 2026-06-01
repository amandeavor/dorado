#pragma once

#include <torch/nn/module.h>

#include <filesystem>
#include <string>
#include <unordered_set>

namespace dorado::secondary {

void load_state_dict(torch::nn::Module& model,
                     const std::filesystem::path& weights_path,
                     const std::unordered_set<std::string>& non_persistent_buffers);

}  // namespace dorado::secondary
