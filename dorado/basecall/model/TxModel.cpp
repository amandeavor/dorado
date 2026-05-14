#include "TxModel.h"

#include "config/BasecallModelConfig.h"
#include "nn/LinearUpsample.h"
#include "nn/TxModules.h"
#include "torch_utils/gpu_profiling.h"

namespace dorado::basecall::model {

TxModelImpl::TxModelImpl(const config::BasecallModelConfig &config,
                         const at::TensorOptions &options)
        : m_options(options) {
    convs = register_module("convs", nn::ConvStack(config.convs));
    tx_encoder =
            register_module("transformer_encoder", nn::TxEncoderStack(config.tx->tx, m_options));
    tx_decoder = register_module("transformer_decoder", nn::LinearUpsample(config.tx->upsample));
    crf = register_module("crf", nn::LinearScaledCRF(config.tx->crf));
}

at::Tensor TxModelImpl::forward(const at::Tensor &chunk_or_batch, nn::AuxiliaryData *const aux) {
    at::Tensor h;
    {
        utils::ScopedProfileRange spr("Conv", 1);
        if (aux->chunk_table.defined()) {
            // If VCS, chunk_or_batch should be entire batch
            h = convs->run_koi_vcs_sup(chunk);
        }
        else {
            // If non-VCS, chunk_or_batch is a single chunk with NCT layout
            // Returns: NTC layout
            h = convs->forward(chunk_or_batch);
        }
    }
    {
        utils::ScopedProfileRange spr("TransEnc", 1);
        h = tx_encoder(h);
    }
    {
        utils::ScopedProfileRange spr("TransDec", 1);
        h = tx_decoder(h);
    }
    {
        utils::ScopedProfileRange spr("CRF", 1);
        h = crf(h);
    }
    // Returns: NTC
    return h;
}

}  // namespace dorado::basecall::model
