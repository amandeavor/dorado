set(PUBLIC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/include/secondary/architectures)

target_sources(dorado_secondary
    PUBLIC
        ${PUBLIC_BASE}/model_torch_base.h
        ${PUBLIC_BASE}/model_factory.h
        ${PUBLIC_BASE}/model_config.h
        ${PUBLIC_BASE}/model_config_validation.h
        ${PUBLIC_BASE}/model_herro.h
        ${PUBLIC_BASE}/model_weights.h
    PRIVATE
        flash_attention_module.cpp
        flash_attention_module.h
        flash_attention3_backend.cpp
        flash_attention3_backend.h
        model_config.cpp
        model_config_validation.cpp
        model_factory.cpp
        model_gru.cpp
        model_gru.h
        model_herro.cpp
        model_latent_space_lstm.cpp
        model_latent_space_lstm.h
        model_slot_attention_consensus.cpp
        model_slot_attention_consensus.h
        model_torch_base.cpp
        model_torch_script.cpp
        model_torch_script.h
        model_variant_perceiver.cpp
        model_variant_perceiver.h
        model_weights.cpp
)

target_include_directories(dorado_secondary
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
