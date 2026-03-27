from zoology.config import ModelConfig, ModuleConfig


# Attention
def add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    for d_model in [32, 64, 128]:
        attention_mixer = dict(
            name="zoology.mixers.attention.MHA",
            kwargs={
                "dropout": 0.1,
                "num_heads": 2
            },
        )
        mixers = [conv_mixer, attention_mixer] if conv_mixer is not None else [attention_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type = "TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="attention",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# BASED
def add_based(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    for d_model in [
        48,
        64, 
        128, 
        # 256
    ]:
        for ftr_dim in [
            8, 
            16, 
            24,
            # 32, 
            # 64
        ]:
            lin_attn = dict(
                name="zoology.mixers.based.Based",
                kwargs={
                    "l_max": input_seq_len,
                    "feature_dim": ftr_dim,
                    "feature_name": "taylor_exp",
                    "num_key_value_heads": 1,
                    "num_heads": 1,
                    "train_view": "quadratic",
                }
            )
            mixers = [conv_mixer, lin_attn] if conv_mixer is not None else [lin_attn]
            mixer = ModuleConfig(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": mixers}
            )
            name = f"based"
            model = ModelConfig(
                block_type="TransformerBlock",
                d_model=d_model,
                n_layers=num_layers,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name=name,
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Sliding window 
def add_sliding_window(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    for d_model in [128]:
        for slide_width in [8, 16, 32, 64, 128, 256, 512, 1024]:
            slide_attn = dict(
                name="zoology.mixers.slide_attn.SlidingAttn",
                kwargs={
                    "block_size": slide_width,
                    "attention_dropout": 0.0
                }
            )
            mixers = [conv_mixer, slide_attn] if conv_mixer is not None else [slide_attn]
            mixer = dict(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": mixers}
            )
            name = f"sliding-window-attention"
            n_layers = 2
            model = ModelConfig(
                block_type="TransformerBlock",
                d_model=d_model,
                n_layers=2,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name=name,
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Mamba 
def add_mamba(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "MambaBlock"
    for d_model in [64, 128, 256]:
        for d_state in [8, 16, 24]:
            mixer = dict(
                name="zoology.mixers.mamba.Mamba",
                kwargs={"d_state": d_state}
            )
            model = ModelConfig(
                block_type="MambaBlock",
                d_model=d_model,
                n_layers=num_layers,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name="mamba",
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Mamba2
def add_mamba2(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "Mamba2Block"
    for d_model in [64, 128, 256]:
        for d_state in [8, 16, 24]:
            mixer = dict(
                name="zoology.mixers.mamba2.Mamba2",
                kwargs={"d_state": d_state}
            )
            model = ModelConfig(
                block_type="Mamba2Block",
                d_model=d_model,
                n_layers=num_layers,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name="mamba2",
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Hyena 
def add_hyena(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        mixer = dict(
            name="zoology.mixers.hyena.Hyena",
            kwargs={"l_max": input_seq_len}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="hyena",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# H3 
def add_h3(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        mixer = dict(
            name="zoology.mixers.h3.H3",
            kwargs={
                "l_max": input_seq_len,
                "d_state": d_model / 4,
                "head_dim": 2
            }
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="h3",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# RWKV7
def add_rwkv7(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        rwkv7_mixer = dict(
            name="zoology.mixers.rwkv7.RWKV7Attention",
            kwargs={
                "l_max": input_seq_len,
                "head_dim": 64, 
                "decay_low_rank_dim": 16,    # Same as head dim? 
                "gate_low_rank_dim": 64,     # Tune
                "a_low_rank_dim": 16,        # Tune
                "v_low_rank_dim": 16,        # Tune
            }
        )
        mixers = [conv_mixer, rwkv7_mixer] if conv_mixer is not None else [rwkv7_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="rwkv7",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# DeltaNet
def add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        delta_net_mixer = dict(
            name="zoology.mixers.delta_net.DeltaNet",
            kwargs={
                "l_max": input_seq_len,
                "num_heads": 2,         # Tune
                "use_beta": True,       # Tune
                "use_gate": False,      # Tune
                "use_short_conv": True, # Tune
                "conv_size": 4
            }
        )
        mixers = [conv_mixer, delta_net_mixer] if conv_mixer is not None else [delta_net_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="delta_net",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# Gated DeltaNet
def add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        delta_net_mixer = dict(
            name="zoology.mixers.gated_delta_net.GatedDeltaNet",
            kwargs={
                "l_max": input_seq_len,
                "num_heads": 2,         # Tune
                "use_gate": False,      # Tune
                "use_short_conv": True, # Tune
                "conv_size": 4
            }
        )
        mixers = [conv_mixer, delta_net_mixer] if conv_mixer is not None else [delta_net_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="gated_delta_net",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# Gated linear attention
def add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        delta_net_mixer = dict(
            name="zoology.mixers.gla.GatedLinearAttention",
            kwargs={
                "num_heads": 2,          # Tune
                "use_short_conv": False, # Tune (False default)
            }
        )
        mixers = [conv_mixer, delta_net_mixer] if conv_mixer is not None else [delta_net_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="gla",
            **model_factory_kwargs
        )
        models.append(model)
    return models


def add_flash_vqg(
    models,
    conv_mixer,
    input_seq_len,
    model_factory_kwargs,
    num_layers=2,
    num_heads=4,
    if_remote_enabled=False,
    num_codebook_vectors=32,
    block_len=8,
    vq_use_triton_shortcodes=False,
    fox_state_build_backend="torch",
    fox_remote_path_backend="torch",
    local_num_blocks=1,
    use_time_mixing="kv_shift",
    vq_score_mode="l2",
    vq_weight_mode="one-hot",
    vq_update_mode="ema",
    if_value_silu=True,
    if_output_gate_use_rmsnorm=True,
    output_gate_activation="swish",
    fox_if_local_use_vq_k=False,
    enable_layer_metrics=False,
):
    """
    Add Flash-VQG models with a shared sweep over d_model in [64, 128, 256].

    Args:
        num_heads: Number of attention heads used by Flash-VQG. This also sets
            key_dim and value_dim as d_model // num_heads.
        if_remote_enabled: Whether to enable the remote VQ/FoX branch. False
            keeps the original local-only behavior used by existing debug
            scripts; True enables the hybrid local + remote Flash-VQG path.
        num_codebook_vectors: Either a fixed integer codebook size for every
            d_model, or a mapping from d_model to codebook size, e.g.
            {64: 64, 128: 128, 256: 256}.
        block_len: Flash-VQG local block length.
        vq_use_triton_shortcodes: Whether to use the Triton shortcode path for
            VQ lookup during training.
        fox_state_build_backend: Backend for FoX state build. One of
            {"torch", "triton"}.
        fox_remote_path_backend: Backend for FoX remote reduce. One of
            {"torch", "triton"}.
        local_num_blocks: Number of local FoX blocks to retain.
        use_time_mixing: Flash-VQG time-mixing mode. Typical values are
            "kv_shift", "shortconv", or None.
        vq_score_mode: VQ score mode, e.g. "l2".
        vq_weight_mode: VQ weight mode, e.g. "one-hot".
        vq_update_mode: VQ update mode, e.g. "ema".
        if_value_silu: Whether to enable value SiLU in Flash-VQG.
        if_output_gate_use_rmsnorm: Whether output gate uses RMSNorm.
        output_gate_activation: Output gate activation, e.g. "swish".
        fox_if_local_use_vq_k: Whether local path uses VQ key states.
    """
    vocab_size = model_factory_kwargs.get("vocab_size", 8_192)
    for d_model in [64, 128, 256]:
        if isinstance(num_codebook_vectors, dict):
            if d_model not in num_codebook_vectors:
                raise ValueError(
                    f"Missing num_codebook_vectors for d_model={d_model}. "
                    f"Got keys={sorted(num_codebook_vectors.keys())}."
                )
            num_codes = int(num_codebook_vectors[d_model])
        else:
            num_codes = int(num_codebook_vectors)
        flash_vqg_mixer = dict(
            name="zoology.mixers.flash_vqg.FlashVQGMixer",
            kwargs={
                "vocab_size": vocab_size,
                "num_heads": int(num_heads),
                "key_dim": d_model // int(num_heads),
                "value_dim": d_model // int(num_heads),
                "num_codebook_vectors": num_codes,
                "block_len": int(block_len),
                "local_num_blocks": int(local_num_blocks),
                "if_remote_enabled": bool(if_remote_enabled),
                "attn_backend": "flash",
                "attn_cfg": {},
                "vq_use_triton_shortcodes": bool(vq_use_triton_shortcodes),
                "fox_state_build_backend": str(fox_state_build_backend),
                "fox_remote_path_backend": str(fox_remote_path_backend),
                "use_time_mixing": use_time_mixing,
                "vq_score_mode": str(vq_score_mode),
                "vq_weight_mode": str(vq_weight_mode),
                "vq_update_mode": str(vq_update_mode),
                "if_value_silu": bool(if_value_silu),
                "if_output_gate_use_rmsnorm": bool(if_output_gate_use_rmsnorm),
                "output_gate_activation": str(output_gate_activation),
                "fox_if_local_use_vq_k": bool(fox_if_local_use_vq_k),
                "codebook_beta": 0.25,
                "enable_layer_metrics": bool(enable_layer_metrics),
            },
        )
        mixers = [conv_mixer, flash_vqg_mixer] if conv_mixer is not None else [flash_vqg_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="flash_vqg",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# Deepseek NSA
def add_deepseek_nsa(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        nsa_mixer = dict(
            name="zoology.mixers.deepseek_nsa.SparseAttention",
            kwargs={
                "num_heads": 2,            # Tune
                "sliding_window_size": 16, # Tune
                "compress_block_size": 8, # Tune
                "selection_block_size": 8, # Tune
                "num_selected_blocks": 4,   # Tune
            }
        )
        mixers = [conv_mixer, nsa_mixer] if conv_mixer is not None else [nsa_mixer]
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": mixers}
        )
        model = ModelConfig(
            block_type=block_type,
            d_model=d_model,
            n_layers=num_layers,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="deepseek_nsa",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# TTT (Test-Time Training)
def add_ttt(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=2):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        for ttt_type in ["mlp", "linear"]:  
            for mini_batch_size in [16, 32]:
                ttt_mixer = dict(
                    name="zoology.mixers.ttt.TTT",
                    kwargs={
                        "num_heads": 2,  # Scale heads with model size
                        "ttt_layer_type": ttt_type,
                        "ttt_base_lr": 1.0,
                        "mini_batch_size": mini_batch_size,
                        "use_gate": False,
                        "share_qk": False,
                        "pre_conv": False,
                        "conv_kernel": 4,
                    }
                )
                mixers = [conv_mixer, ttt_mixer] if conv_mixer is not None else [ttt_mixer]
                mixer = ModuleConfig(
                    name="zoology.mixers.hybrid.Hybrid",
                    kwargs={"configs": mixers}
                )
                model = ModelConfig(
                    block_type=block_type,
                    d_model=d_model,
                    n_layers=num_layers,
                    sequence_mixer=mixer,
                    max_position_embeddings=0,
                    name=f"ttt_{ttt_type}",
                    **model_factory_kwargs
                )
                models.append(model)
    return models
