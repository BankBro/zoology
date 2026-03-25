from __future__ import annotations

import torch
import torch.nn as nn


class FlashVQGMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer_idx: int | None = None,
        num_heads: int = 4,
        key_dim: int | None = None,
        value_dim: int | None = None,
        num_codebook_vectors: int = 32,
        block_len: int = 8,
        local_num_blocks: int = 1,
        if_remote_enabled: bool = False,
        attn_backend: str = "flash",
        attn_cfg: dict | None = None,
        use_time_mixing: str | None = "kv_shift",
        codebook_beta: float = 0.25,
        enable_layer_metrics: bool = False,
        vocab_size: int = 32_000,
        **kwargs,
    ):
        super().__init__()
        try:
            from flash_vqg import FlashVQGConfig
            from flash_vqg.nn.attn import FlashVQGAttention
        except ImportError as e:
            raise ImportError(
                "FlashVQGMixer requires the local flash-vqg package. "
                "Install it with `pip install -e /home/lyj/mnt/project/Flash-VQG`."
            ) from e

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")

        key_dim = d_model // num_heads if key_dim is None else int(key_dim)
        value_dim = d_model // num_heads if value_dim is None else int(value_dim)

        if key_dim <= 0 or value_dim <= 0:
            raise ValueError(f"key_dim and value_dim must be positive, got {key_dim=} {value_dim=}.")

        self.num_heads = int(num_heads)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_codebook_vectors = int(num_codebook_vectors)
        self.block_len = int(block_len)
        self.local_num_blocks = int(local_num_blocks)
        self.if_remote_enabled = bool(if_remote_enabled)
        self.codebook_beta = float(codebook_beta)
        self.enable_layer_metrics = bool(enable_layer_metrics)
        self._last_aux: dict | None = None

        cfg = FlashVQGConfig(
            vocab_size=int(vocab_size),
            hidden_size=int(d_model),
            num_hidden_layers=1,
            num_attention_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            num_codebook_vectors=self.num_codebook_vectors,
            block_len=self.block_len,
            local_num_blocks=self.local_num_blocks,
            if_remote_enabled=self.if_remote_enabled,
            attn_backend=attn_backend,
            attn_cfg={} if attn_cfg is None else attn_cfg,
            use_time_mixing=use_time_mixing,
            codebook_beta=self.codebook_beta,
            enable_layer_metrics=self.enable_layer_metrics,
            **kwargs,
        )
        self.attn = FlashVQGAttention(cfg, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        self._last_aux = None
        out = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        self._last_aux = out.aux
        return out.hidden_states

    def get_auxiliary_loss(self) -> torch.Tensor:
        zero = self.attn.res_proj.weight.new_zeros(())
        if not self._last_aux:
            return zero
        l_commit = self._last_aux.get("l_commit")
        if not isinstance(l_commit, torch.Tensor):
            return zero
        return self.codebook_beta * l_commit

    def state_size(self, sequence_length: int = 2048):
        local_window_len = self.local_num_blocks * self.block_len
        local_state_size = self.num_heads * local_window_len * (self.key_dim + self.value_dim)
        if not self.if_remote_enabled:
            return local_state_size
        remote_state_size = self.num_heads * self.num_codebook_vectors * (self.value_dim + 1)
        return remote_state_size + local_state_size
