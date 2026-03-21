from zoology.config import DataConfig, ModelConfig, ModuleConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig


vocab_size = 256
input_seq_len = 64

config = TrainConfig(
    data=DataConfig(
        train_configs=[
            MQARConfig(
                num_examples=10_000,
                vocab_size=vocab_size,
                input_seq_len=input_seq_len,
                num_kv_pairs=4,
            )
        ],
        test_configs=[
            MQARConfig(
                num_examples=1_000,
                vocab_size=vocab_size,
                input_seq_len=input_seq_len,
                num_kv_pairs=4,
            )
        ],
    ),
    model=ModelConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        max_position_embeddings=input_seq_len,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.flash_vqg.FlashVQGMixer",
            kwargs={
                "vocab_size": vocab_size,
                "num_heads": 4,
                "key_dim": 16,
                "value_dim": 16,
                "num_codebook_vectors": 32,
                "block_len": 8,
                "local_num_blocks": 1,
                "if_remote_enabled": False,
                "attn_backend": "flash",
                "attn_cfg": {},
                "codebook_beta": 0.25,
                "enable_layer_metrics": False,
            },
        ),
    ),
)

configs = [config]
