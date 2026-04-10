# -*- coding: utf-8 -*-
# 此文件由 run_flash_vqg_suite.py 自动生成.
# 如需调整实验参数, 请修改 wrapper 入参后重新生成.

from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs

configs = build_configs(
    sweep_id='flash-vqg-20260402-clr-v1-e1',
    flash_backend='torch',
    logger_backend='swanlab',
    include_gdn=False,
    block_len_values=[32],
    local_num_blocks_values=[2],
    dmodels=[128],
    learning_rates=[0.001],
    if_remote_enabled_values=[True],
    train_batch_orders=['global_shuffle'],
    seed_values=[123],
    data_seed=123,
    num_codebook_vectors_values=[128],
    num_codebook_vectors_map=None,
    fox_remote_path_backend='torch',
    fox_remote_read_topk_values=[None, 1, 2, 4],
    fox_remote_formula='clr_v1',
    fox_clr_rank=4,
    fox_clr_use_den_residual=True,
    fox_clr_remat_mode='off',
    gradient_accumulation_steps=2,
    train_batch_size=128,
    eval_batch_size=16,
    cache_dir='./data/flash_vqg',
    wandb_project='flash_vqg_clr_v1_mainline',
    wandb_entity='scu-mclab',
    max_epochs=32,
    metrics_white_list=['train/loss', 'valid/loss', 'valid/accuracy', 'valid/input_seq_len/*', 'valid/num_kv_pairs/*', 'valid/mqar_case/*', 'attn/nan_inf_count', 'attn/den_min', 'attn/o_remote_energy_ratio', 'attn/clr_alpha_norm_mean', 'attn/clr_den_neg_ratio', 'attn/remote_routing_entropy', 'attn/remote_top1_top2_margin', 'attn/remote_topk_den_capture_ratio', 'valid/attn/nan_inf_count', 'valid/attn/den_min', 'valid/attn/o_remote_energy_ratio', 'valid/attn/clr_alpha_norm_mean', 'valid/attn/clr_den_neg_ratio', 'valid/attn/remote_routing_entropy', 'valid/attn/remote_top1_top2_margin', 'valid/attn/remote_topk_den_capture_ratio', 'vq/relative_err_mean', 'valid/vq/relative_err_mean'],
)
