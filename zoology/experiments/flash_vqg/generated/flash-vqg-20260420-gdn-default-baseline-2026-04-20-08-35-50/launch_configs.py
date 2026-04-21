# -*- coding: utf-8 -*-
# 此文件由 run_flash_vqg_suite.py 自动生成.
# 如需调整实验参数, 请修改 wrapper 入参后重新生成.

import argparse

from zoology.experiments.flash_vqg.run_flash_vqg_suite import _load_config_builder

_builder = _load_config_builder('/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260420-gdn-default-baseline/config_builder.py:build_gdn_default_config')
_builder_args = argparse.Namespace(**{'backend': 'torch', 'logger_backend': 'swanlab', 'include_gdn': False, 'block_len': '8', 'paired_block_local': None, 'dmodels': '128', 'num_codebook_vectors': None, 'num_codebook_vectors_map': None, 'learning_rates': '1e-3', 'if_remote_enabled': 'true', 'local_num_blocks': '2', 'train_batch_order': 'global_shuffle', 'seed_values': '123', 'data_seed': 123, 'cache_dir': './data/flash_vqg', 'fox_remote_path_backend': None, 'fox_remote_read_topk_values': None, 'fox_remote_formula': 'legacy', 'fox_clr_rank': 4, 'fox_clr_use_den_residual': 'true', 'fox_clr_remat_mode': 'off', 'fox_clr_residual_update_mode': 'additive', 'fox_clr_residual_forget_mode': 'global', 'fox_clr_state_write_topk': 4, 'fox_clr_delta_target_mode': 'residual_to_coarse', 'vq_score_mode': 'l2', 'vq_weight_mode': 'one-hot', 'vq_update_mode': 'ema', 'vq_softmax_tau': 1.0, 'vq_topk': 4, 'gradient_accumulation_steps': 4, 'train_batch_size': 64, 'eval_batch_size': 16, 'metrics_white_list': None, 'metrics_white_list_file': '/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260420-gdn-default-baseline/metrics.yaml', 'project': 'flash_vqg_vs_gdn', 'entity': 'scu-mclab', 'max_epochs': 32, 'launch_id_prefix': 'flash-vqg-20260420-gdn-default-baseline', 'config_builder': '/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260420-gdn-default-baseline/config_builder.py:build_gdn_default_config', 'outdir': None, 'gpus': '0', 'parallelize': False, 'analysis': 'local'})
configs = _builder(_builder_args)
