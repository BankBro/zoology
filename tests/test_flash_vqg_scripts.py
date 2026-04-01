from zoology.experiments.flash_vqg.scripts.compare_remat_training import _build_config as build_remat_config
from zoology.experiments.flash_vqg.scripts.smoke_clr_oom_grid import _build_one_config as build_smoke_config


def test_compare_remat_training_build_config_supports_batch_and_gradient_accumulation():
    config = build_remat_config(
        "post_phase1",
        train_batch_size=16,
        eval_batch_size=24,
        gradient_accumulation_steps=8,
    )

    assert config.data.batch_size == (16, 24)
    assert config.gradient_accumulation_steps == 8
    assert config.run_id.endswith("-rformula-clr1-r4-den1-rremat-postp1-tbs16-ebs24-ga8")


def test_smoke_clr_oom_grid_build_config_supports_batch_and_gradient_accumulation():
    config = build_smoke_config(
        4,
        True,
        "off",
        train_batch_size=32,
        eval_batch_size=12,
        gradient_accumulation_steps=4,
    )

    assert config.data.batch_size == (32, 12)
    assert config.gradient_accumulation_steps == 4
    assert config.run_id.endswith("-rformula-clr1-r4-den1-rremat-off-tbs32-ebs12-ga4")
