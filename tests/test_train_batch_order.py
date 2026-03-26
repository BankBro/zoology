import torch

from zoology.data.utils import DataSegment, _BatchOrderSampler, _SyntheticDataset
from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs


def _build_dataset() -> _SyntheticDataset:
    segments = []
    for segment_idx, num_examples in enumerate([10, 4, 4]):
        inputs = torch.arange(num_examples).unsqueeze(-1)
        labels = inputs.clone()
        segments.append(
            DataSegment(
                inputs=inputs,
                labels=labels,
                slices={"mqar_case": f"seg-{segment_idx}"},
            )
        )
    return _SyntheticDataset(segments=segments, batch_size=2)


def test_sequential_sampler_matches_dataset_order():
    dataset = _build_dataset()
    sampler = _BatchOrderSampler(dataset, mode="sequential", seed=123)

    assert list(iter(sampler)) == list(range(len(dataset)))


def test_global_shuffle_is_deterministic_per_epoch_and_permutation():
    dataset = _build_dataset()
    sampler = _BatchOrderSampler(dataset, mode="global_shuffle", seed=123)

    sampler.set_epoch(0)
    epoch0 = list(iter(sampler))
    sampler.set_epoch(0)
    epoch0_repeat = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))

    assert sorted(epoch0) == list(range(len(dataset)))
    assert epoch0 == epoch0_repeat
    assert epoch0 != epoch1


def test_balanced_interleave_preserves_counts_and_front_loads_minor_segments():
    dataset = _build_dataset()
    sampler = _BatchOrderSampler(dataset, mode="balanced_interleave", seed=123)

    sampler.set_epoch(0)
    order = list(iter(sampler))
    segment_order = [dataset.batches[batch_idx][0] for batch_idx in order]

    assert sorted(order) == list(range(len(dataset)))
    assert set(segment_order[:3]) == {0, 1, 2}
    assert segment_order[:5].count(0) == 3
    assert segment_order.count(0) == 5
    assert segment_order.count(1) == 2
    assert segment_order.count(2) == 2


def test_build_configs_expands_multiple_train_batch_orders_under_one_sweep():
    configs = build_configs(
        sweep_id="flash-vqg-e0-all",
        include_gdn=False,
        dmodels=[128],
        learning_rates=[1e-3],
        max_epochs=1,
        train_batch_orders=["sequential", "global_shuffle", "balanced_interleave"],
    )

    assert len(configs) == 3
    assert {config.sweep_id for config in configs} == {"flash-vqg-e0-all"}
    assert {config.data.train_batch_order for config in configs} == {
        "sequential",
        "global_shuffle",
        "balanced_interleave",
    }
    assert {config.run_id for config in configs} == {
        "flash_vqg_h2_accel-dmodel128-lr1.0e-03-sampler-seq",
        "flash_vqg_h2_accel-dmodel128-lr1.0e-03-sampler-gshuffle",
        "flash_vqg_h2_accel-dmodel128-lr1.0e-03-sampler-binterleave",
    }


def test_build_configs_writes_logger_backend_and_preserves_default():
    default_configs = build_configs(
        include_gdn=False,
        dmodels=[128],
        learning_rates=[1e-3],
        max_epochs=1,
    )
    swanlab_configs = build_configs(
        include_gdn=False,
        dmodels=[128],
        learning_rates=[1e-3],
        max_epochs=1,
        logger_backend="swanlab",
    )

    assert len(default_configs) == 1
    assert default_configs[0].logger.backend == "wandb"
    assert len(swanlab_configs) == 1
    assert swanlab_configs[0].logger.backend == "swanlab"
