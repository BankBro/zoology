import importlib.util
from pathlib import Path

import torch

from zoology.data.multiquery_ar import multiquery_ar
from zoology.data.utils import DataSegment, _SyntheticDataset


def test_mqar_retrieval_supervision_points_to_source_keys():
    segment = multiquery_ar(
        vocab_size=128,
        num_examples=4,
        input_seq_len=32,
        seed=123,
        num_kv_pairs=4,
        random_non_queries=False,
        include_retrieval_supervision=True,
    )

    ret_source_pos = segment.extra_tensors["ret_source_pos"]
    query_mask = segment.labels != -100

    assert ret_source_pos.shape == segment.inputs.shape
    assert torch.equal(ret_source_pos >= 0, query_mask)
    for example_idx in range(segment.inputs.size(0)):
        for query_pos in query_mask[example_idx].nonzero(as_tuple=False).flatten():
            source_pos = int(ret_source_pos[example_idx, query_pos].item())
            assert source_pos % 2 == 0
            assert source_pos < 2 * 4
            assert source_pos < int(query_pos.item())
            assert segment.inputs[example_idx, source_pos] == segment.inputs[example_idx, query_pos]
            assert segment.labels[example_idx, query_pos] == segment.inputs[example_idx, source_pos + 1]


def test_synthetic_dataset_keeps_legacy_batches_without_extras():
    inputs = torch.arange(6).view(3, 2)
    labels = inputs.clone()
    dataset = _SyntheticDataset(
        [DataSegment(inputs=inputs, labels=labels, slices={"case": "legacy"})],
        batch_size=2,
    )

    batch = dataset[0]

    assert len(batch) == 3


def test_synthetic_dataset_returns_extra_tensors_when_present():
    inputs = torch.arange(6).view(3, 2)
    labels = inputs.clone()
    extras = {"ret_source_pos": torch.full_like(inputs, -1)}
    dataset = _SyntheticDataset(
        [DataSegment(inputs=inputs, labels=labels, slices={"case": "ret"}, extra_tensors=extras)],
        batch_size=2,
    )

    batch = dataset[0]

    assert len(batch) == 4
    assert torch.equal(batch[3]["ret_source_pos"], extras["ret_source_pos"][:2])


def test_e5_builder_emits_four_locked_runs():
    builder_path = (
        Path(__file__).parents[1]
        / "zoology/experiments/flash_vqg/scripts/20260402-clr-v1-mainline/e5-retrieval-aware/config_builder.py"
    )
    spec = importlib.util.spec_from_file_location("e5_config_builder_for_test", builder_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class Args:
        backend = "torch"
        logger_backend = "none"
        dmodels = "128"
        learning_rates = "1e-3"
        seed_values = "123"
        num_codebook_vectors = "128"
        metrics_white_list = None
        metrics_white_list_file = None
        launch_id_prefix = "e5-test"
        train_batch_order = "global_shuffle"
        data_seed = 123
        fox_remote_path_backend = "torch"
        fox_clr_rank = 4
        fox_clr_use_den_residual = "true"
        fox_clr_remat_mode = "off"
        gradient_accumulation_steps = 1
        train_batch_size = 2
        eval_batch_size = 2
        cache_dir = None
        project = "test"
        entity = "test"
        max_epochs = 1
        vq_topk = 4

    configs = module.build_e5_train_configs(Args())

    assert [cfg.run_id for cfg in configs] == [
        "dense-t025-retoff-s123-d123",
        "dense-t025-retl002-t050-s123-d123",
        "dense-t025-retl002-t100-s123-d123",
        "dense-t025-retl005-t100-s123-d123",
    ]
    enabled = [
        cfg.model.sequence_mixer.kwargs["configs"][-1]["kwargs"]["retrieval_loss_enabled"]
        for cfg in configs
    ]
    assert enabled == [False, True, True, True]

    reton_configs = module.build_e5_reton_train_configs(Args())
    assert [cfg.run_id for cfg in reton_configs] == [
        "dense-t025-retl002-t050-s123-d123",
        "dense-t025-retl002-t100-s123-d123",
        "dense-t025-retl005-t100-s123-d123",
    ]
