import pytest
import torch

from zoology.mixers import gated_delta_net as gdn


def test_gated_delta_kernel_dtype_auto_cpu(monkeypatch):
    monkeypatch.delenv("GDN_KERNEL_DTYPE", raising=False)

    assert gdn._gated_delta_kernel_dtype(torch.device("cpu")) == torch.float32


def test_gated_delta_kernel_dtype_auto_cuda_sm75(monkeypatch):
    monkeypatch.delenv("GDN_KERNEL_DTYPE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    assert gdn._gated_delta_kernel_dtype(torch.device("cuda")) == torch.float16


def test_gated_delta_kernel_dtype_auto_cuda_sm80(monkeypatch):
    monkeypatch.delenv("GDN_KERNEL_DTYPE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    assert gdn._gated_delta_kernel_dtype(torch.device("cuda")) == torch.bfloat16


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("input", None),
        ("keep", None),
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("half", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
    ],
)
def test_gated_delta_kernel_dtype_env_overrides(monkeypatch, policy, expected):
    monkeypatch.setenv("GDN_KERNEL_DTYPE", policy)

    assert gdn._gated_delta_kernel_dtype(torch.device("cuda")) == expected


def test_gated_delta_kernel_dtype_invalid(monkeypatch):
    monkeypatch.setenv("GDN_KERNEL_DTYPE", "float64")

    with pytest.raises(ValueError, match="GDN_KERNEL_DTYPE"):
        gdn._gated_delta_kernel_dtype(torch.device("cuda"))


def test_maybe_cast_gated_delta_kernel_inputs_input_policy(monkeypatch):
    monkeypatch.setenv("GDN_KERNEL_DTYPE", "input")
    q = torch.ones(1, dtype=torch.float32)
    k = torch.ones(1, dtype=torch.float32)

    q_out, k_out = gdn._maybe_cast_gated_delta_kernel_inputs(q, k)

    assert q_out is q
    assert k_out is k


def test_maybe_cast_gated_delta_kernel_inputs_explicit_policy(monkeypatch):
    monkeypatch.setenv("GDN_KERNEL_DTYPE", "float16")
    q = torch.ones(1, dtype=torch.float32)
    k = torch.ones(1, dtype=torch.float32)

    q_out, k_out = gdn._maybe_cast_gated_delta_kernel_inputs(q, k)

    assert q_out.dtype == torch.float16
    assert k_out.dtype == torch.float16
