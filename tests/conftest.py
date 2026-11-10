from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def global_datadir():
    return Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(autouse=True)
def restore_torch_state():
    """Isolate global torch settings used by the scientific regression tests."""
    dtype, grad = torch.get_default_dtype(), torch.is_grad_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_default_dtype(dtype)
    torch.set_grad_enabled(grad)
    torch.set_num_threads(threads)
