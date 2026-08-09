import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import load_training_checkpoint, save_training_checkpoint


def _cfg():
    return SimpleNamespace(dataset=SimpleNamespace(name="synthetic"), model=SimpleNamespace(variant="mstc"), _seed=7)


def _step(model, optimizer):
    optimizer.zero_grad()
    loss = (model(torch.tensor([[1.0]])).sum() - 0.25).square()
    loss.backward()
    optimizer.step()


def test_complete_checkpoint_fields_and_restore(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _step(model, optimizer)
    checkpoint = save_training_checkpoint(model, optimizer, 3, str(tmp_path), cfg=_cfg())
    assert {"model_state_dict", "optimizer_state_dict", "epoch", "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "config_hash"} <= checkpoint.keys()
    assert (tmp_path / "checkpoint.pt").is_file()

    expected = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.data.add_(10)
    restored = load_training_checkpoint(model, str(tmp_path), optimizer=optimizer, cfg=_cfg())
    assert restored["epoch"] == 3
    assert restored["complete"] is True
    assert all(torch.equal(parameter, value) for parameter, value in zip(model.parameters(), expected))


def test_rng_state_is_really_restored(tmp_path):
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    random.random(); np.random.rand(); torch.rand(1)
    save_training_checkpoint(model, optimizer, 1, str(tmp_path), cfg=_cfg())
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(999); np.random.seed(999); torch.manual_seed(999)
    load_training_checkpoint(model, str(tmp_path), optimizer=optimizer, cfg=_cfg())
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(3)
    uninterrupted = torch.nn.Linear(1, 1)
    resumed = torch.nn.Linear(1, 1)
    resumed.load_state_dict(uninterrupted.state_dict())
    first_optimizer = torch.optim.Adam(uninterrupted.parameters(), lr=0.01)
    _step(uninterrupted, first_optimizer); _step(uninterrupted, first_optimizer)

    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
    _step(resumed, resumed_optimizer)
    save_training_checkpoint(resumed, resumed_optimizer, 1, str(tmp_path), cfg=_cfg())
    reloaded = torch.nn.Linear(1, 1)
    reloaded_optimizer = torch.optim.Adam(reloaded.parameters(), lr=0.01)
    state = load_training_checkpoint(reloaded, str(tmp_path), optimizer=reloaded_optimizer, cfg=_cfg())
    assert state["epoch"] + 1 == 2
    _step(reloaded, reloaded_optimizer)
    assert all(torch.allclose(a, b, rtol=0, atol=0) for a, b in zip(uninterrupted.parameters(), reloaded.parameters()))


def test_legacy_checkpoint_is_inference_only(tmp_path):
    model = torch.nn.Linear(1, 1)
    torch.save(model.state_dict(), tmp_path / "state_dict.pkl")
    with pytest.warns(UserWarning, match="inference"):
        state = load_training_checkpoint(torch.nn.Linear(1, 1), str(tmp_path), optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    assert state["complete"] is False
