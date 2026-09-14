"""CPU behavioral checks for frozen-base refiner training and state identity."""
from collections import OrderedDict

import pytest
import torch
from torch import nn

from fsd.training_policy import base_state_digest, configure_trainable_scope


class SmallScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.refiner = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, images):
        return self.refiner(self.base(images))


def test_frozen_features_train_refiner_while_base_parameters_and_buffers_stay_exact():
    torch.random.default_generator.manual_seed(42)
    model = SmallScene().train()
    policy = configure_trainable_scope(model, "b", "refiner")
    model.base.eval()  # requires_grad alone does not freeze BN running statistics.
    before = base_state_digest(model)
    refiner_before = {n: p.detach().clone() for n, p in model.refiner.named_parameters()}
    features = model.base(torch.randn(5, 3))
    assert not features.requires_grad
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    loss = (model.refiner(features) - torch.ones(5, 2)).square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.refiner.parameters())
    assert any(torch.count_nonzero(p.grad) for p in model.refiner.parameters())
    optimizer.step()
    assert base_state_digest(model) == before
    assert any(not torch.equal(refiner_before[n], p) for n, p in model.refiner.named_parameters())
    assert policy["trainable_names"] == [n for n, _ in model.named_parameters() if n.startswith("refiner.")]
    assert policy["trainable_parameter_tensors"] == 4
    assert policy["trainable_numel"] == sum(p.numel() for p in model.refiner.parameters())
    assert policy["frozen_numel"] == sum(p.numel() for p in model.base.parameters())


def test_all_scope_restores_trainability_and_does_not_change_training_modes():
    model = SmallScene().eval()
    configure_trainable_scope(model, "b", "refiner")
    result = configure_trainable_scope(model, "a")
    assert all(p.requires_grad for p in model.parameters())
    assert not result["frozen_names"] and result["frozen_numel"] == 0
    assert not model.training and not model.base.training and not model.refiner.training


@pytest.mark.parametrize("stage,scope", [("a", "refiner"), ("unknown", "all"), ("b", "head")])
def test_invalid_policy_is_rejected_without_partial_requires_grad_mutation(stage, scope):
    model = SmallScene()
    model.base[0].weight.requires_grad_(False)
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    with pytest.raises(ValueError):
        configure_trainable_scope(model, stage, scope)
    assert before == {n: p.requires_grad for n, p in model.named_parameters()}


def test_missing_or_shared_refiner_parameters_are_rejected():
    empty = nn.Module()
    empty.base = nn.Linear(2, 2)
    empty.refiner = nn.Identity()
    with pytest.raises(ValueError, match="actual refiner parameters"):
        configure_trainable_scope(empty, "b", "refiner")
    shared = SmallScene()
    shared.refiner[0].weight = shared.base[0].weight
    before = [p.requires_grad for p in shared.parameters()]
    with pytest.raises(ValueError, match="share a parameter"):
        configure_trainable_scope(shared, "b", "refiner")
    assert before == [p.requires_grad for p in shared.parameters()]


def test_base_digest_includes_frozen_buffers_and_excludes_only_refiner_state():
    model = SmallScene().eval()
    before = base_state_digest(model)
    with torch.no_grad():
        model.refiner[0].weight.add_(1)
    assert base_state_digest(model) == before
    with torch.no_grad():
        model.base[1].running_mean.add_(1)
    assert base_state_digest(model) != before
    changed = base_state_digest(model)
    model.base[1].num_batches_tracked.add_(1)
    assert base_state_digest(model) != changed
    # A frozen BN in training mode still mutates buffers; this must be detected.
    configure_trainable_scope(model, "b", "refiner")
    model.base.train()
    changed = base_state_digest(model)
    model.base(torch.ones(4, 3))
    assert base_state_digest(model) != changed


def test_digest_is_order_and_layout_independent_but_value_shape_dtype_sensitive():
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    state = {"base.weight": matrix, "base.counter": torch.tensor(2),
             "base.bfloat": torch.tensor([1., 2.], dtype=torch.bfloat16), "refiner.extra": object()}
    before = base_state_digest(state)
    ordered = OrderedDict(reversed(list(state.items())))
    ordered["base.weight"] = matrix.T.contiguous().T
    assert not ordered["base.weight"].is_contiguous()
    assert base_state_digest(ordered) == before
    for altered in (matrix + 1, matrix.reshape(3, 2), matrix.double()):
        assert base_state_digest({**state, "base.weight": altered}) != before
    assert base_state_digest({"base.refiner.weight": matrix}) != base_state_digest({})


def test_checkpoint_roundtrip_preserves_exact_base_digest(tmp_path):
    model = SmallScene()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert base_state_digest(reloaded["model"]) == base_state_digest(model)
    with pytest.raises(TypeError, match="Non-tensor base"):
        base_state_digest({"base.extra": {"value": 1}})


def test_freezing_clears_stale_base_gradients_even_if_optimizer_contains_all_parameters():
    model = SmallScene().eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)
    for parameter in model.base.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = base_state_digest(model)
    configure_trainable_scope(model, "b", "refiner")
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step()
    assert base_state_digest(model) == before
