"""Explicit trainable scopes and exact checks of the preserved base state.

Freezing parameters does not freeze running buffers. The caller must preserve
base module evaluation behavior (SceneModel already freezes backbone BN stats)
and verify the base state digest around training/checkpoint operations.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import torch
from torch import nn


REFINER_PREFIX = "refiner."


def configure_trainable_scope(model: nn.Module, stage: str, scope: str = "all") -> dict[str, Any]:
    """Set requires_grad after validating the entire policy; do not change modes.

    Refiner-only training requires stage ``b`` and parameters under the actual
    ``refiner.`` module prefix. Aliasing a parameter between the base and refiner
    cannot preserve the base while training the refiner, so that case is rejected.
    The returned names/counts describe unique parameter tensors, not buffers.
    """
    if stage not in ("a", "b"):
        raise ValueError(f"Unknown training stage: {stage!r}")
    if scope not in ("all", "refiner"):
        raise ValueError(f"Unknown trainable scope: {scope!r}")
    named = list(model.named_parameters())
    if scope == "refiner":
        if stage != "b":
            raise ValueError("Refiner-only training is allowed only in stage b")
        if not any(name.startswith(REFINER_PREFIX) for name, _ in named):
            raise ValueError("Refiner-only training requires actual refiner parameters")
        aliases: dict[int, set[bool]] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), set()).add(name.startswith(REFINER_PREFIX))
        if any(len(groups) > 1 for groups in aliases.values()):
            raise ValueError("Base and refiner share a parameter; the base cannot remain frozen")
    trainable, frozen = [], []
    for name, parameter in named:
        enabled = scope == "all" or name.startswith(REFINER_PREFIX)
        parameter.requires_grad_(enabled)
        if not enabled:
            parameter.grad = None  # A pre-existing optimizer must not apply stale base gradients.
        (trainable if enabled else frozen).append((name, parameter))
    return {
        "stage": stage, "scope": scope,
        "trainable_names": [name for name, _ in trainable],
        "frozen_names": [name for name, _ in frozen],
        "trainable_parameter_tensors": len(trainable),
        "frozen_parameter_tensors": len(frozen),
        "trainable_numel": sum(parameter.numel() for _, parameter in trainable),
        "frozen_numel": sum(parameter.numel() for _, parameter in frozen),
    }


def base_state_digest(model_or_state: nn.Module | Mapping[str, torch.Tensor]) -> str:
    """SHA256 of non-refiner parameters AND buffers, independent of device/layout.

    Include sorted names, shapes, dtypes and exact value bytes. Reinterpret bytes
    through uint8 so bfloat16 and scalar integer buffers are supported without a
    numerical conversion. This neither initializes CUDA nor queries a GPU; an
    existing device tensor is copied to CPU only when needed for its bytes.
    Non-tensor extra state and non-dense/quantized base state are rejected rather
    than silently omitted. Call between updates, not during concurrent mutation.
    """
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256(b"fsd-base-state-v1\0")
    for name in sorted(state):
        if name.startswith(REFINER_PREFIX):
            continue
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor base state is not supported: {name}")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"Only dense non-quantized base state is supported: {name}")
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("utf8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()
