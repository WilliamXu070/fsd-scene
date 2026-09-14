"""Camera-only scene network with depth-informed BEV and late box refinement.

The model never reads targets. Geometry conventions are documented in
``docs/execution/CONTRACTS.md``: ego x forward, y left, z up; radial depth;
box geometric centers. Temporal state is detached and belongs to one sequence.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import FeaturePyramidNetwork

from .geometry import box_corners, project_points_torch


def model_config(config: dict | None) -> dict:
    """Accept the shared nested configuration or a model-only dictionary."""
    config = config or {}
    return config.get("model", config)


def _groups(channels: int) -> int:
    return next(g for g in (32, 16, 8, 4, 2, 1) if channels % g == 0)


def _conv(in_channels: int, out_channels: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualBEVBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x + self.net(x))


class ImageEncoder(nn.Module):
    """Shared ResNet-50 and FPN outputs at strides 8, 16, 32."""

    def __init__(self, channels: int = 128, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        # Download/load errors are intentional failures, never random-weight fallback.
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.fpn = FeaturePyramidNetwork([512, 1024, 2048], channels)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self._freeze_bn()

    def _freeze_bn(self) -> None:
        # Running statistics must not drift in single-sequence microbatches.
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_bn()
        return self

    def forward(self, images: Tensor) -> list[Tensor]:
        b, v, c, h, w = images.shape
        x = images.reshape(b * v, c, h, w)
        x = (x - self.mean) / self.std
        trunk = self.backbone
        x = trunk.maxpool(trunk.relu(trunk.bn1(trunk.conv1(x))))
        x = trunk.layer1(x)
        c3 = trunk.layer2(x)
        c4 = trunk.layer3(c3)
        c5 = trunk.layer4(c4)
        features = self.fpn(OrderedDict(p3=c3, p4=c4, p5=c5))
        return [f.reshape(b, v, *f.shape[1:]) for f in features.values()]


class DepthLift(nn.Module):
    """Differentiable weighted pooling without a retained dense frustum volume.

    Pixel rays are calibrated unit vectors; each bin is a radial range. The
    scatter indices are geometry-only. Image context and depth probabilities
    both retain gradients. FP32 accumulation avoids half-precision atomic drift.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.size = int(config.get("bev_size", 160))
        self.minimum = float(config.get("bev_min", -40.0))
        self.step = float(config.get("bev_step", 0.5))
        self.chunk_size = int(config.get("depth_chunk", 8))
        self.register_buffer(
            "depths",
            torch.linspace(config.get("depth_min", 1.0), config.get("depth_max", 80.0),
                           int(config.get("depth_bins", 64))),
        )

    def forward(self, context: Tensor, depth_logits: Tensor, batch: dict) -> Tensor:
        b, v, channels, h, w = context.shape
        rays = batch["rays"].to(device=context.device, dtype=torch.float32)
        if rays.shape != (b, v, h, w, 3):
            raise ValueError(f"Stride-8 rays {tuple(rays.shape)} do not match features {(b,v,h,w,3)}")
        valid_ray = batch["ray_valid"].bool() & batch["camera_valid"].bool()[:, :, None, None]
        transforms = batch["camera_to_ego"].float()
        probabilities = depth_logits.float().softmax(dim=2)
        output = context.new_zeros((b * self.size * self.size, channels), dtype=torch.float32)
        mass = context.new_zeros((b * self.size * self.size, 1), dtype=torch.float32)
        flat_context = context.permute(0, 1, 3, 4, 2).reshape(b * v * h * w, channels)
        ray_ids = torch.arange(b * v * h * w, device=context.device).reshape(b, v, h, w)
        batch_ids = torch.arange(b, device=context.device)[:, None, None, None, None]
        # Calibration must stay FP32 even under the caller's autocast context.
        with torch.autocast(device_type=context.device.type, enabled=False):
            for start in range(0, self.depths.numel(), self.chunk_size):
                bins = self.depths[start:start + self.chunk_size]
                camera = rays[:, :, None] * bins[None, None, :, None, None, None]
                ego = torch.einsum("bvij,bvdhwj->bvdhwi", transforms[:, :, :3, :3], camera)
                ego = ego + transforms[:, :, None, None, None, :3, 3]
                col = torch.floor((ego[..., 0] - self.minimum) / self.step).long()
                row = torch.floor((ego[..., 1] - self.minimum) / self.step).long()
                valid = (valid_ray[:, :, None] & torch.isfinite(ego).all(-1)
                         & (col >= 0) & (col < self.size) & (row >= 0) & (row < self.size))
                indices = (batch_ids * (self.size * self.size) + row * self.size + col)[valid]
                source = ray_ids[:, :, None].expand(b, v, len(bins), h, w)[valid]
                weights = probabilities[:, :, start:start + len(bins)][valid, None]
                weighted = flat_context[source].float() * weights
                output = output.index_add(0, indices, weighted)
                mass = mass.index_add(0, indices, weights)
            # Normalize excessive overlap while retaining probability amplitude
            # at sparsely supported cells. This also keeps camera count stable.
            output = output / mass.clamp_min(1.0)
        return output.reshape(b, self.size, self.size, channels).permute(0, 3, 1, 2).contiguous()


def gather_map(feature: Tensor, flat_indices: Tensor) -> Tensor:
    """Read channels at selected spatial cells: BCHW, BK -> BKC."""
    return feature.flatten(2).transpose(1, 2).gather(
        1, flat_indices[..., None].expand(-1, -1, feature.shape[1]))


def decode_proposals(center_logits: Tensor, geometry: Tensor, features: Tensor,
                     config: dict) -> dict[str, Tensor]:
    """Local center peaks select geometry predictions, never ground-truth boxes."""
    b, classes, h, w = center_logits.shape
    k = min(int(config.get("num_proposals", 200)), classes * h * w)
    scores = center_logits.sigmoid()
    maxima = F.max_pool2d(scores, 3, stride=1, padding=1)
    peak_scores = scores.masked_fill(scores != maxima, -1.0)
    selected_scores, indices = peak_scores.flatten(1).topk(k, dim=1)
    labels = indices // (h * w)
    cells = indices % (h * w)
    raw = gather_map(geometry, cells)
    offsets = raw[..., :2].sigmoid()
    minimum, step = float(config.get("bev_min", -40.0)), float(config.get("bev_step", 0.5))
    x = minimum + ((cells % w).to(raw.dtype) + offsets[..., 0]) * step
    y = minimum + ((cells // w).to(raw.dtype) + offsets[..., 1]) * step
    dimensions = raw[..., 3:6].clamp(-4.0, 4.0).exp()
    # FP16 reciprocal overflow in atan2 backward can turn even zero gradients into NaNs.
    with torch.autocast(device_type=raw.device.type, enabled=False):
        yaw = torch.atan2(raw[..., 6].float(), (raw[..., 7] + 1e-7).float()).to(raw.dtype)
    boxes = torch.cat([x[..., None], y[..., None], raw[..., 2:3], dimensions, yaw[..., None]], dim=-1)
    return {
        "boxes": boxes, "logits": gather_map(center_logits, cells),
        "features": gather_map(features, cells), "cells": cells,
        "labels": labels, "peak_valid": selected_scores >= 0,
    }


def align_history_boxes(boxes: Tensor, previous_pose: Tensor, current_pose: Tensor) -> Tensor:
    """Transform previous ego-frame boxes into current ego coordinates."""
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        transform = torch.linalg.solve(current_pose.float(), previous_pose.float())
        centers = torch.einsum("bij,bkj->bki", transform[:, :3, :3], boxes[..., :3].float())
        centers = centers + transform[:, None, :3, 3]
        direction = torch.stack([boxes[..., 6].cos(), boxes[..., 6].sin(),
                                 torch.zeros_like(boxes[..., 6])], dim=-1).float()
        direction = torch.einsum("bij,bkj->bki", transform[:, :3, :3], direction)
        yaw = torch.atan2(direction[..., 1], direction[..., 0])
        return torch.cat([centers, boxes[..., 3:6].float(), yaw[..., None]], dim=-1)


def _box_encoding(boxes: Tensor) -> Tensor:
    return torch.cat([boxes[..., :3] / 40.0, boxes[..., 3:6].clamp_min(1e-3).log(),
                      boxes[..., 6:7].sin(), boxes[..., 6:7].cos()], dim=-1)


class ObjectRefinement(nn.Module):
    """One late refinement block: sparse image attention then temporal attention."""

    def __init__(self, config: dict):
        super().__init__()
        width = int(config.get("bev_channels", 128))
        image_width = int(config.get("fpn_channels", 128))
        heads = int(config.get("attention_heads", 4))
        self.history_length = int(config.get("history_length", 3))
        self.max_gap = float(config.get("max_history_seconds", 2.0))
        self.image_projection = nn.Linear(image_width, width)
        self.position = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, width))
        self.level_embedding = nn.Parameter(torch.zeros(3, width))
        self.point_embedding = nn.Parameter(torch.zeros(9, width))
        self.spatial_attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.temporal_attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.spatial_norm = nn.LayerNorm(width)
        self.temporal_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.ReLU(), nn.Linear(width * 2, width))
        self.final_norm = nn.LayerNorm(width)
        self.box_delta = nn.Linear(width, 7)
        self.class_delta = nn.Linear(width, 2)
        nn.init.normal_(self.box_delta.weight, std=0.001)
        nn.init.zeros_(self.box_delta.bias)
        nn.init.normal_(self.class_delta.weight, std=0.001)
        nn.init.zeros_(self.class_delta.bias)

    def sample_images(self, boxes: Tensor, levels: list[Tensor], batch: dict) -> tuple[Tensor, Tensor]:
        b, k, _ = boxes.shape
        points = torch.cat([boxes[..., None, :3], box_corners(boxes)], dim=-2)
        # Projection/camera calibration is deliberately outside mixed precision.
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            grid, valid = project_points_torch(points.float().reshape(b, k * 9, 3), batch)
        valid = valid & batch["camera_valid"].bool()[..., None]
        v = grid.shape[1]
        tokens, masks = [], []
        for level_index, feature in enumerate(levels):
            _, _, c, h, w = feature.shape
            sampled = F.grid_sample(feature.reshape(b * v, c, h, w),
                                    grid.reshape(b * v, k * 9, 1, 2).to(feature.dtype),
                                    mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.reshape(b, v, c, k, 9).permute(0, 3, 1, 4, 2)
            projected = self.image_projection(sampled)
            projected = projected + self.level_embedding[level_index] + self.point_embedding[None, None, None]
            tokens.append(projected.reshape(b, k, v * 9, -1))
            masks.append(valid.reshape(b, v, k, 9).permute(0, 2, 1, 3).reshape(b, k, v * 9))
        return torch.cat(tokens, dim=2), torch.cat(masks, dim=2)

    @staticmethod
    def _safe_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        """MHA with entirely masked keys is undefined; bypass it for those rows."""
        has_evidence = valid.any(dim=-1)
        safe = valid.clone()
        safe[~has_evidence, 0] = True
        return ~safe, has_evidence

    def temporal_memory(self, history: list[dict], batch: dict, dtype: torch.dtype) -> tuple[Tensor | None, Tensor | None]:
        memories, masks = [], []
        b = len(batch["sequence"])
        for state in history[-self.history_length:] if self.history_length else []:
            if state["features"].shape[0] != b:
                continue
            boxes = align_history_boxes(state["boxes"].to(batch["ego_to_world"].device),
                                        state["ego_to_world"].to(batch["ego_to_world"].device),
                                        batch["ego_to_world"])
            # Subtract before FP32 conversion to preserve subsecond Unix intervals.
            elapsed = (batch["timestamp"].double() - state["timestamp"].to(batch["timestamp"].device).double()).float()
            same = torch.tensor([a == p for a, p in zip(batch["sequence"], state["sequence"])],
                                device=boxes.device, dtype=torch.bool)
            compatible = same & (elapsed > 0) & (elapsed <= self.max_gap)
            stored_valid = state.get("valid", torch.ones(boxes.shape[:2], device=boxes.device, dtype=torch.bool)).to(boxes.device)
            valid = compatible[:, None] & stored_valid & torch.isfinite(boxes).all(dim=-1)
            feature = state["features"].to(device=boxes.device, dtype=dtype)
            tokens = (feature + self.position(_box_encoding(boxes).to(dtype))
                      + self.time(elapsed[:, None, None].to(dtype)))
            memories.append(tokens)
            masks.append(valid)
        if not memories:
            return None, None
        return torch.cat(memories, dim=1), torch.cat(masks, dim=1)

    def forward(self, proposals: dict, levels: list[Tensor], batch: dict,
                history: list[dict] | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        boxes, logits = proposals["boxes"], proposals["logits"]
        b, k, _ = boxes.shape
        query = proposals["features"] + self.position(_box_encoding(boxes).to(proposals["features"].dtype))
        tokens, valid = self.sample_images(boxes, levels, batch)
        valid = valid & proposals["peak_valid"][..., None]
        key_mask, spatial_valid = self._safe_mask(valid.reshape(b * k, -1))
        q = query.reshape(b * k, 1, -1)
        image_tokens = tokens.reshape(b * k, tokens.shape[2], tokens.shape[3])
        attended, _ = self.spatial_attention(q, image_tokens, image_tokens, key_padding_mask=key_mask, need_weights=False)
        attended = attended.reshape(b, k, -1) * spatial_valid.reshape(b, k, 1)
        query = self.spatial_norm(query + attended)
        memory, memory_valid = self.temporal_memory(history or [], batch, query.dtype)
        temporal_update = torch.zeros_like(query)
        if memory is not None:
            key_mask, temporal_valid = self._safe_mask(memory_valid)
            attended, _ = self.temporal_attention(query, memory, memory, key_padding_mask=key_mask, need_weights=False)
            temporal_update = attended * temporal_valid[:, None, None]
        query = self.temporal_norm(query + temporal_update)
        query = self.final_norm(query + self.ffn(query))
        delta = self.box_delta(query)
        new_center = boxes[..., :3] + delta[..., :3]
        new_size = boxes[..., 3:6] * delta[..., 3:6].clamp(-1.5, 1.5).exp()
        new_yaw = boxes[..., 6] + delta[..., 6]
        new_yaw = torch.atan2(new_yaw.sin(), new_yaw.cos())
        refined = torch.cat([new_center, new_size, new_yaw[..., None]], dim=-1)
        spatial_valid = spatial_valid.reshape(b, k)
        # No valid image evidence: preserve the proposal exactly as agreed.
        refined = torch.where(spatial_valid[..., None], refined, boxes)
        refined_logits = torch.where(spatial_valid[..., None], logits + self.class_delta(query), logits)
        return refined, refined_logits, query, spatial_valid


class SceneModel(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = model_config(config)
        c = self.config
        fpn = int(c.get("fpn_channels", 128))
        context = int(c.get("context_channels", 64))
        bev = int(c.get("bev_channels", 128))
        depth = int(c.get("depth_bins", 64))
        self.image_encoder = ImageEncoder(fpn, bool(c.get("pretrained", True)))
        self.depth_head = nn.Sequential(_conv(fpn, fpn), nn.Conv2d(fpn, depth, 1))
        self.context_head = nn.Conv2d(fpn, context, 1)
        self.lift = DepthLift(c)
        self.bev_encoder = nn.Sequential(_conv(context, bev), *(ResidualBEVBlock(bev) for _ in range(3)))
        self.road_head = nn.Sequential(_conv(bev, bev), nn.Conv2d(bev, 3, 1))
        self.center_head = nn.Sequential(_conv(bev, bev), nn.Conv2d(bev, 2, 1))
        self.geometry_head = nn.Sequential(_conv(bev, bev), nn.Conv2d(bev, 8, 1))
        nn.init.constant_(self.center_head[-1].bias, -2.19)
        nn.init.normal_(self.geometry_head[-1].weight, std=0.001)
        with torch.no_grad():
            self.geometry_head[-1].bias.copy_(torch.tensor([0., 0., 1., math.log(4.), math.log(1.8), math.log(1.5), 0., 1.]))
        self.refiner = ObjectRefinement(c)

    def forward(self, batch: dict, history: list[dict] | None = None, refine: bool = True) -> dict[str, Any]:
        features = self.image_encoder(batch["images"])
        b, v, channels, h, w = features[0].shape
        image_feature = features[0].reshape(b * v, channels, h, w)
        depth = self.depth_head(image_feature).reshape(b, v, -1, h, w)
        context = self.context_head(image_feature).reshape(b, v, -1, h, w)
        lifted = self.lift(context, depth, batch)
        bev = self.bev_encoder(lifted.to(context.dtype))
        road, centers, geometry = self.road_head(bev), self.center_head(bev), self.geometry_head(bev)
        proposals = decode_proposals(centers, geometry, bev, self.config)
        if refine:
            boxes, logits, query, spatial_valid = self.refiner(proposals, features, batch, history)
        else:
            boxes, logits, query = proposals["boxes"], proposals["logits"], proposals["features"]
            with torch.no_grad():
                _, valid = project_points_torch(boxes[..., :3].float(), batch)
                spatial_valid = (valid & batch["camera_valid"].bool()[..., None]).any(dim=1) & proposals["peak_valid"]
        state = {
            "features": query.detach(), "boxes": boxes.detach(), "logits": logits.detach(),
            "valid": spatial_valid.detach(), "ego_to_world": batch["ego_to_world"].detach().clone(),
            "timestamp": batch["timestamp"].detach().clone(), "sequence": list(batch["sequence"]),
        }
        return {
            "road_logits": road, "depth_logits": depth, "center_logits": centers,
            "geometry_map": geometry, "initial_boxes": proposals["boxes"],
            "initial_logits": proposals["logits"], "refined_boxes": boxes, "refined_logits": logits,
            "proposal_features": query, "proposal_cells": proposals["cells"],
            "proposal_labels": proposals["labels"], "spatial_valid": spatial_valid,
            "state": state,
        }
