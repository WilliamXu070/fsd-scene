"""Masked, metric-geometry supervision for the scene model.

All supervision is annotation driven. Unknown depth, ground, and detection
regions never become background labels. Hungarian assignment uses predicted
proposals, not teacher-provided centers at inference.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import Tensor
import torch.nn.functional as F

from .model import model_config
from .metrics import pairwise_iou


def _zero(outputs: dict) -> Tensor:
    return outputs["center_logits"].float().sum() * 0.0


def _gaussian_radius(length: float, width: float, overlap: float = 0.7) -> int:
    # CenterNet's minimum-overlap construction in grid units.
    a1, b1, c1 = 1.0, length + width, width * length * (1 - overlap) / (1 + overlap)
    r1 = (b1 + math.sqrt(max(0., b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2, c2 = 4., 2 * (length + width), (1 - overlap) * width * length
    r2 = (b2 + math.sqrt(max(0., b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3, c3 = 4 * overlap, -2 * overlap * (length + width), (overlap - 1) * width * length
    r3 = (b3 + math.sqrt(max(0., b3 * b3 - 4 * a3 * c3))) / 2
    return max(0, int(min(r1, r2, r3)))


def _draw_gaussian(heatmap: Tensor, row: int, col: int, radius: int) -> None:
    h, w = heatmap.shape
    radius = min(radius, max(h, w))
    top, bottom = max(0, row - radius), min(h, row + radius + 1)
    left, right = max(0, col - radius), min(w, col + radius + 1)
    y = torch.arange(top, bottom, device=heatmap.device, dtype=torch.float32) - row
    x = torch.arange(left, right, device=heatmap.device, dtype=torch.float32) - col
    sigma = (radius * 2 + 1) / 6
    kernel = torch.exp(-(y[:, None].square() + x[None].square()) / (2 * sigma * sigma))
    patch = heatmap[top:bottom, left:right]
    patch.copy_(torch.maximum(patch, kernel))


def make_center_targets(outputs: dict, batch: dict, config: dict) -> tuple[Tensor, list[tuple[int, Tensor, Tensor, Tensor]]]:
    """Heatmap plus valid per-object (batch, cells, offsets, boxes) records."""
    heatmap = torch.zeros_like(outputs["center_logits"], dtype=torch.float32)
    _, _, h, w = heatmap.shape
    c = model_config(config)
    step, minimum = float(c.get("bev_step", .5)), float(c.get("bev_min", -40.))
    records = []
    for b, (boxes, labels) in enumerate(zip(batch.get("boxes", []), batch.get("labels", []))):
        boxes, labels = boxes.to(heatmap.device).float(), labels.to(heatmap.device).long()
        if not len(boxes):
            continue
        xy = (boxes[:, :2] - minimum) / step
        valid = (torch.isfinite(boxes).all(-1) & (boxes[:, 3:6] > 0).all(-1)
                 & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
                 & (labels >= 0) & (labels < heatmap.shape[1]))
        boxes, labels, xy = boxes[valid], labels[valid], xy[valid]
        if not len(boxes):
            continue
        cells = xy.floor().long()
        for box, label, cell in zip(boxes.detach().cpu(), labels.detach().cpu(), cells.detach().cpu()):
            radius = _gaussian_radius(float(box[3] / step), float(box[4] / step))
            _draw_gaussian(heatmap[b, int(label)], int(cell[1]), int(cell[0]), radius)
        records.append((b, cells[:, 1] * w + cells[:, 0], xy - cells.float(), boxes))
    return heatmap, records


def center_focal_loss(logits: Tensor, target: Tensor, valid: Tensor,
                      positive_normalization: str = "global", return_stats: bool = False
                      ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Optionally balance positive evidence; known-negative pressure is unchanged.

    Counts are unique positive heatmap cells per class across this microbatch,
    not raw boxes or the accumulated optimizer batch. Diagnostic sums/counts are
    detached and unnormalized; they never contribute an extra loss term.
    """
    if not isinstance(positive_normalization, str) or positive_normalization not in ("global", "present_class"):
        raise ValueError("center_positive_normalization must be 'global' or 'present_class'")
    probabilities = logits.float().sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = target.eq(1)
    # Annotated centers override an incomplete negative-validity mask.
    known = valid.bool() | positive
    negative = target.lt(1) & known
    positive_terms = probabilities.log() * (1 - probabilities).square() * positive
    negative_terms = ((1 - probabilities).log() * probabilities.square()
                      * (1 - target).pow(4) * negative)
    positive_loss = -positive_terms.sum()
    negative_loss = -negative_terms.sum()
    # Keep this expression/order for default, no-positive and single-class cases.
    loss = (positive_loss + negative_loss) / positive.sum().clamp_min(1)
    stats = {}
    if positive_normalization == "present_class" or return_stats:
        reduce_dims = (0, *range(2, logits.ndim))
        positive_count = positive.sum(dim=reduce_dims)
        positive_sum = -positive_terms.sum(dim=reduce_dims)
        present = positive_count > 0
        if positive_normalization == "present_class" and int(present.sum()) > 1:
            balanced_positive = (positive_sum[present] / positive_count[present]).mean()
            loss = balanced_positive + negative_loss / positive.sum().clamp_min(1)
        if return_stats:
            stats = {"positive_sum": positive_sum.detach(),
                     "negative_sum": -negative_terms.detach().sum(dim=reduce_dims),
                     "positive_count": positive_count.detach(),
                     "negative_count": negative.sum(dim=reduce_dims).detach()}
    return (loss, stats) if return_stats else loss


def masked_class_focal(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    p = logits.float().sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    alpha = .25 * target + .75 * (1 - target)
    weighted = ce * (1 - pt).square() * alpha * valid
    return weighted.sum() / (target * valid).sum().clamp_min(1)


def _eligible_targets(batch: dict, b: int, output: Tensor, config: dict) -> tuple[Tensor, Tensor]:
    boxes = batch.get("boxes", [])[b].to(output.device).float()
    labels = batch.get("labels", [])[b].to(output.device).long()
    c = model_config(config)
    minimum, extent = float(c.get("bev_min", -40.)), float(c.get("bev_step", .5)) * int(c.get("bev_size", 160))
    keep = (torch.isfinite(boxes).all(-1) & (boxes[:, 3:6] > 0).all(-1)
            & (boxes[:, :2] >= minimum).all(-1) & (boxes[:, :2] < minimum + extent).all(-1)
            & (labels >= 0) & (labels < 2))
    return boxes[keep], labels[keep]


def _proposal_validity(boxes: Tensor, batch: dict, b: int, config: dict) -> Tensor:
    """Class-specific known negatives sampled at each predicted proposal center."""
    valid = torch.zeros((len(boxes), 2), device=boxes.device, dtype=torch.bool)
    if "detection_valid" not in batch:
        return valid
    region = batch["detection_valid"][b].to(boxes.device).bool()
    c = model_config(config)
    xy = ((boxes[:, :2].detach() - float(c.get("bev_min", -40.))) / float(c.get("bev_step", .5))).floor().long()
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] < region.shape[2])
              & (xy[:, 1] >= 0) & (xy[:, 1] < region.shape[1]))
    valid[inside] = region[:, xy[inside, 1], xy[inside, 0]].T
    return valid


@torch.no_grad()
def match_boxes(boxes: Tensor, logits: Tensor, targets: Tensor, labels: Tensor,
                match_bev_iou_weight: float = 0.0) -> tuple[Tensor, Tensor]:
    """One-to-one matching; optional rotated BEV overlap changes assignment only."""
    if (isinstance(match_bev_iou_weight, bool) or not isinstance(match_bev_iou_weight, Real)
            or not math.isfinite(match_bev_iou_weight) or match_bev_iou_weight < 0
            or match_bev_iou_weight > float(np.finfo(np.float32).max)):
        raise ValueError("match_bev_iou_weight must be a finite nonnegative number within float32 range")
    if not len(boxes) or not len(targets):
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        return empty, empty
    pred = boxes.float()
    truth = targets.float()
    if not all(torch.isfinite(value).all() for value in (pred, truth, logits)):
        raise FloatingPointError("Non-finite proposal assignment input; stop and diagnose model outputs")
    position_cost = torch.cdist(pred[:, :3], truth[:, :3], p=1) / 5.
    size_cost = torch.cdist(pred[:, 3:6].clamp_min(1e-3).log(), truth[:, 3:6].clamp_min(1e-3).log(), p=1)
    angle_cost = 1 - torch.cos(pred[:, None, 6] - truth[None, :, 6])
    class_cost = -logits.float().sigmoid()[:, labels]
    cost = position_cost + .5 * size_cost + .2 * angle_cost + 2 * class_cost
    cost_array = cost.cpu().numpy()
    if not np.isfinite(cost_array).all():
        raise FloatingPointError("Non-finite proposal assignment cost; stop and diagnose model outputs")
    if match_bev_iou_weight > 0:
        # Assignment has no gradient. Reuse evaluation's rotated-box geometry,
        # while leaving the zero-weight cost expression and solver input intact.
        overlap = pairwise_iou(pred.cpu().numpy(), truth.cpu().numpy(), mode="bev")
        if not np.isfinite(overlap).all():
            raise FloatingPointError("Non-finite BEV overlap in proposal assignment")
        cost_array = cost_array + np.float32(match_bev_iou_weight) * (1 - overlap.astype(np.float32))
        if not np.isfinite(cost_array).all():
            raise FloatingPointError("Non-finite overlap-weighted proposal assignment cost")
    rows, cols = linear_sum_assignment(cost_array)
    return (torch.as_tensor(rows, device=boxes.device, dtype=torch.long),
            torch.as_tensor(cols, device=boxes.device, dtype=torch.long))


def compute_losses(outputs: dict, batch: dict, config: dict | None = None, refine: bool = True) -> dict[str, Tensor]:
    config = config or {}
    c = model_config(config)
    weights = config.get("loss", {})
    zero = _zero(outputs)
    result = {}
    depth_target = batch.get("depth_target", batch.get("depth"))
    if depth_target is not None:
        target = depth_target.to(outputs["depth_logits"].device).long()
        valid = target >= 0
        if "camera_valid" in batch:
            valid = valid & batch["camera_valid"].bool()[:, :, None, None]
        if "ray_valid" in batch:
            valid = valid & batch["ray_valid"].bool()
        logits = outputs["depth_logits"].float().permute(0, 1, 3, 4, 2)
        valid = valid & (target < logits.shape[-1])
        result["depth"] = F.cross_entropy(logits[valid], target[valid]) if valid.any() else zero
    else:
        result["depth"] = zero

    if "road" in batch:
        target = batch["road"].to(outputs["road_logits"].device).long()
        known = (target >= 0) & (target < 3)
        if known.any():
            logits = outputs["road_logits"].float().permute(0, 2, 3, 1)[known]
            true = target[known]
            result["road_ce"] = F.cross_entropy(logits, true)
            probability = logits.softmax(-1)
            onehot = F.one_hot(true, 3).float()
            intersection = (probability * onehot).sum(0)
            denominator = probability.sum(0) + onehot.sum(0)
            present = onehot.sum(0) > 0
            result["road_dice"] = (1 - (2 * intersection + 1) / (denominator + 1))[present].mean()
        else:
            result["road_ce"], result["road_dice"] = zero, zero
    else:
        result["road_ce"], result["road_dice"] = zero, zero
    result["road"] = result["road_ce"] + result["road_dice"]

    target, records = make_center_targets(outputs, batch, config)
    known = batch.get("detection_valid", torch.zeros_like(target, dtype=torch.bool)).to(target.device)
    result["center"], center_stats = center_focal_loss(
        outputs["center_logits"], target, known,
        positive_normalization=weights.get("center_positive_normalization", "global"), return_stats=True)
    # Scalars are logged by the engine, but excluded from every loss total.
    for class_index, class_name in enumerate(("car", "pedestrian")):
        for name, values in center_stats.items():
            result[f"center_{class_name}_{name}"] = values[class_index].detach()
    geometry = outputs["geometry_map"].float()
    predicted_geometry, desired_offset, desired_boxes = [], [], []
    for b, cells, offsets, boxes in records:
        predicted_geometry.append(geometry[b].flatten(1).T[cells])
        desired_offset.append(offsets)
        desired_boxes.append(boxes)
    if predicted_geometry:
        pred, offset, truth = torch.cat(predicted_geometry), torch.cat(desired_offset), torch.cat(desired_boxes)
        result["initial_position"] = (F.smooth_l1_loss(pred[:, :2].sigmoid(), offset, beta=.1)
                                      + F.smooth_l1_loss(pred[:, 2], truth[:, 2]))
        result["initial_dimensions"] = F.smooth_l1_loss(pred[:, 3:6], truth[:, 3:6].log(), beta=.1)
        # Normalize the sin/cos vector; epsilon prevents undefined zero-norm gradients.
        unit = F.normalize(pred[:, 6:8], dim=-1, eps=1e-6)
        target_unit = torch.stack([truth[:, 6].sin(), truth[:, 6].cos()], dim=-1)
        result["initial_yaw"] = (1 - (unit * target_unit).sum(-1)).mean()
    else:
        result["initial_position"] = result["initial_dimensions"] = result["initial_yaw"] = zero
    result["initial_total"] = result["center"] + result["initial_position"] + result["initial_dimensions"] + result["initial_yaw"]

    class_losses, pred_boxes, true_boxes = [], [], []
    if refine and "boxes" in batch and "labels" in batch:
        for b in range(outputs["refined_boxes"].shape[0]):
            boxes = outputs["refined_boxes"][b].float()
            logits = outputs["refined_logits"][b].float()
            truth, labels = _eligible_targets(batch, b, boxes, config)
            row, col = match_boxes(boxes, logits, truth, labels,
                                   match_bev_iou_weight=weights.get("match_bev_iou_weight", 0.0))
            target_class = torch.zeros_like(logits)
            valid = _proposal_validity(boxes, batch, b, config)
            if len(row):
                target_class[row, labels[col]] = 1
                # A matched, labeled object supplies a known class identity.
                valid[row] = True
                pred_boxes.append(boxes[row])
                true_boxes.append(truth[col])
            class_losses.append(masked_class_focal(logits, target_class, valid))
    result["refined_class"] = torch.stack(class_losses).mean() if class_losses else zero
    if pred_boxes:
        pred, truth = torch.cat(pred_boxes), torch.cat(true_boxes)
        result["refined_position"] = F.smooth_l1_loss(pred[:, :3], truth[:, :3])
        result["refined_dimensions"] = F.smooth_l1_loss(pred[:, 3:6].clamp_min(1e-3).log(), truth[:, 3:6].log(), beta=.1)
        result["refined_yaw"] = (1 - (pred[:, 6] - truth[:, 6]).cos()).mean()
    else:
        result["refined_position"] = result["refined_dimensions"] = result["refined_yaw"] = zero
    result["refined_total"] = result["refined_class"] + result["refined_position"] + result["refined_dimensions"] + result["refined_yaw"]
    result["total"] = (float(weights.get("depth", 1.)) * result["depth"]
                       + float(weights.get("road", 1.)) * result["road"]
                       + float(weights.get("initial", 1.)) * result["initial_total"]
                       + float(weights.get("refined", 1.)) * result["refined_total"])
    return result
