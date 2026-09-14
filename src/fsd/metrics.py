"""Custom, visibility-aware KITTI-360 subset metrics (not official KITTI scores).

Boxes are geometric-center (x, y, z, length, width, height, yaw), metres/radians.
AP uses score-ordered one-to-one matching and the all-point precision envelope.
"""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any
import math
import numpy as np


def _array(value, dtype=np.float64):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _boxes(value):
    boxes = _array(value).reshape(-1, 7)
    if not np.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
        raise ValueError("Boxes must be finite with strictly positive dimensions")
    return boxes


def wrap_angle(angle):
    """Shortest signed angular displacement, preserving physical front/back."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _corners(boxes):
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=float)
    local = boxes[:, None, 3:5] * signs[None] * .5
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rotation = np.stack((c, -s, s, c), axis=-1).reshape(-1, 2, 2)
    return np.einsum("nij,nkj->nki", rotation, local) + boxes[:, None, :2]


def _clip_polygon(subject, clip):
    # Convex Sutherland-Hodgman fallback; both polygon windings are CCW.
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    out = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        source, out = out, []
        if not source:
            break
        edge = b - a
        prev = source[-1]
        prev_side = cross(edge, prev - a)
        for curr in source:
            curr_side = cross(edge, curr - a)
            if (curr_side >= -1e-10) != (prev_side >= -1e-10):
                denom = prev_side - curr_side
                if abs(denom) > 1e-15:
                    out.append(prev + (curr - prev) * prev_side / denom)
            if curr_side >= -1e-10:
                out.append(curr)
            prev, prev_side = curr, curr_side
    if len(out) < 3:
        return 0.
    pts = np.asarray(out)
    return abs(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1))) * .5


def pairwise_ious(pred_boxes, gt_boxes):
    """Return exact upright oriented BEV and 3D IoU matrices, [N,M]."""
    p, g = _boxes(pred_boxes), _boxes(gt_boxes)
    overlap = np.zeros((len(p), len(g)), dtype=float)
    if not len(p) or not len(g):
        return overlap.copy(), overlap
    pc, gc = _corners(p), _corners(g)
    pmin, pmax, gmin, gmax = pc.min(1), pc.max(1), gc.min(1), gc.max(1)
    candidate = ((pmax[:, None] > gmin) & (gmax > pmin[:, None])).all(-1)
    pi, gi = np.where(candidate)
    if len(pi):
        try:
            import shapely
            if hasattr(shapely, "polygons"):
                ppoly, gpoly = shapely.polygons(pc), shapely.polygons(gc)
                overlap[pi, gi] = shapely.area(shapely.intersection(ppoly[pi], gpoly[gi]))
            else:
                from shapely.geometry import Polygon
                for i, j in zip(pi, gi):
                    overlap[i, j] = Polygon(pc[i]).intersection(Polygon(gc[j])).area
        except ImportError:
            for i, j in zip(pi, gi):
                overlap[i, j] = _clip_polygon(pc[i], gc[j])
    parea, garea = p[:, 3] * p[:, 4], g[:, 3] * g[:, 4]
    bev = overlap / np.maximum(parea[:, None] + garea - overlap, 1e-12)
    bottom = np.maximum(p[:, None, 2] - p[:, None, 5] / 2, g[:, 2] - g[:, 5] / 2)
    top = np.minimum(p[:, None, 2] + p[:, None, 5] / 2, g[:, 2] + g[:, 5] / 2)
    intersect3d = overlap * np.maximum(top - bottom, 0)
    union3d = (parea * p[:, 5])[:, None] + garea * g[:, 5] - intersect3d
    return np.clip(bev, 0, 1), np.clip(intersect3d / np.maximum(union3d, 1e-12), 0, 1)


def pairwise_iou(pred_boxes, gt_boxes, mode="bev"):
    if mode not in {"bev", "3d"}:
        raise ValueError("mode must be bev or 3d")
    return pairwise_ious(pred_boxes, gt_boxes)[mode == "3d"]


def detection_match(iou, scores, threshold=.5, supported=None):
    """Score-ordered match; status in original order: 1 TP, 0 FP, -1 ignored.

    Positive overlap establishes support even when a visibility mask is false,
    so matched positives and duplicates of known objects cannot disappear.
    """
    iou, scores = _array(iou), _array(scores).reshape(-1)
    if iou.ndim != 2 or len(iou) != len(scores):
        raise ValueError("IoU must be [N,M] and scores [N]")
    support = np.ones(len(scores), bool) if supported is None else _array(supported, bool).reshape(-1)
    if len(support) != len(scores):
        raise ValueError("supported must have one value per prediction")
    status = np.full(len(scores), -1, np.int64)
    matched = np.full(len(scores), -1, np.int64)
    used = np.zeros(iou.shape[1], bool)
    for i in np.argsort(-scores, kind="stable"):
        candidates = np.where((iou[i] >= threshold) & ~used)[0]
        if len(candidates):
            j = candidates[np.argmax(iou[i, candidates])]
            used[j], status[i], matched[i] = True, 1, j
        elif support[i] or np.any(iou[i] >= threshold):
            status[i] = 0
    return status, matched


def _detection_summary(records, n_gt):
    records = sorted(records, key=lambda row: -row[0])
    tp = np.array([row[1] == 1 for row in records], dtype=float)
    fp = 1 - tp
    ntp, nfp = int(tp.sum()), int(fp.sum())
    ap = None
    if n_gt:
        recall = np.cumsum(tp) / n_gt
        precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1)
        r = np.r_[0., recall, 1.]
        p = np.r_[0., precision, 0.]
        p = np.maximum.accumulate(p[::-1])[::-1]
        changed = np.flatnonzero(r[1:] != r[:-1])
        ap = float(np.sum((r[changed + 1] - r[changed]) * p[changed + 1]))
    return {"ap": ap, "recall": ntp / n_gt if n_gt else None,
            "precision": ntp / (ntp + nfp) if ntp + nfp else (0. if n_gt else None),
            "gt_count": int(n_gt), "prediction_count": len(records), "true_positives": ntp,
            "false_positives": nfp, "false_negatives": int(n_gt - ntp)}


def road_confusion(prediction, target, num_classes=3):
    pred, gt = _array(prediction, np.int64), _array(target, np.int64)
    if pred.shape != gt.shape:
        raise ValueError("Road prediction and target must have identical shapes")
    valid = (gt >= 0) & (gt < num_classes)
    if ((pred[valid] < 0) | (pred[valid] >= num_classes)).any():
        raise ValueError("Road predictions must be class indices on supervised pixels")
    return np.bincount((gt[valid] * num_classes + pred[valid]).ravel(), minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _road_summary(confusion):
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    ious = [float(i / u) if u else None for i, u in zip(intersection, union)]
    represented = [v for v in ious if v is not None]
    return {"iou": dict(zip(("road", "sidewalk", "other_ground"), ious)),
            "miou": float(np.mean(represented)) if represented else None,
            "accuracy": float(intersection.sum() / confusion.sum()) if confusion.sum() else None,
            "valid_pixels": int(confusion.sum()), "confusion": confusion.tolist()}


def depth_metrics(prediction, target, valid=None):
    """Metric RADIAL depth diagnostics; target <=0/nonfinite or mask false ignored."""
    p, t = _array(prediction), _array(target)
    if p.shape != t.shape:
        raise ValueError("Depth shapes differ")
    mask = np.isfinite(t) & (t > 0)
    if valid is not None:
        mask &= _array(valid, bool)
    count = int(mask.sum())
    if not count:
        return {"count": 0, "invalid_prediction_count": 0, "mae_m": None, "rmse_m": None, "abs_rel": None, "delta_1": None}
    p, t = p[mask], t[mask]
    invalid = ~np.isfinite(p) | (p <= 0)
    if invalid.any():
        return {"count": count, "invalid_prediction_count": int(invalid.sum()), "mae_m": None, "rmse_m": None, "abs_rel": None, "delta_1": None}
    difference = p - t
    ratio = np.maximum(p / t, t / p)
    return {"count": count, "invalid_prediction_count": 0,
            "mae_m": float(np.mean(abs(difference))), "rmse_m": float(np.sqrt(np.mean(difference ** 2))),
            "abs_rel": float(np.mean(abs(difference) / t)), "delta_1": float(np.mean(ratio < 1.25))}


def latency_summary(latencies_ms):
    """Summarize externally timed fresh-scene latencies; CUDA must be synchronized.

    Throughput assumes sequential updates and is 1000/mean, not 1000/median.
    Input must exclude warmup; this helper does not itself perform GPU timing.
    """
    values = _array(latencies_ms).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Supply nonempty finite positive elapsed milliseconds")
    return {"samples": len(values), "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "min_ms": float(values.min()),
            "max_ms": float(values.max()), "sequential_fps": float(1000 / values.mean())}


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


class Evaluator:
    """Accumulate custom per-frame metrics with sealed-test handling left to caller.

    frame_metadata: sequence, timestamp (seconds), gt_instance_ids [M], optional
    ego_to_world [4,4]. Temporal metrics require identity/time AND ego_to_world;
    positions are world-aligned before computing changes in residual error.
    Updates must be chronological within each sequence. Duplicate/reverse times
    and gaps greater than max_temporal_gap_s reset a track comparison.
    """
    thresholds = {"bev_ap50": ("bev", .5), "3d_ap25": ("3d", .25), "3d_ap50": ("3d", .5)}
    distance_bins = (("0_20m", 0., 20.), ("20_40m", 20., 40.), ("40m_plus", 40., math.inf))

    def __init__(self, num_classes=2, class_names=None, bev_min=-40., bev_step=.5, max_temporal_gap_s=1.):
        self.num_classes = int(num_classes)
        self.class_names = list(class_names or (["car", "pedestrian"] if num_classes == 2 else [str(i) for i in range(num_classes)]))
        if len(self.class_names) != num_classes:
            raise ValueError("class_names must match num_classes")
        self.bev_min, self.bev_step = float(bev_min), float(bev_step)
        self.max_temporal_gap_s = max_temporal_gap_s
        self.frames = 0
        self.records = defaultdict(list)
        self.gt_counts = defaultdict(int)
        self.ignored_counts = defaultdict(int)
        self.confusion = np.zeros((3, 3), np.int64)
        self.geometry = defaultdict(lambda: defaultdict(list))
        self.temporal = defaultdict(list)
        self.track_state = {}

    def _support(self, boxes, labels, detection_valid):
        if detection_valid is None:
            return np.ones(len(boxes), bool)
        mask = _array(detection_valid, bool)
        if mask.ndim != 3 or mask.shape[0] != self.num_classes:
            raise ValueError("detection_valid must be [num_classes,H,W]")
        ij = np.floor((boxes[:, :2] - self.bev_min) / self.bev_step).astype(int)
        valid = ((ij >= 0) & (ij < np.array([mask.shape[2], mask.shape[1]]))).all(1)
        support = np.zeros(len(boxes), bool)
        support[valid] = mask[labels[valid], ij[valid, 1], ij[valid, 0]]
        return support

    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
               road_pred=None, road_target=None, detection_valid=None, frame_metadata=None):
        p, g = _boxes(pred_boxes), _boxes(gt_boxes)
        scores = _array(pred_scores).reshape(-1)
        pl, gl = _array(pred_labels, np.int64).reshape(-1), _array(gt_labels, np.int64).reshape(-1)
        if len(scores) != len(p) or len(pl) != len(p) or len(gl) != len(g):
            raise ValueError("Boxes/scores/labels have inconsistent counts")
        if not np.isfinite(scores).all() or (pl < 0).any() or (pl >= self.num_classes).any() or (gl < 0).any() or (gl >= self.num_classes).any():
            raise ValueError("Scores must be finite and labels valid class IDs")
        if (road_pred is None) != (road_target is None):
            raise ValueError("Supply both road_pred and road_target, or neither")
        if road_pred is not None:
            self.confusion += road_confusion(road_pred, road_target)
        support = self._support(p, pl, detection_valid)
        bev, iou3d = pairwise_ious(p, g)
        pd, gd = np.linalg.norm(p[:, :2], axis=1), np.linalg.norm(g[:, :2], axis=1)
        meta = frame_metadata or {}
        ids = meta.get("gt_instance_ids", meta.get("instance_ids"))
        if ids is not None:
            ids = _array(ids, np.int64).reshape(-1)
            if len(ids) != len(g):
                raise ValueError("gt_instance_ids must correspond to GT boxes")
        for cls in range(self.num_classes):
            pi, gi = np.flatnonzero(pl == cls), np.flatnonzero(gl == cls)
            self.gt_counts[(cls, "all")] += len(gi)
            for name, lo, hi in self.distance_bins:
                self.gt_counts[(cls, name)] += int(np.sum((gd[gi] >= lo) & (gd[gi] < hi)))
            for metric, (mode, threshold) in self.thresholds.items():
                iou = (bev if mode == "bev" else iou3d)[np.ix_(pi, gi)]
                status, matches = detection_match(iou, scores[pi], threshold, support[pi])
                self.ignored_counts[(metric, cls)] += int((status == -1).sum())
                for j, index in enumerate(pi):
                    if status[j] == -1:
                        continue
                    self.records[(metric, cls, "all")].append((float(scores[index]), int(status[j])))
                    distance = gd[gi[matches[j]]] if matches[j] >= 0 else pd[index]
                    for name, lo, hi in self.distance_bins:
                        if lo <= distance < hi:
                            self.records[(metric, cls, name)].append((float(scores[index]), int(status[j])))
                            break
                    if metric == "bev_ap50" and status[j] == 1:
                        target_index = gi[matches[j]]
                        self._geometry(cls, p[index], g[target_index])
                        if ids is not None and ids[target_index] >= 0:
                            self._temporal(cls, int(ids[target_index]), p[index], g[target_index], meta)
        self.frames += 1

    def _geometry(self, cls, p, g):
        values = self.geometry[cls]
        values["center_error_m"].append(float(np.linalg.norm(p[:3] - g[:3])))
        values["ground_center_error_m"].append(float(np.linalg.norm(p[:2] - g[:2])))
        values["dimension_mae_m"].append(float(np.mean(abs(p[3:6] - g[3:6]))))
        values["yaw_error_deg"].append(float(abs(wrap_angle(p[6] - g[6])) * 180 / np.pi))

    def _temporal(self, cls, instance_id, p, g, meta):
        if "timestamp" not in meta or "sequence" not in meta or "ego_to_world" not in meta:
            return
        timestamp = float(meta["timestamp"])
        pose = _array(meta["ego_to_world"]).reshape(4, 4)
        key = (str(meta["sequence"]), cls, instance_id)
        # Common translation cancels; rotation is required to compare residuals
        # in a stable world frame when the ego vehicle turns.
        position_residual = pose[:3, :3] @ (p[:3] - g[:3])
        dimension_residual = p[3:6] - g[3:6]
        yaw_residual = float(wrap_angle(p[6] - g[6]))
        prev = self.track_state.get(key)
        if prev is not None:
            dt = timestamp - prev[0]
            if 0 < dt <= self.max_temporal_gap_s:
                self.temporal["center_residual_change_m"].append(float(np.linalg.norm(position_residual - prev[1])))
                self.temporal["dimension_residual_change_m"].append(float(np.mean(abs(dimension_residual - prev[2]))))
                self.temporal["yaw_residual_change_deg"].append(float(abs(wrap_angle(yaw_residual - prev[3])) * 180 / np.pi))
                self.temporal["center_residual_rate_mps"].append(float(np.linalg.norm(position_residual - prev[1]) / dt))
                self.temporal["yaw_residual_rate_degps"].append(float(abs(wrap_angle(yaw_residual - prev[3])) * 180 / np.pi / dt))
        self.track_state[key] = (timestamp, position_residual, dimension_residual, yaw_residual)

    def compute(self) -> dict[str, Any]:
        result = {"metric_protocol": "custom_KITTI360_subset_all_point_AP_not_official_benchmark",
                  "frames": self.frames, "objects": {}, "road": _road_summary(self.confusion),
                  "geometry_matching": "class-consistent BEV IoU >= 0.5; conditional on matched detections",
                  "temporal_matching": "GT identity; class-consistent BEV IoU >= 0.5; world-aligned error residuals",
                  "distance_assignment": "matched predictions use GT center range; false positives use predicted center range"}
        for metric in self.thresholds:
            cls_result = {}
            aps = []
            for cls, name in enumerate(self.class_names):
                summary = _detection_summary(self.records[(metric, cls, "all")], self.gt_counts[(cls, "all")])
                summary["ignored_predictions"] = self.ignored_counts[(metric, cls)]
                summary["distance_bins"] = {bin_name: _detection_summary(self.records[(metric, cls, bin_name)], self.gt_counts[(cls, bin_name)]) for bin_name, _, _ in self.distance_bins}
                cls_result[name] = summary
                if summary["ap"] is not None:
                    aps.append(summary["ap"])
            result["objects"][metric] = {"map": _mean_or_none(aps), "classes": cls_result, "classes_with_gt": len(aps)}
        result["geometry"] = {name: {"matched_count": len(self.geometry[cls]["center_error_m"]),
            **{key: _mean_or_none(self.geometry[cls][key]) for key in ("center_error_m", "ground_center_error_m", "dimension_mae_m", "yaw_error_deg")}}
            for cls, name in enumerate(self.class_names)}
        temporal_names = ("center_residual_change_m", "dimension_residual_change_m", "yaw_residual_change_deg", "center_residual_rate_mps", "yaw_residual_rate_degps")
        result["temporal"] = {"matched_transition_count": len(self.temporal["center_residual_change_m"]),
                              **{name: _mean_or_none(self.temporal[name]) for name in temporal_names}}
        return result
