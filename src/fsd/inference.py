"""Decode predicted scenes. This module never reads ground-truth boxes."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def select_detections(outputs, index=0, threshold=0.1, max_objects=100, refined=True):
    from fsd.metrics import pairwise_iou
    prefix = 'refined' if refined else 'initial'
    boxes = outputs[f'{prefix}_boxes'][index].detach().float().cpu().numpy()
    prob = outputs[f'{prefix}_logits'][index].detach().float().sigmoid().cpu().numpy()
    labels = prob.argmax(-1)
    scores = prob.max(-1)
    valid = np.isfinite(boxes).all(-1) & np.isfinite(scores) & (scores >= threshold)
    valid &= (boxes[:, 3:6] > 0.05).all(-1) & (boxes[:, 3:6] < 30).all(-1)
    order = np.flatnonzero(valid)[np.argsort(-scores[valid], kind='stable')]
    keep = []
    for idx in order:
        same = [j for j in keep if labels[j] == labels[idx]]
        if same:
            # Near-identical center proposals should yield a single rendered object.
            overlaps = pairwise_iou(boxes[idx:idx+1], boxes[same], mode='bev')
            if np.max(overlaps) > 0.5:
                continue
        keep.append(int(idx))
        if len(keep) >= max_objects:
            break
    return dict(boxes=boxes[keep], scores=scores[keep], labels=labels[keep])


class SceneTracker:
    """Class-constrained association in world coordinates; no size/yaw smoothing."""
    def __init__(self, max_age=1.1):
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        self.sequence = None
        self.previous_time = None

    def update(self, detections, timestamp, ego_to_world, sequence):
        if sequence != self.sequence or (self.previous_time is not None and timestamp <= self.previous_time):
            self.tracks.clear()
        self.sequence, self.previous_time = sequence, timestamp
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp-v['time'] <= self.max_age}
        boxes, labels = detections['boxes'], detections['labels']
        centers = boxes[:, :3] @ ego_to_world[:3, :3].T + ego_to_world[:3, 3]
        ids = np.full(len(boxes), -1, dtype=np.int64)
        keys = list(self.tracks)
        if keys and len(boxes):
            predicted = np.array([self.tracks[k]['center'] + self.tracks[k]['velocity'] *
                                  (timestamp-self.tracks[k]['time']) for k in keys])
            distance = np.linalg.norm(centers[:, None, :2]-predicted[None, :, :2], axis=-1)
            same = labels[:, None] == np.array([self.tracks[k]['label'] for k in keys])[None]
            gates = np.where(labels[:, None] == 0, 8., 3.)
            cost = np.where(same & (distance < gates), distance, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e6:
                    ids[i] = keys[j]
        for i in range(len(boxes)):
            velocity = np.zeros(3)
            if ids[i] < 0:
                ids[i] = self.next_id
                self.next_id += 1
            else:
                old = self.tracks[int(ids[i])]
                dt = timestamp-old['time']
                if dt > 1e-3:
                    velocity = (centers[i]-old['center'])/dt
            self.tracks[int(ids[i])] = dict(center=centers[i], velocity=velocity,
                                           label=int(labels[i]), time=timestamp)
        return ids
