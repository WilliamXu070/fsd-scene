# Custom scene metrics and falsification controls

These are custom KITTI-360 subset metrics, not official KITTI or KITTI-360 leaderboard scores. AP is the all-point precision-envelope integral, with class-consistent, score-descending, one-to-one oriented-box IoU matching. All AP/IoU values are fractions, not percentages.

## API

`Evaluator(num_classes=2).update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, road_pred=None, road_target=None, detection_valid=None, frame_metadata=None)` accepts NumPy arrays, lists or detached Torch-compatible inputs. Boxes are `[x,y,z,length,width,height,yaw]` with geometric centers, metres and radians. Class IDs are 0 car and 1 pedestrian. Road tensors are class-index maps (0 road, 1 sidewalk, 2 other known ground, -1 unknown), not logits.

`compute()` returns a JSON-serializable snapshot without resetting state:

- `objects.bev_ap50`, `objects.3d_ap25`, `objects.3d_ap50`: each contains `map`, `classes_with_gt`, and `classes.car|pedestrian` with AP, recall, precision, counts, ignored count and `distance_bins`.
- `road`: per-class `iou`, `miou`, pixel accuracy, valid-pixel count and confusion matrix.
- `geometry.car|pedestrian`: matched count, Euclidean center and ground-center errors, dimension mean absolute error, circular heading error in degrees.
- `temporal`: matched transition count and changes/rates of position, size and heading **error residuals**, avoiding penalties for real motion.

AP is null when a class has no ground truth; it is zero when ground truth exists but every detection is missed. False-positive counts remain available for classes without ground truth. mAP excludes absent classes and reports its denominator; report class results independently so a missing class cannot look successful. IoU is null only where its union is zero. Geometry and temporal errors are conditional on successful BEV-IoU-0.5 matches; always report recall alongside them.

## Annotation validity and distance

`detection_valid` is `[2,H,W]`, with row y and column x; defaults are [-40,40) metres and 0.5m cells. Unsupported unmatched predictions are ignored. A matching known positive remains evaluable even where the mask is false. Duplicates overlapping known positives also remain false positives. An out-of-bounds prediction is unsupported when a validity mask is supplied. The evaluator does not invent ground truth beyond the mask.

Distance bins are [0,20), [20,40), and [40,infinity) horizontal radial metres. Successful predictions are assigned their matched ground truth's distance; false positives use their own distance. Overall matching is performed before partitioning, preventing a small boundary-crossing localization error from losing a match.

## Temporal metadata

Supply `frame_metadata={sequence, timestamp, gt_instance_ids, ego_to_world}`; `instance_ids` is accepted as an alias for ground-truth IDs. Timestamps are seconds. IDs index the supplied ground truth; they are evaluation association only and never network inputs. World rotation aligns position residuals across ego motion. Dimension and shortest yaw residual changes are compared only within the same sequence/class/GT ID, with strictly increasing time and gaps no greater than 1 second by default. Missing history/pose/identity yields null temporal metrics. These metrics measure geometry stability, not tracker ID switches or official tracking AP.

## Additional helpers

- `pairwise_ious(pred_boxes, gt_boxes)` returns exact upright oriented BEV and 3D IoU matrices. Vectorized Shapely intersection uses a cheap bounding-box candidate filter; a NumPy convex-clipping fallback is available.
- `pairwise_iou(..., mode='bev'|'3d')` selects one IoU matrix.
- `detection_match(iou, scores, threshold=.5, supported=None)` returns original-order status (1 TP, 0 FP, -1 ignored) and matched target indices (-1 unmatched).
- `road_confusion(prediction, target, num_classes=3)` accumulates valid class-index pixels.
- `depth_metrics(prediction_metres, target_metres, valid=None)` reports radial-depth MAE/RMSE/absolute-relative/delta-1. Convert predicted bins using the same radial-depth convention as targets. Invalid predictions on valid targets are counted and make metrics null rather than silently being dropped. Aggregate raw observations or squared errors across frames; do not average per-frame RMSE as if it were pooled RMSE.
- `latency_summary(latencies_ms)` reports mean/median/p95 and sequential FPS as 1000/mean latency. Caller must remove warmup, synchronize CUDA and define timing boundaries; this function cannot certify timing methodology.

## Verification

Run `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -q`.

21 passing controls verify analytic rotated/translated/scaled/vertically displaced intersections; perfect scores; misses; duplicates; confidence ordering; incorrect classes; validity and grid conventions; range bins; circular headings; unknown road pixels; invalid depths; ego-motion-aligned and real-motion-adjusted temporal residuals; sequence/gap resets; and latency arithmetic. Inputs are synthetic numerical controls only. No sealed-test predictions are read.
