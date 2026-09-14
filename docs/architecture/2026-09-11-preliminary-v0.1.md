# FSD scene demo: preliminary architecture v0.1

Recorded: 2026-09-11. Status: agreed design direction, not an implemented or validated model.
Project: C:\Users\William\projects\fsd
Owner of final architecture decisions: William.

## Purpose and authority

Preserve what we decided, why we decided it, and how experiments should change it. Keep this dated baseline intact. Later changes belong in a new architecture version with links to experiments; update the current-version pointer in docs/architecture/README.md. New explicit user decisions take precedence over this record. Implementation details may be resolved within the agreed scope; material architecture changes should be presented to William with evidence.

## Objective and scope

Optimize the latency/accuracy tradeoff on William's RTX 5070, used for both training and inference, while preserving the required simplified 3D scene. No single configuration can maximize accuracy and minimize latency simultaneously: compare measured alternatives and retain useful tradeoffs.

Input: multiple calibrated cameras, chronological observations, timestamps, and ego motion for historical alignment. Camera configuration and deployment ego-motion source are unresolved.

Output, in priority order:
1. Road and sidewalk surface geometry and supported boundaries.
2. Individual cars: class, confidence, 3D position, dimensions, heading.
3. Individual people: class, confidence, 3D position and dimensions; box heading where meaningful/supervised, not articulated pose.

Render stock assets in a common metric coordinate system from an elevated viewpoint, inspired by the supplied Tesla UI reference. Graphics supply visual detail, not missing perception evidence. Persistent IDs are needed for stable scene assembly.

Lane detection is explicitly deferred. Exact vehicle meshes, dense voxel output, driving control/planning, and Raspberry Pi deployment are outside this baseline. Flat ground is a local prototype assumption, not a property guaranteed by KITTI.

## Complete data flow

Current camera images
  -> shared ResNet-50
  -> FPN image features
  -> depth-informed image-to-BEV conversion [exact implementation provisional]
  -> small BEV CNN encoder
  -> parallel road/layout head and initial object head
       road head -> surface regions -> flat mesh/boundaries
       object head -> initial boxes -> spatial/temporal object refinement
  -> detection cleanup and temporal association
  -> common-coordinate scene -> renderer

Refinement also reads current FPN features and cached historical object features/reference positions. Camera calibration, timestamps, and ego motion accompany this pathway.

IMPORTANT: temporal attention is proposed AFTER initial object generation. Do not silently restore the earlier temporal-BEV-before-heads design. An additional BEV temporal stage requires a separate justified experiment. Current images alone enter the image CNN; old images are not repeatedly encoded during streaming inference.

## Component contracts and rationale

### 1. Image encoder: accepted ResNet-50 + FPN

Apply the same backbone/neck weights to every camera. Weight sharing does not eliminate per-camera computation. Extract and combine multi-scale image features. Do not add SECONDFPN as an extra encoder. ResNet-50 + FPN is the accepted baseline, not a proven fastest model on the 5070.

Rationale: established multi-camera reference used by SuperOcc, regular GPU operations, and less aggressive starting size than EfficientNet-B7. FPN width, levels, normalization, pretrained checkpoint, and fine-tuning schedule remain open. Earlier 64-channel suggestions are hypotheses, not fixed requirements.

### 2. Depth-informed BEV conversion: proposed mechanism

Predict a distribution P(d | u,v) over depth bins at feature locations. Lift contextual image features using F_lifted(u,v,d,c) = F_image(u,v,c) * P(d | u,v), then aggregate into a shared top-down feature map using calibration.

Depth represents camera-axis depth if using standard pinhole backprojection. Camera-to-vehicle transforms establish common metric coordinates. Fisheyes require their actual projection model or appropriately calibrated rectification. Keep resizing/cropping transforms consistent with calibration.

Projected LiDAR can supervise valid visible surface depth during training; it is not a required inference input. Surface depth is not object-center depth. Lifting distributes evidence; it does not by itself refine a box or guarantee correct depth. A second lifting pass is not automatically useful. Avoid materializing large intermediate volumes when an equivalent efficient pooling operator is available.

Exact depth network, bins, pooling, FPN interface, and BEV dimensions remain unresolved. Do not treat approximately 640 feature channels from an earlier plan as a requirement.

### 3. Shared BEV encoder

A compact CNN integrates spatial evidence across cameras. BEV is a ground-indexed grid of feature vectors, not merely an overhead RGB image. Its channels can retain vertical evidence, but compression can lose detail. Vertical position/height predictions must be validated, not assumed accurate because the output is 3D.

Rationale: share camera fusion and spatial processing between road and object heads. This is a compute hypothesis, not a measured comparison against direct image-query detection.

### 4. Road/layout head

Predict road/sidewalk and appropriate background/unknown regions from BEV; exact taxonomy and output resolution remain open. Prepare supervision in the same coordinate system as the head, preferably using valid labeled 3D ground evidence or carefully projected image labels. A camera mask cannot directly supervise a BEV mask without conversion.

Convert supported regions to flat meshes/boundaries. Do not interpret a car occluding road as a curb. Road/sidewalk class interfaces provide boundary evidence, not guaranteed measured curb geometry. Hidden-road completion, road temporal stabilization, islands, and curb heights are not established capabilities. Retain actual measured training labels; apply flat-ground approximation deliberately to ground targets/rendering, not by flattening all object ground truth.

### 5. Initial object proposals

Center heatmaps for car and person classes, plus geometry regression: center offset, vertical position, length/width/height, and yaw representation. Local peaks identify proposals; confidence filtering and duplicate removal follow.

Center-based does not mean using one isolated pixel: surrounding context is encoded in its feature vector. Initial centers and geometry are allowed to change in refinement.

Illustrative settings only: 80 m x 80 m coverage, 0.5 m cells -> 160 x 160 BEV; retain up to 100 objects TOTAL across classes. A candidate budget may be larger than the final display cap. Neither grid nor cap is finalized. A cap does not guarantee recall; nearby centers colliding in a cell cannot be fixed by increasing it.

### 6. Spatial and temporal object refinement: accepted design direction

Treat boxes as proposals rather than immutable answers.
1. Select a small set of 3D points around each proposed box/extent.
2. Project points into valid camera views and sample current multi-scale FPN features; optional BEV sampling is an implementation choice.
3. Aggregate useful spatial evidence with attention.
4. Use cached historical object features and reference positions after ego-motion alignment. Allow independent object motion; ego alignment alone is insufficient.
5. Predict updates to center, dimensions, orientation, class/confidence.

Example update: c_new = c + delta_c; size_new = size * exp(delta_size); theta_new = theta + delta_theta with circular handling.

Begin with one lightweight refinement stage as a proposal. Sampling count, feature width, temporal memory length, query selection, and correspondence mechanism need design. Historical queries are not automatically ground-truth tracks. Prevent training from relying on future frames. Handle empty history and disappearing/new objects.

Rationale: William wanted spatial/temporal evidence to correct noisy box geometry after initial generation, borrowing the projection/attention idea of VoxFormer/IPFormer/SuperOcc without their complete voxel reconstruction. This is our adaptation, not a drop-in implementation of any paper. Refinement cannot reliably recover objects missing from the candidate set.

### 7. Scene assembly

Remove duplicate boxes across proposals/views and associate objects across timestamps for IDs. Do not assume temporal attention creates persistent IDs. Explicit output smoothing is not mandatory initially: measure whether learned temporal refinement suffices before adding filters. Do not force turning cars to maintain heading or snap parked vehicles into lanes.

Use one documented metric frame, recommended x forward, y left, z up. Define geometric-center versus bottom-center box conventions consistently at dataset conversion, loss, and rendering boundaries. Assets use dimensions and yaw; they are not recovered meshes.

## Training and loss plan

Primary dataset direction: KITTI-360. Audit usable 3D object labels, dynamic annotations, class counts, confidence, timestamps, and calibration BEFORE fixing the final training set. Its perspective image pair has released image semantics/instances; fisheye views do not inherit those dense labels automatically. Deriving surround-view targets from 3D annotations requires visibility and timing handling.

L_total = lambda_depth L_depth + lambda_road L_road + lambda_initial L_initial + lambda_refined L_refined.

- Depth: masked loss against valid projected LiDAR surface depth; categorical depth-bin loss or a suitable regression formulation must be selected to match the output.
- Road: valid semantic targets in head coordinates; cross-entropy plus Dice is a proposed baseline.
- Initial objects: center heatmap focal-style loss; positive-location regression for position and dimensions; circular orientation handling (for example 1 - cos(angle error)).
- Refined objects: classification and box geometry against matched annotations. Matching/reassignment policy remains open; supervise both initial and refined predictions.
- Missing/unknown labels: mask them, never interpret them as background. Normalize valid task losses before selecting weights.
- Temporal supervision: chronological examples and ordinary geometry losses first. Temporal attention can learn consistency, but per-frame accuracy loss does not directly penalize jitter. Add explicit temporal penalties only if evidence warrants it; allow real object motion.

Keep entire drives/sequences separated. Select configurations/checkpoints on validation, not on the final test set. Document every label transformation and retain source annotations.

## Performance expectations, NOT measurements

Target: 30 perception updates/s (33.3 ms/update). Earlier 20-40 FPS / 25-50 ms is an unvalidated planning estimate, not a commitment or demonstrated capability.

Estimate assumed four cameras at about 704 x 256, FP16, compact BEV, roughly 100 proposals, one sparse refinement stage, cached short history and optimized operators. These are assumptions, not locked configuration.

Illustrative 33 ms budget: preprocessing/transfer 3; backbone+FPN 12; depth/lifting/BEV 9; heads/refinement 7; cleanup/tracking 2. No stage has been measured. Capture, ego-motion estimation, rendering/display and queuing were excluded. Measure those separately for true end-to-end latency. A 60 FPS renderer does not imply 60 fresh detections/s.

Published comparison context: SuperOcc-T reports 30.3 FPS on RTX 4090 FP32 with 8-frame history; IPFormer reports 0.33 s (~3 FPS) on A100. Neither predicts our 5070 latency or proves our accuracy. IPFormer's 14x comparison largely removes baseline clustering cost.

## Experiment and change rules

1. Establish a runnable baseline and verify actual CUDA execution on the target device before training. No inference-only claim from device enumeration.
2. Save exact config, code revision or source snapshot, environment, weights, dataset split and preprocessing with each run. Seed and timing protocol belong in the record.
3. Change one main mechanism per comparison where practical. Keep baseline checkpoints/results intact and distinguish proposals from tested changes.
4. Report latency (median and tail, e.g. p95), throughput, peak memory, and GPU precision/backend. Use warmup, correct GPU synchronization/timing, batch one scene, and fixed camera count/resolution/history. Separate cold start from steady state.
5. Report geometry performance: detection recall and 3D box overlap, position/dimension/yaw errors by class and distance; road IoU/boundary quality; temporal dimension/heading stability using matched tracks and accounting for real motion. Smooth-but-wrong predictions are failures too.
6. Useful controlled comparisons: no refinement vs spatial refinement vs spatial+temporal; grid/width/resolution changes; depth supervision on/off. Do not silently substitute a different output task to achieve FPS.
7. For failures, inspect labels/calibration/alignment and stage outputs before increasing model complexity. Missed initial proposals need recall improvements, not only box refinement. Unstable dimensions suggest temporal evidence/matching issues; large latency needs measured stage profiling.
8. Propose material changes with problem, evidence, expected benefit/cost and rollback. William makes final decisions. Record accepted changes in a new architecture version and update the pointer. User instructions supersede these workflow defaults.

Use docs/experiments/TEMPLATE.md for experiment records. Quantitative acceptance thresholds beyond the provisional FPS target remain to be chosen; do not invent pass/fail gates retrospectively.

## Next unresolved decisions

- Camera setup, synchronized sampling, image resolution and distortion handling.
- FPN outputs/width and pretrained weights.
- Depth/lifting implementation and metric range/binning.
- BEV range, resolution, channels and CNN depth.
- Ground target construction and object annotation audit.
- Proposal/matching policy and refinement sampling/history configuration.
- Deployment ego-motion source and persistent track association.
- Task-specific accuracy/jitter thresholds and full display latency budget.

## Primary references

- KITTI-360 annotation/calibration documentation: https://www.cvlibs.net/datasets/kitti-360/documentation.php
- KITTI-360 class definitions: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
- IPFormer, depth lifting and instance/voxel attention, not direct box detection: https://arxiv.org/html/2506.20671v3
- SuperOcc, sparse geometric queries and temporal evidence: https://arxiv.org/html/2601.15644v1
- CenterPoint, center proposals and second-stage refinement; LiDAR evidence does not validate our camera design: https://arxiv.org/abs/2006.11275
- DETR3D, 3D-to-image feature sampling: https://proceedings.mlr.press/v164/wang22b.html
- BEVPoolv2, efficient view transformation: https://arxiv.org/abs/2211.17111
