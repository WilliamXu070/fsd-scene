# Original scene model: evolution and experiment rationale

Recorded 2026-09-13. Owner of material design decisions: William.

This is a retrospective and a proposed experiment register, not a new implemented architecture or permission to restart training. It preserves the distinction between design, implementation, training and evidence. Earlier undated steps are reconstructed from this thread; exact dates are not invented. Source documents can contain intermediate status notes; completed artifacts take precedence.

Follow-up audit: [continuation readiness review](2026-09-13-continuation-readiness-review.md). It identified an entrypoint guard blocking weights-only initialization and a partial epoch 22 after the 21 completed epochs. The guard was subsequently replaced with validated same-lineage transfers; [branching verification](../../experiments/nuscenes/CHECKPOINT_BRANCHING.md) passed. No new accuracy experiment has started.

## 1. Evolution from the first idea

| Step | Proposal or change | Reason | What actually happened / evidence |
|---|---|---|---|
| Early concept, before v0.1 | Monocular camera inference, with LiDAR and semantics available during training; roads, cars, bikes and people; possible Raspberry Pi deployment | Learn geometry from richer training streams while using cameras at inference | Discussion only at this scope. Later output scope narrowed and RTX 5070 became the target. |
| Early geometry exploration | Depth probabilities at image locations, backprojection from image/depth to metric coordinates, then voxel proposals and VoxFormer-like attention | Recover shape/location and correct an uncertain initial 3D estimate | IPFormer/VoxFormer were conceptual references. This was not a verified reproduction of their architectures. The approximately 640-channel sketch was never a binding final interface. |
| Scene representation decision | Replace full voxel reconstruction with road surfaces plus object boxes and stock graphics | The goal became a Tesla-like scene visualization, not arbitrary volumetric reconstruction | Retained metric center, dimensions and yaw; road/sidewalk regions. Lanes deferred. Graphics do not replace missing perception. Flat ground is a rendering approximation, not a guarantee about KITTI terrain. |
| Camera/time decision | Multiple calibrated cameras and prior observations | Surround coverage and evidence for stable geometry | Temporal attention explicitly placed AFTER initial object proposals; no temporal image mixing inside the CNN and no accepted pre-head temporal BEV fusion. |
| Backbone decision | EfficientNet-B7 and extra SECONDFPN considered; shared ResNet-50 + FPN selected | Familiar GPU-friendly building blocks and a more practical initial compute budget | ImageNet initialization; shared weights still incur computation for each view. No proof that ResNet is universally faster or more accurate. |
| v0.1, 2026-09-11 | ResNet/FPN → depth-informed BEV → compact BEV CNN → road and center/geometry heads → late spatial/temporal object refinement → tracking/rendering | Share fusion across tasks while allowing initial geometry to be corrected | Preserved design: [v0.1](2026-09-11-preliminary-v0.1.md). Exact tensor widths/bins were initially provisional. |
| v0.2, KITTI implementation | Four cameras, native fisheye geometry, radial-depth lifting, FPN128, BEV160×160, 200 initial candidates, one late refiner, 100 final detections | Make the proposed model executable on the selected KITTI-360 subset | [Implementation record](2026-09-11-implementation-v0.2.md). Radial range along unit rays superseded the provisional camera-axis depth description. |
| KITTI baseline A/B | Train initial heads, then joint refinement | Test the complete proposed pathway | Stage A completed 24 epochs; Stage B completed nine. Joint B degraded the selected scene score. This is a DIFFERENT lineage from the later nuScenes full run. |
| KITTI R1 | Train only the refiner; freeze all base weights and buffers | Investigate whether joint training damaged the shared representation | Completed. Base stayed exact, but selected score did not beat A. History controls did not establish reliable learned temporal correspondence. [R1](../experiments/REVISION01_REFINER_ONLY.md) |
| KITTI R2 | Add rotated BEV overlap to Hungarian training assignment; keep base frozen | Prefer geometrically better proposals when assigning targets | Completed; small validation improvement. R2 became the selected KITTI artifact. [R2](../experiments/REVISION02_OVERLAP_MATCH.md), [final selection](../../artifacts/final/selection.json) |
| KITTI R3 | Class-balanced positive center-loss normalization versus unchanged-loss continuation | Address missing pedestrian proposals | Completed and rejected; pedestrian issue unresolved and car guard failed. This is loss balancing, not the newly proposed per-class inference quota. [Assessment](../../artifacts/revision03-assessment/assessment.json) |
| KITTI quality critique | Smaller region, stricter confidence, fisheye downweighting, smoother roads, stronger yaw | Reduce clutter and unreliable distant/side detections | Discussion and diagnostics must not be confused with accepted universal fixes. Threshold/range reductions change coverage; invisible ground must not automatically become grass. No inherited KITTI camera weights or filters in the nuScenes model. |
| nuScenes v0.4, 2026-09-12 | Six pinhole views, capture-time calibration/poses, full trainval data; fresh ImageNet-backed task training | Improve surround-view supervision and remove KITTI fisheye handling | Six-camera adapter and target pipeline implemented. Map drivable/walkway targets plus observed ground; unknown/conflicts masked. [v0.4](../../experiments/nuscenes/ARCHITECTURE_v0.4.md) |
| nuScenes mini and preparation | Real-data geometry/recovery/learnability gates; pedestrian peak kernel changed from 3×3 to 1×1 | Verify the actual learning path and prevent nearby pedestrian peaks being suppressed | Mini fit demonstrated learnability, not generalization. Full-path checks with an initialized refiner did not demonstrate learned temporal gain. |
| nuScenes full baseline | Stage A on 700 train scenes / 28,130 timestamps, validation on 150 scenes / 6,019 timestamps | Establish full-data baseline before revisions | Epoch log records 21 completed Stage A epochs including validation. Best selected checkpoint: epoch 18. Stage B did NOT start in this full run. Training stopped for documentation. [Epoch log](../../experiments/nuscenes/artifacts/full-baseline/stage-a/epochs.jsonl), [state](../../experiments/nuscenes/CURRENT_STATE.md) |
| nuScenes diagnosis | Inspect proposal allocation, confidence and geometry instead of assuming backbone insufficiency | Isolate causes of poor detection | Epoch-6, 24-timestamp diagnostic: 88.2% of raw candidate slots pedestrian-labelled; correcting yaw at GT center cells materially improved car overlap. These are old diagnostic findings, not measured epoch-18 failure rates. [Diagnosis](../../experiments/nuscenes/ARCHITECTURE_DIAGNOSIS_2026-09-13.md) |
| Sparse4Dv3 comparison, 2026-09-13 | Run official pretrained R50 detector separately | Establish a stronger local reference and measure real RTX 5070 cost | On 403 validation timestamps, car BEV AP50 57.46% versus custom epoch-6 25.45%. Native FP32 preloaded-input throughput 13.24–13.42 scenes/s. Different training budgets; not official NDS or full-validation replication. It did not replace the custom model or road branch. [Results](../../experiments/sparse4dv3/RESULTS.md) |
| SparseDrive/SuperOcc/BEVDepth comparisons | Learn from richer features, quality estimation, repeated refinement and training recipes | Identify concrete candidate improvements without importing unrelated planning/occupancy outputs | Reference studies, not local SparseDrive/SuperOcc training or proof that any one mechanism caused the observed gap. |
| Current discussion | Return to the custom pipeline, retain epoch 18, test targeted improvements and complete nuScenes refinement | Evaluate the original idea more fully | Proposed only. No new runs, architecture edits or checkpoint migration performed by this documentation task. |

## 2. Current nuScenes architecture, as implemented

Six RGB views, each 704×256 → shared ImageNet ResNet-50 → FPN128 at strides 8/16/32.

P3/stride-8 features → 64 radial-depth bins from 1–80 m + 64-channel context → calibrated probability-weighted pooling → 64×160×160 BEV → three 128-channel residual blocks → shared 128×160×160 features.

Parallel heads produce road/walkway/other-known-ground logits and car/person center scores plus center offsets, z, log dimensions and sine/cosine yaw. Global peak selection retains 200 proposals. Stage B optionally samples box centers/eight corners from current multiscale images, performs spatial then temporal attention, and predicts residual boxes/scores. Temporal state holds up to three prior observations. Final cleanup retains at most 100 detections; geometric association supplies IDs.

The BEV covers 80×80 m with 0.5 m cells. Unknown ground/supervision remains masked. Source: [model](../../experiments/nuscenes/code/src/fsd/model.py), [full configuration](../../experiments/nuscenes/configs/full.json), [figure](../figures/original-pipeline.svg).

Full-data best epoch 18: car BEV AP50 37.46%, pedestrian 2.02%, road IoU 85.58%, sidewalk 57.70%. These are our custom validation metrics. That validation set has already informed model selection; it is not a sealed test. Source: [review](../../experiments/nuscenes/artifacts/full-baseline/epoch-018-review.json).

### Corrections to conversational shorthand

- “Refinement was never trained” applies to the full nuScenes run, not to the earlier KITTI experiments.
- The newer epoch log establishes completion of epoch 21 validation; an older state snapshot still says validation was underway.
- Joint refinement can hurt an already useful base: the KITTI lineage provides an actual caution, not a reason to assume nuScenes must behave identically.
- Passing tiny-set overfit proves learnability under those conditions; it does not establish generalization or temporal benefit.
- Sparse4D vs epoch 6 does not isolate attention, pretraining, FPN width or training duration as the cause of improvement.

## 3. Experiment rationale and falsifiable predictions

All entries below are PROPOSED and unexecuted in the current nuScenes revision series. Expected effects are hypotheses. Preserve the original evaluation region, labels and checkpoints. Every trained change needs an equal-budget unchanged continuation control. Do not promote all changes as one unexplained bundle.

### A1 — Class-aware proposal allocation

**Mechanism:** A global top-200 ranking can let one miscalibrated class consume the budget. Reserving 100 slots per class gives lower-scoring car candidates a chance to reach refinement.

**Why plausible here:** The epoch-6 diagnostic directly observed overwhelming pedestrian slot allocation. A near-correct candidate discarded by ranking cannot be repaired downstream.

**Prediction:** Car candidate recall rises at unchanged total count. Test raw recall by center distance and overlap, plus pedestrian losses, on epoch 18 before training.

**Failure/tradeoff:** A quota wastes slots in one-class scenes or harms crowded pedestrian scenes. It creates no missing geometry. Reject as a default if coverage merely shifts harm between classes. This is our diagnostic, not Sparse4D's policy.

**Weights/cost:** No weight changes; negligible added selection work. Reference: local proposal diagnosis and [S4 config] below for the distinction from its shared anchor bank.

### A2 — Center-quality prediction and confidence reranking

**Mechanism:** Semantic confidence answers whether a feature resembles a car; it does not necessarily express whether the box is well localized. Add a small quality head trained against assigned geometric error. Sparse4Dv3 uses a center-quality target exp(-3D center distance); its decoder combines predicted centerness with class scores when reranking.

**Why plausible here:** Our ranking can prefer confident inaccurate boxes over useful boxes. A separate localization-quality signal can rank the useful ones higher and protect the finite output budget.

**Prediction:** AP and precision at matched recall improve, even if raw box geometry initially stays unchanged. Report raw classification and quality scores separately. Decide explicitly whether quality is used before proposal selection or only afterward; upstream application is our adaptation.

**Failure/tradeoff:** The head can be miscalibrated or penalize hard distant objects. Lower false-positive count from dropping recall is insufficient. Supervise with appropriate valid assignments and document background handling. An IoU-quality target is an alternative experiment, not the original Sparse4D formulation.

**Weights/cost:** Keep base weights, initialize quality head, warm it up before optional joint fine-tuning. Small head overhead. Reference: [S4 loss], [S4 decoder].

### A3 — Heading quality and direct geometry correction

**Mechanism:** Separate two ideas. A heading-quality output predicts whether orientation is trustworthy; it primarily supports scoring/diagnosis. Increasing yaw-loss emphasis or adding footprint-corner supervision directly changes the pressure on predicted geometry.

**Why plausible here:** In the old diagnostic, replacing yaw alone at correct center cells raised passing car boxes from 142/213 to 199/213. Long narrow rectangles lose overlap quickly when rotated, even with a good center.

**Prediction:** A direct geometry-loss experiment should improve yaw across a fixed cohort and increase box overlap without shrinking matched coverage. Quality prediction alone need not improve yaw.

**Failure/tradeoff:** Extra yaw weight can compete with position/size learning; corner losses mix these errors and depend on corner correspondence/normalization. A rectangle's footprint does not uniquely determine forward direction; retain circular directional supervision. Sparse4D's yawness target is a coarse directional-agreement signal, not a precise angle-error regressor. Do not conflate auxiliary quality with a proven correction mechanism.

**Weights/cost:** Keep existing regression weights; initialize only any added quality head. Loss-only changes add training cost but no inference layers. Reference: [S4 loss]. Corner loss remains our separate hypothesis.

### A4 — Stride-4 image features

**Mechanism:** A small object may occupy only a few stride-8 cells; pooling discards edges useful for position and orientation. P2 supplies higher-resolution evidence. First expose P2 directly to refinement. For Stage A, separately test learned gated P2-to-P3 fusion while retaining stride-8 lifting.

**Why plausible here:** Distant/small object geometry is weak. More spatial detail can make different nearby positions and boundaries distinguishable before BEV projection.

**Prediction:** Improvement concentrated in small/distant-object recall and geometry, with added latency measured. Do not automatically double channels as well.

**Failure/tradeoff:** Upsampling cannot recover information already lost when source images were resized. P2 can contain less semantic context. Gated fusion is our adaptation, not the published reference. Direct stride-4 lifting has roughly four times the spatial sampling locations; it does not imply exactly four times end-to-end latency.

**Weights/cost:** Preserve existing FPN levels with explicit name mapping; initialize the new level and fusion. A zero-start residual gate can preserve the old pathway initially, but verify gradients reach the new branch as the gate opens. Adding P2 to refinement requires extending its level handling/embeddings. Reference: [S4 config], [SO config].

### A5 — Reliable depth supervision on objects

**Mechanism:** Our lifting deposits features according to predicted depth. Incorrect object depth distributes car evidence into the wrong BEV cells; downstream heads must compensate. Reliable surface labels directly constrain that placement.

**Why plausible here:** Current direct depth targets deliberately exclude semantic cars/people, including stationary instances. Box losses still supervise objects, but provide a more indirect learning signal to the depth head.

**Prediction:** Lower object-surface depth error followed by better center localization/proposal coverage. Report these separately; depth improvement alone need not improve detection.

**Failure/tradeoff:** LiDAR/image timing and moving-object misalignment can introduce worse labels. Restore only evidence whose projection and visibility pass an audit; preserve unknown masks. Surface depth is not box-center depth. Keep radial targets consistent with our unit-ray representation.

**Weights/cost:** Architecture unchanged; regenerate versioned targets and fine-tune all relevant layers from current weights. No required inference cost. Reference: [BD], a closer depth-to-BEV reference than Sparse4D's auxiliary-only depth branch.

### B1 — Repeated spatial/temporal refinement

**Mechanism:** The first inaccurate box samples imperfect image locations. After one correction, reprojecting the updated box can obtain better evidence for another correction. Historical features can fill gaps when current evidence is occluded.

**Why plausible here:** Our single block has only one opportunity to correct the initial estimate. The reference sparse decoder repeatedly samples and refines instead of treating the first geometry estimate as final.

**Prediction:** After training the existing one-block model, compare it against three passes with intermediate supervision. Measure step-by-step geometry and full-sequence stability/coverage. Shared weights are a compute/parameter hypothesis; sharing saves parameters but does not remove repeated runtime work.

**Failure/tradeoff:** Repeated wrong sampling can reinforce errors. Ego-motion alignment does not itself compensate independently moving cars. Temporal memory can introduce ghosts or stale features. Missing initial proposals remain a limit. Start with a frozen-base refiner experiment because KITTI joint B damaged the base; later joint fine-tuning is a distinct measured step.

**Weights/cost:** First train existing initialized nuScenes refiner; then reuse trained weights for repeated passes. Measure latency; three passes are our adaptation of a deeper reference. Reference: [S4 config].

### T1 — Appearance augmentation

Vary illumination/contrast/colour without changing geometry so the detector cannot rely as strongly on drive-specific appearance. This plausibly addresses the train/validation gap. Expect held-out gains under lighting variation, possibly higher training loss. Unrealistic distortion can destroy useful cues. Use moderate settings and consistent clip handling; test masking separately. All weights reusable, no inference cost. Reference: photometric augmentation/GridMask in [S4 config].

### T2 — Calibration-consistent geometric augmentation

Vary scale, crop, flips or scene rotation to increase geometric variety while preserving the exact relationship between images, projections and supervision. This can reduce reliance on fixed apparent positions/scales. Verify point correspondences and targets after every transform before training; inconsistent rays/poses create contradictory supervision. Sequence consistency is required where the transform is shared across time. All compatible weights reusable, no required inference cost. Reference: resize/crop/flip, box rotation and sequence-consistent settings in [S4 config].

### T3 — Lower backbone learning rate

Pretrained visual features already encode useful structure, whereas our newer heads must adapt to the task. A smaller backbone update can preserve transferable evidence while heads learn faster. Test head LR held fixed with backbone multipliers 1.0 versus 0.5, then 0.1 only if warranted. Too little adaptation can preserve unsuitable features and underfit. All weights reusable, new optimizer groups/state for the new experiment; no inference cost. References: 0.5 in [S4 config], 0.1 in [SO config]. This is a hypothesis, not proof that our current backbone forgot useful features.

### T4 — Driving-domain pretraining

Detection-trained road-image features may provide better object/scale representations than classification-only initialization, reducing sample demand. Evaluate this as a separate initialization study against ImageNet under matched downstream training. Replacing the trained epoch-18 backbone alone can break feature compatibility with its existing FPN/heads; it is not a routine resume. Gains may reflect extra pretraining data/compute, which must be disclosed. Same architecture can preserve inference cost. Reference: COCO/nuImages detection backbone checkpoint in [SO config]; the inspected Sparse4Dv3 R50 configuration itself uses ImageNet initialization.

## 4. Execution and checkpoint policy for a future authorized run

1. Bind the selected epoch-18 checkpoint, source/config/manifest hashes and baseline metrics. Refresh old proposal/geometry diagnostics.
2. Run A1 as an inference-only diagnostic; verify a small frozen-base Stage B learnability experiment.
3. Select the first learning intervention from A2/A3/A5 or the training recipe based on current failures. Equal-budget continuation is the control. Small screening runs are not convergence claims.
4. Add feature capacity or repeated refinement only as a separate candidate with measured latency.
5. Keep compatible weights; reset optimizer/scheduler/scaler/progress/history for changed experiments. Exact resume applies only to unchanged runs. Reject undocumented dropped/mismatched tensors.
6. Preserve both model-quality tradeoffs and all failures. Compare AP, precision at matched recall, proposal coverage, geometry on common cohorts, temporal gaps, road quality and latency. Do not tune on KITTI's sealed test or call nuScenes validation a new sealed test.
7. The recently discussed three-revision cap is a proposed bound for the new nuScenes study, not a reset of the completed KITTI R1–R3 record. No new training launched by this log.

## 5. Reference implementations

[S4 config]: https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
[S4 loss]: https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/mmdet3d_plugin/models/detection3d/losses.py
[S4 decoder]: https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/mmdet3d_plugin/models/detection3d/decoder.py
[SO config]: https://github.com/Yzichen/SuperOcc/blob/main/projects/configs/superocc/superocc-t_r50_704_seq_nui_48e.py
[BD]: https://arxiv.org/abs/2206.10092

References identify mechanisms, not proof of an accuracy gain in this project. Configurations were inspected during this discussion; local Sparse4Dv3 vendor code provides a preserved reference. No external performance numbers are equated with our custom AP or hardware timing.

## 6. Future change entry template

For each new entry record: date; experiment ID; parent checkpoint/hash; dataset and split; observed failure; hypothesis; exact code/config/target change; reference and adaptation differences; reused/new weights; trainable scope; matched control; budget; expected metric change; actual per-task/latency results; decision; rollback artifact. Status must be one of proposed, implemented, smoke-tested, trained, evaluated, retained or rejected.

Authorship: reconstructed and drafted by Codex from William's discussion and local evidence. Proposed mechanisms remain subject to William's design decisions. This file does not assert experimental results for unrun variants.
