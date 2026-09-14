# Readiness of the custom scene-model continuation

**Subsequent resolution (2026-09-13):** the checkpoint branching blocker in finding 1 is now fixed and verified for identical-architecture, identical-manifest full nuScenes parents. [Implementation and evidence](../../experiments/nuscenes/CHECKPOINT_BRANCHING.md). All 410 epoch-18 model tensors were preserved; real six-view inference matched exactly; two isolated updates/resume passed. The remaining findings below remain preparation or research items. Historical diagnosis is retained.

## Decision

The project is ready for a bounded continuation-preparation phase, but the proposed experiment suite is not ready for unattended long training. Preserve the selected nuScenes epoch-18 checkpoint. Repair experiment initialization and define the temporal and scoring comparisons before launching a new learning run. There is no evidence requiring wholesale replacement of ResNet, BEV, or the road branch.

This review inspects current implementation, saved checkpoints, previous experiments and reference implementations. It adds CPU-only checkpoint/target/mechanism checks, not new model evaluation or training. The separate KITTI lineage trained refinement and exhausted three revisions; the six-camera nuScenes lineage did not reach full Stage B. Those histories must remain separate.

## Verified state and scope

The best checkpoint is Stage A epoch 18, with 126,549 optimizer updates and no pending validation. SHA256: `28d570f3688ce1dd0b2f52c59ae92568abfb1a0ca634147642e5c88139013b50`. The latest checkpoint has internal epoch index 21 and next_step 13,644: 21 epochs completed, followed by 13,644 observations of epoch 22. It records 151,050 updates and no pending validation for that partial epoch. Its SHA256 is `d28d9d58fcffb7612f7c3f1ea9911798ad1da43b4a4c783c4946168d802f4bee`.

Both checkpoints include 34 refiner parameter tensors. Presence in a state dictionary does not mean those tensors trained: Stage A bypasses the refiner. Their manifest hashes agree with the full nuScenes lineage. Epoch 18's scheduler had decayed LR to about 1.46e-5; latest to 3.81e-6. A new experiment must deliberately choose its learning-rate restart, not inherit an almost finished schedule accidentally.

The selected model has a shared eight-channel geometry map, six cameras, stride-8 depth/context, 64 radial bins, and a 160×160 BEV. A label-only sample selected evenly by chronological metadata examined 256 training timestamps (3,600 objects) and 128 validation timestamps (1,531 objects). No images or predicted boxes were evaluated. The sample and all identities are saved in [CPU audit evidence](../../experiments/nuscenes/artifacts/continuation-readiness/audit.json). The runnable [audit script](../../experiments/nuscenes/audit_continuation_readiness.py) performs no optimizer updates.

## Findings that change the proposed plan

### 1. Weights-only continuation is currently blocked

**Confirmed blocker.** The nuScenes `train` entry point immediately raises if `initialize` is supplied, before the later weights-only loading implementation can execute. A safe CPU call reproduced the exact rejection: `Fresh nuScenes training forbids imported task checkpoints; resume only nuScenes continuation`. Our earlier claim that the new experiment pathway was already usable was incorrect. The underlying logic exists, but is unreachable through this entry point. [Trainer](../../experiments/nuscenes/code/src/fsd/engine.py)

Do not remove the guard indiscriminately. Replace it with explicit authorization of same-lineage nuScenes parents: validate dataset identity, camera and coordinate conventions, source/manifest hashes and declared changes. Separate exact resume, same-architecture warm start and architecture migration. Keep rejection of accidental KITTI/mini task checkpoints. New layers require a named migration map with an explicit initialized-tensor report, not a blanket non-strict load.

Acceptance before training: same-architecture loading produces equal predictions; optimizer/scheduler/scaler/history/progress are fresh for a new experiment; exact resume restores them; unexpected mismatches fail; interrupt/reload preserves the intended state. Fresh manifests for an audited target revision need an explicit permitted relationship, not silent hash bypasses.

### 2. An equal-epoch comparison can still be confounded

**Confirmed experimental-design gap.** The existing optimizer has one parameter group. A backbone LR multiplier is not wired into it. Each experiment needs actual named parameter groups, logged rates and correct scheduler behavior. Adding a new quality head also requires checking that its parameters reach the optimizer.

The proper Stage A control starts from the same epoch-18 weights with the same new LR schedule, data order, successful-update target and validation frequency as the candidate. Comparing a restarted candidate at 1e-4 against an exact-resume control near 1e-5 confounds the architecture with the LR restart. Equal wall time, equal frame exposure and equal optimizer updates answer different questions; use update/frame matching for learning and separately report compute. Small-screening superiority is not convergence.

### 3. Quality scoring is not yet a complete training specification

**Design blocker for A2.** Before implementation, specify where quality is applied, what its target means, and which examples supervise it. Stage A trains box geometry at target center cells, but inference selects predicted peaks. Training quality only at perfect center cells would miss the errors encountered at inference.

Train and evaluate the quality mechanism on actual proposals from the initial head. Match valid ground truth without treating unknown areas as negatives. Distinguish true background from duplicate detections of a real object: a duplicate can have high geometric quality even though one-to-one evaluation marks it false. Quality should estimate geometry; duplicate suppression handles duplication. Treat the target as detached in the first custom experiment so the head learns reliability without silently adding a second regression objective.

Use a small proposal-conditioned scorer with BEV appearance and predicted box encoding. A feature-only dense quality map cannot distinguish different box hypotheses at the same cell. This is a proposed adaptation; measure whether the extra encoding helps before claiming necessity. Initially rerank a fixed candidate set to isolate scoring. Using quality to change candidate generation is a separate comparison. A center-quality score can miss wrong dimensions/yaw, so measure overlap AP as well as center metrics. Sparse4Dv3 supplies a useful center-quality reference, not a complete solution to our geometry errors. [Quality loss](https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/mmdet3d_plugin/models/detection3d/losses.py)

### 4. Proposal quotas can exchange one failure for another

**Hypothesis, not a guaranteed remedy.** Fixed 100/100 allocation addresses cross-class competition, but it cannot cure repeated pedestrian peaks within the pedestrian allocation. The 1×1 pedestrian peak kernel was introduced to preserve nearby people; it also makes dense local candidates possible. Rerun raw candidates → suppression → final cap diagnostics on epoch 18.

Compare global 200 with fixed 100/100 at identical geometry and budget first. Report both class recall curves and unused/duplicate slots. If quotas help only a subset of scenes, test a minimum reservation plus shared remainder as a separately documented decoder choice. Do not sweep many policies and report only a winner. Preserve a low-cutoff candidate audit so confidence changes cannot conceal missing objects.

### 5. One geometry vector per cell is real, but currently not a broad failure explanation

**Confirmed structural constraint; small sampled impact.** Both class heatmaps read the same eight-channel geometry vector at a cell. Two different targets sharing a cell ask one regression output to satisfy different boxes; same-class collisions also collapse heatmap centers. A synthetic gather check verified identical geometry for identical cell indices.

The sample found zero collision frames among 256 training timestamps, and two among 128 validation timestamps, containing 10 of 1,531 objects (about 0.65%). No cross-class collision cells appeared. This is not a population estimate or proof of absence, but it does not support making separate class geometry heads the first revision. Class-specific geometry would not resolve same-class collisions. Count collisions over the full training cache only if the refreshed error audit implicates crowded cells.

### 6. Depth evaluation currently measures quantized static supervision

**Confirmed metric limitation.** Sixty-four uniformly spaced centers over 1–80 m are about 1.254 m apart. Evaluation reconstructs target distances from the stored bin IDs, rather than raw LiDAR range, and those targets exclude semantic vehicles/people. Therefore the present depth metric cannot establish accurate object-surface depth or faithfully quantify sub-bin range error. [Depth evaluation](../../experiments/nuscenes/code/src/fsd/engine.py), [configuration](../../experiments/nuscenes/configs/full.json)

This spacing is not a hard 1.254 m box-position limit: probability mixtures and downstream offsets can express intermediate positions. Do not double bin count reflexively. First retain raw valid radial distances for an isolated diagnostic target, report object and background errors separately, and audit temporal alignment. For A5, preserve old training targets and a new target version, with an invariant evaluation set. Otherwise changing labels changes both the learner and its ruler.

Nearest visible LiDAR surface must be determined before semantic exclusion; never replace an excluded foreground with background depth along that ray. Camera capture poses compensate ego motion, not independent car motion. Add reliable stationary/low-timing-error evidence first if dynamic compensation is unverified. BEVDepth motivates explicit depth supervision but does not prove our label changes safe. [BEVDepth](https://arxiv.org/abs/2206.10092)

### 7. Our temporal negative control is incomplete

**Confirmed control limitation.** The current mismatched-history diagnostic rolls stored appearance features while leaving boxes, timestamps and poses attached to their original positions. Thus it tests appearance-to-geometry correspondence, not whether the model uses historical geometry at all. An unchanged result does not prove temporal memory is ignored.

Conversely, jointly permuting full key/value tokens is largely invariant for attention over an unordered set. A CPU attention control changed outputs by only 1.04e-7 after such a permutation; it is not a useful negative history test.

Before claiming temporal gains, add separate controls for empty history, wrong appearance-to-position binding, perturbed historical geometry/ego alignment, stale history and scene resets. Keep the corruptions physically interpretable and report in-distribution normal-history results separately. Training a spatial-only control from the same initialization distinguishes learned temporal benefit from inference-time removal of a component the model expects. No future observations may enter any cache. [Current evaluator](../../experiments/nuscenes/code/src/fsd/engine.py), [memory construction](../../experiments/nuscenes/code/src/fsd/model.py)

### 8. Ego alignment and temporal smoothing are not equivalent

**Confirmed architecture gap, impact unmeasured.** Historical boxes are transformed by vehicle poses, but no explicit object velocity propagation is applied. At 2 Hz, even 5 m/s relative motion is 2.5 m per sampled interval. Historical geometry can therefore be several metres away from the current object. Time embeddings allow learned compensation in principle; they do not guarantee it.

The current memory admits spatially valid peaks rather than a demonstrated confidence/quality-filtered object bank, potentially storing many false candidates. Measure attention distance, valid-history occupancy, moving-versus-stationary errors, entry/exit behavior and ghost persistence. If memory pollution is confirmed, test a quality-limited history bank before adding more temporal layers. Motion-based association bias or velocity supervision is a later architectural candidate, not silently part of B1.

The model explicitly keeps the original proposal when every current spatial sample is invalid. Temporal evidence cannot change the returned box in that case. This was an accepted fallback, but it limits full current-view occlusion recovery. Decide separately whether to permit temporal-only correction with confidence decay; do not remove the fallback and create unbounded ghost persistence. [Refinement](../../experiments/nuscenes/code/src/fsd/model.py)

### 9. Extra refinement passes need explicit gradient and memory rules

**Design blocker for B1.** Specify whether updated boxes are detached before the next projection, whether block parameters are shared, and how intermediate losses are weighted. Geometry-dependent image sampling can carry gradients through coordinates; stopping them versus keeping them changes optimization.

Compare one trained pass with three shared passes using equal training exposure and separate latency accounting. Average or explicitly weight intermediate losses rather than tripling total loss unintentionally. Read past-frame memory consistently and update the state once after the final pass; do not insert the same timestamp repeatedly. Reset/new-object handling must work after every pass. Small residual initialization and a frozen-base warm-up reduce disruption, but do not prove long-run stability. Sparse4D's repeated decoder is architectural motivation; its full training stack and learned sampling differ. [Sparse4Dv3 configuration](https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py)

### 10. P2 and augmentation are not isolated configuration switches

**Confirmed implementation work.** Current images are cached at the training resolution; adding P2 cannot restore original source pixels. Keep source-resolution effects separate from FPN-resolution effects. Extending the pyramid changes indexed module identities and the refiner's three-level embeddings. Remap old P3/P4/P5 weights by meaning, and initialize only new P2-related tensors. Do not shift indices under a permissive load and accidentally misassign weights.

For image-only resize/crop/flip, transform the projection mapping and recompute rays/depth targets; metric ego boxes should not be flipped merely because the image was flipped. For a genuine ego/BEV coordinate augmentation, transform boxes, road raster, camera extrinsics and historical poses coherently. Avoid mixing those two operations. At present, loader images are [0,1] before encoder normalization: photometric code copied from a [0,255] pipeline needs adjustment. Sequence-consistent parameters should be reproducible across resume. Start with moderate appearance augmentation before geometry augmentation.

### 11. Our metric selection can hide the objective we are optimizing

**Confirmed protocol concern.** The saved selector averages road IoU, sidewalk IoU, car AP and pedestrian AP equally. A road gain can offset worse objects. Preserve that historical score for comparison, but register an object-improvement decision rule before new trials. Report both class APs, car priority, matched-recall precision and road non-regression. Do not change ranking rules after seeing outcomes. The original one-point/five-percent backend parity gate is not automatically a universal learning-model acceptance threshold.

For temporal geometry, use common matched object/time cohorts and also report disappearances, false tracks and unmatched objects. Conditional yaw improvement can be manufactured by dropping difficult detections. Near/far breakdowns need fixed distance bands and fixed ROI; smaller display regions cannot masquerade as a better detector. Check the two-class ontology: trucks, buses and excluded pedestrian subclasses should not be counted informally as car/person misses without acknowledging the label contract.

Validation has already been used for selection. A newly subdivided portion is a future confirmation panel, not an untouched test. Use scene-level paired comparisons/uncertainty, not independent-frame assumptions. Do not select hundreds of variants against the same 150 scenes. Preserve KITTI's sealed-test boundary. Official nuScenes test labels are unavailable locally.

### 12. Latency and batching assumptions need measurement

**Not established by this review.** No GPU smoke, full epoch-18 inference refresh or new latency benchmark was run. The CPU audit does not certify live CUDA readiness. All accuracy candidates need six-view scene timing with real cached history and scene resets, separately from decode/render. Measure before wider features or additional refinement become defaults.

Effective batch four via accumulation is not automatically equivalent to a diverse four-scene batch: consecutive samples can be strongly correlated, and per-frame positive normalization gives sparse and crowded frames different effective object weights. Log positive counts and task gradients before changing normalization; do not repeat failed KITTI balancing by default. Compare exposure and successful updates as well as epochs. No 30/60 fresh-prediction target has been demonstrated for a new variant.

## Revised experiment tracks and gates

| Track | Concrete work | Required result before promotion |
|---|---|---|
| P0: trustworthy continuation | Implement validated same-lineage warm start; fresh state versus exact resume; optimizer groups; named architecture migration; baseline equality and recovery | Gates pass on actual six-camera batches, with source and checkpoint receipts. This is preparation, not an accuracy revision. |
| D0: current failure baseline | Refresh epoch-18 proposal/ranking/yaw/position audit; negative-region/ontology audit; fixed validation scene panel; A1 global-vs-quota comparison | A measured failure ranking; no claim that old epoch-6 ratios describe current behavior. |
| R1: one Stage A learning intervention | Choose quality scoring if ranking dominates, direct geometry supervision if yaw dominates, or moderate augmentation if overfitting dominates. Keep other mechanisms unchanged | Equal-budget restart control; predefined car/person/road criteria; improvements exceed ordinary run/scene variability. Do not combine all three as a hidden recipe. |
| B0: complete original architecture | From the selected A parent, warm-start frozen-base single-block refiner; compare spatial-only and spatial+temporal; preserve exact base digest | Small learnability and full-held-out evaluation; valid negative history controls; no base regression or unreported ghost/coverage tradeoff. |
| R2: additional representation if needed | Choose audited object-depth supervision OR P2 based on remaining measured failure | Target-version invariance or checkpoint-migration proof; task gain and latency reported. Defer if B0 already meets practical needs. |
| R3: richer refinement if needed | Three passes or a filtered/motion-aware memory intervention, selected separately | A single causal comparison against trained B0, with intermediate losses/state handling defined and measured timing. |

B0 completes an existing untested nuScenes component; R1–R3 are at most three new substantive revisions. The names record intended order, not approval to run all variants. If the diagnostics favor completing B0 before R1, preserve that decision and rationale rather than force an unnecessary Stage A change. Screening arms and controls must have equal budgets; finalists should have replication when gains are small. Driving-domain initialization and a wider backbone are deferred because they complicate attribution and weight reuse.

## Recommended immediate action

Implement P0 and the corrected temporal diagnostics first, then run D0. This is preferable to relaunching the old controller, whose default path finishes Stage A and starts jointly trained Stage B without the revised controls. Keep the existing controller/automation stopped during preparation. P0 requires targeted code and actual workflow checks; D0 requires new inference authorization/execution under the continuation task. This review itself launches neither.

No requested improvement has yet been proved on epoch 18. The strongest new finding is operational, not a verdict that the architecture is too weak: the intended experiment branching path currently cannot run. Once that and the evaluation controls are corrected, the retained weights are a useful starting point for the bounded study.

## Evidence and reproduction

- [CPU audit JSON](../../experiments/nuscenes/artifacts/continuation-readiness/audit.json): checkpoint identities, entrypoint rejection, sampled collision counts and attention permutation control.
- Reproduce from repository root: `.venv/Scripts/python.exe experiments/nuscenes/audit_continuation_readiness.py`. Writes its own audit artifact; reads checkpoints/targets only; does not invoke model inference or optimizer steps.
- [Evolution and rationale](2026-09-13-evolution-and-experiment-rationale.md): decisions and source references across KITTI/nuScenes lineages.
- [Training engine](../../experiments/nuscenes/code/src/fsd/engine.py), [losses](../../experiments/nuscenes/code/src/fsd/losses.py), [model](../../experiments/nuscenes/code/src/fsd/model.py), [loader](../../experiments/nuscenes/code/src/fsd/nuscenes_data.py), [temporal reset](../../experiments/nuscenes/code/src/fsd/runtime.py).
- [Original proposal diagnosis](../../experiments/nuscenes/ARCHITECTURE_DIAGNOSIS_2026-09-13.md), [KITTI final selection](../../artifacts/final/selection.json).
- Horizon Robotics, Sparse4Dv3 official [configuration](https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py) and [quality loss](https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/mmdet3d_plugin/models/detection3d/losses.py), inspected 2026-09-13.
- Li et al., [BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection](https://arxiv.org/abs/2206.10092), 2022, inspected 2026-09-13.

The report separates confirmed implementation behavior, sampled evidence, proposed mechanisms and unmeasured effects. CPU mechanism controls cannot establish learned accuracy, and no fresh GPU/environment/storage acceptance is implied.
