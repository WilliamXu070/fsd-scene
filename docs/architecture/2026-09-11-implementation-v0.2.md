# Implemented 3D scene model v0.2

This is the implemented architecture and decision record. Preserve `2026-09-11-preliminary-v0.1.md` as the original design. The user-approved execution plan controls scope; actual accuracy, runtime and stage completion are recorded separately under `artifacts/`.

**Completion status:** baseline A/B, all three revisions, frozen final test, and operational/assistant replay QA are complete. R2 is selected; original eager FP16 remains the runtime. Pedestrian performance and faithful reconstruction remain inadequate. No further learning is authorized; final inventory packaging remains pending.

## Implemented flow

Four calibrated RGB views -> shared ImageNet ResNet-50 -> 128-channel FPN at strides 8, 16 and 32 -> 64-bin radial depth plus 64-channel context -> depth-weighted pooling into BEV -> compact 128-channel BEV CNN -> road and initial object heads -> late spatial/temporal object refinement -> geometric tracking -> prediction-only replay.

The depth/context heads consume FPN stride-8 features, which already include top-down contributions from coarser levels. The other levels are also sampled during object refinement. No full rich voxel volume is retained.

Temporal attention occurs after initial proposals. Cached object features are detached. Baseline and R3 training fine-tune current-frame convolutions; the selected R2 experiment instead freezes the Stage-A base and trains only refinement. Ground-truth centers are never inference inputs. There is no output smoothing.

## Geometry and outputs

- Input: all four cameras, RGB 704 x 256; native pinhole and Mei fisheye geometry, resized calibration and unit camera rays.
- Ego coordinates: dataset IMU origin, x forward, y left, z up. Boxes use geometric centers, length/width/height, and yaw about z.
- Depth: radial range from each camera center, including fisheyes. There are 64 evenly spaced bin centers from 1 to 80 m. Training ignores unsupported/out-of-range samples. Current standard depth metrics use quantized LiDAR bin centers; raw-range diagnostics are labeled separately.
- BEV: 160 x 160 cells, 0.5 m per cell, spanning [-40, 40) m on both horizontal axes. A 64-to-128-channel stem precedes three 128-channel residual blocks.
- Ground classes: road, sidewalk and other known ground. Unknown supervision is masked. A calibrated potential-camera-FOV mask limits rendering; it does not certify visibility through occlusion or supply ground truth.
- Objects: car and pedestrian center heatmaps, subcell offsets, vertical center, log-dimensions and sine/cosine heading. Keep 200 initial proposals and at most 100 total final detections after class-consistent oriented BEV NMS and confidence filtering.
- Refinement: sample the center and eight corners in valid current views at all three FPN levels. A 128-channel, four-head block performs sparse spatial attention followed by temporal attention over up to three pose-aligned cached observations. Predict residual geometry and confidence. Entirely invalid spatial evidence retains the original proposal.
- Temporal state resets on sequence changes, unavailable history or discontinuities. Timestamp subtraction precedes FP32 conversion. Tracking uses class-consistent world-coordinate geometric association and supplied poses.
- Replay ground height: -0.930941852176808 m in the chosen ego frame, estimated only from 4,300 training scenes and 74,462,835 labeled road points. Actual 3D object targets are not flattened or shifted. The earlier 125-frame estimate (-0.9343111743493324 m) is retained in `data/kitti360/ground_plane.smoke.json` for old training-fit replay provenance.

## Environment, data and training

The isolated native-Windows Python 3.12 environment uses official PyTorch 2.14.0 CUDA 13.0 and torchvision 0.29.0 wheels. Actual RTX 5070 convolution, attention, backward and optimizer-update checks passed. The global environment is unchanged. Exact dependencies and hardware evidence are in `requirements.lock.txt` and `artifacts/environment/`.

Official ZIP members were selectively acquired through verified HTTP ranges. Member CRC, source hashes and processed hashes are recorded; this is not a claim that every entire multi-gigabyte archive was downloaded and hashed. Keep at least 50 GB free and preserve source provenance.

The immutable full manifest uses drives 0000/0002 for training, 0003 for validation and 0004 reserved for the once-only final test: 4,300 / 197 / 1,622 eligible scenes. Forty-four timestamps missing released fisheye images were excluded explicitly. Nominal 2 Hz is implemented as every fifth native frame; median training spacing is 0.522773862 s (about 1.913 Hz). Alignment and replay use actual timestamps.

Only supported semantic windows, exact dynamic annotation times and current LiDAR evidence supply targets. Unknown regions are not negative labels. Training ground supervision covers about 6.57% of BEV cells on average. Validation contains six pedestrian observations of one identity and partial geographic overlap with training; it cannot establish broad pedestrian or entirely unseen-location generalization.

Baseline optimization uses AdamW, learning rate 1e-4, weight decay 1e-2, clipping at 5, seed 42, FP16 mixed precision, one scene per microbatch and accumulation to four. Pretrained backbone batch-normalization statistics remain frozen. Stage A trains initial outputs; Stage B adds late refinement. Each has a 24-epoch cap, minimum eight epochs and patience six. Selection equally weights road IoU, sidewalk IoU, car BEV AP50 and pedestrian BEV AP50. Per-task losses and metrics must accompany the combined score.

## Verified implementation corrections

1. Native axis conversion, camera resizing and radial depth preserve pinhole/fisheye calibration. Numerical round trips and real projection audits passed.
2. FP32 geometric/scatter accumulation is used under AMP. CUDA scatter order can still perturb weak rankings; exact reload/isolation checks use the documented deterministic control.
3. The default AMP scale of 65,536 overflowed real gradients. Initialization at 1,024 plus explicit skip/backoff recovery passed actual updates. Skipped updates are logged separately; persistent failure stops for diagnosis.
4. Inactive loss zeros reduce in FP32: summing 51,200 half-precision logits could overflow before multiplication by zero. A regression check covers this.
5. Checkpoints preserve model, optimizer, scheduler, scaler, RNG, epoch/step and temporal state. Real interrupted/resumed training was compared with an uninterrupted control; deserialization alone is not the gate.
6. Final test requires a frozen checkpoint/configuration/data/source identity and a single logical-run ledger. Model-selection diagnostics stay on validation.
7. Some official car cuboids use the short side as their first axis. For cars only, width greater than length is canonicalized by swapping the dimensions and adding pi/2 to yaw. Center, height and occupied geometry are preserved; physical front/back remains unverified. Sixty-nine observations across five identities changed among 12,557 recomputed drive-0000 targets. All other parameterizations and the tiny-set targets were identical. Maximum corner-set change was 1.19e-6 m and minimum BEV/3D IoU exceeded 0.99999934. Coarse primitives were flagged, not silently tightened or removed. Original annotations/caches and the original-axes manifest remain available.
8. After full Stage A, the first Stage B attempt failed after 224 frames. FP16 atan2 backward overflowed the reciprocal of a small nonzero squared sine/cosine norm, producing NaNs even with zero upstream gradients. Decoded yaw now evaluates atan2 in FP32, preserving the original pre-cast epsilon addition and casting back to the original dtype. Tested proposal outputs were bitwise equal. The fix passed 240 real frames/60 updates without skips, 27 targeted controls and a separate 60-update actual-engine recovery gate. The failed run is preserved under `artifacts/runs/baseline-b-failed-amp-001`.

These are correctness fixes or disclosed implementation conventions, not substantive post-baseline learning revisions. The maximum is three; actual decisions belong in `artifacts/experiments/ledger.json`.

## Evidence references and limits

- Data/calibration: `docs/execution/FULL_DATA_GATE.md`, `artifacts/data-audit/full-gate-report.json`.
- Car-axis correction: `artifacts/data-audit/car-canonicalization-accepted.json`, `docs/execution/BOX_SUPERVISION_AUDIT.md`.
- Render calibration: `artifacts/integration/full-ground-calibration.json`.
- Angle-gradient correction: `artifacts/diagnostics/stage-b-backward/`.
- Contracts, losses and metrics: `docs/execution/CONTRACTS.md`, `MODEL.md`, `METRICS.md`.
- Live completion and experiments: `artifacts/stages/`, `artifacts/experiments/ledger.json`, `docs/execution/CONTINUATION.md`.

Tiny-set learning is a correctness gate, not held-out accuracy. AP uses the custom subset protocol, not an official leaderboard score. Geometry/temporal errors are conditional on matched objects and must retain coverage counts. Replay uses supplied poses and recorded images; it is not a physical camera system. Display refresh and cached replay cadence are not fresh perception throughput. The completed full timing protocol measured 28.79â€“28.94 predictions/s for the selected cached-image pipeline; the 30 FPS target is unmet.

## Revision 1: frozen base during refinement training

The first post-baseline learning experiment changes training scope only. It starts from the exact best-A checkpoint and trains `refiner.*`; all initial scene parameters and buffers remain frozen. This is a deliberate exception to the baseline's trainable current-frame convolutions, not a change to the inference architecture. Constant image/BEV features still support gradients into refinement projections, attention and residual heads.

Evidence: completed Stage B degraded road/sidewalk and initial car accuracy, and reduced the fixed training diagnostic's pedestrian matches from 10/59 to 1/59. The experiment tests preservation of the stronger initial representation. It does not repair initial pedestrian peaks. Nine epochs retain the original 24-epoch cosine schedule. The baseline remains preserved; the completed revision was not promoted: its best ordinary validation composite was 0.420788, below baseline A 0.422218. Pedestrian validation AP remained zero.

Enforcement: explicit trainable-scope policy, optimizer restricted to trainable parameters, base parameter/buffer SHA checked each epoch, isolated revision stage records, source/config/initialization-bound recovery gate, and unchanged-base inference parity. Actual stop2/resume4 versus uninterrupted4 state passed bitwise under deterministic math attention; a separate 50-update production-backend gate passed with zero skipped updates and unchanged base state. Details and selection criteria: `docs/experiments/REVISION01_REFINER_ONLY.md`, `artifacts/revision01-gate/`, `artifacts/experiments/ledger.json`.

## Recorded revision2 and benchmark protocol changes

Revision2 changes only the training matcher with2*(1-rotatedBEVIoU), using revision1 frozen-base protocol from the same bestA weights. Explicit weights-only initialization is now distinct from strict complete-state resume. Fixed matching-weight0 behavior is preserved. See `docs/experiments/REVISION02_OVERLAP_MATCH.md`. Benchmark comparison now binds actual sensor workload and accepted frame order/frequency; different cached scenes cannot masquerade as a speedup. R2 completed nine epochs and its epoch-1 checkpoint was selected by the unchanged composite: repeated validation 0.423047, only 0.000829 above preserved A. Pedestrian validation AP remained zero; this small gain is not broad superiority. Final measured runtime and once-only test results are recorded below.


## Revision 3: positive center balance with a paired control

The final learning revision tests only positive center-loss normalization. The default remains the original global positive-count reduction. `loss.center_positive_normalization: present_class` averages positive focal loss within each present class, then across those classes; the known-negative term remains normalized by the original global positive count. Single-class and no-positive frames keep the original expression. Eight detached positive/negative sums/counts are exposed for diagnostics without entering the optimized total.

Both new runs start from exact R2best weights with fresh optimizer/scheduler/scaler/RNG/history/progress, and train the full network for eight fixed epochs under the same cosine8 schedule. The loss option is the only model-training difference between arms; geometry, masks, inference and late temporal attention are unchanged. This completed controlled test did not solve the missing-pedestrian-proposal problem. Fixed epoch8 comparison, learning-specific guards and unavailable pedestrian geometry are recorded before training in `artifacts/revision03/decision.json`. No fourth learning revision is authorized. Final backend optimization acceptance remains separate.


## Completed learning and frozen selection

Baseline A completed 24 epochs (best epoch 21); baseline B completed nine (best epoch 3). R1 trained refinement only for nine epochs and was not promoted. R2 used the same frozen-base protocol plus overlap-aware matching, completed nine epochs, and its epoch-1 checkpoint was selected. R3 completed the prespecified eight epochs per arm: control 8,597 accepted updates/three AMP skips, balanced 8,598/two skips. Equal scheduled frames do not imply exactly equal successful updates.

R3 fixed-endpoint composites were 0.417916/control and 0.418863/balanced versus 0.423047/R2. Balanced failed the +0.02 pedestrian-AP and +0.05 proposal-coverage improvement guards; its car AP decline versus R2 also exceeded the one-point guard. Initial pedestrian candidates increased 122â†’440 across validation, but neither arm placed a center within 2 m of any of the six valid pedestrian observations. More training-fit matches did not transfer to validation. The three-revision learning budget is complete.

The selected checkpoint is `artifacts/runs/revision02-overlap-match/best.pt`, SHA256 `3dfa52ea6a14d3f080a7d8507489c550419425969e8006773a91502706f9aaf0`. Frozen model/config/data/source identities are in [selection.json](../../artifacts/final/selection.json). The [training/validation diagnosis](../../artifacts/final-diagnosis/TRAINING_VALIDATION_DIAGNOSIS.md) predates test and remains a training/validation-only record.

Initialization semantics follow the actual engine. A recorded fresh experiment uses `--initialize`, with matching experiment metadata, and resets optimizer/scheduler/scaler/RNG/history/progress. Same-stage `--resume` restores all state. The legacy baseline launcher explicitly supports Aâ†’B via cross-stage `--resume`, which loads weights but starts fresh B training state. See the verified reference commands in [README](../../README.md); no new training run is authorized.

## Final test and measured runtime

The frozen original eager-FP16 model ran test drive 0004 once: 1,622 scenes, ledger status complete, attempt 1. No test-based model or threshold selection followed. [Test report](../../artifacts/final/test.json), [once-only ledger](../../artifacts/final/test_ledger.json).

| Final-test metric | Result |
| --- | ---: |
| Road / sidewalk IoU | 87.91% / 61.39% |
| Car BEV AP50 / recall | 36.05% / 45.40% (1,708/3,762 observations) |
| Pedestrian BEV AP50 / recall | 0.355% / 1.54% (3/195 observations) |
| Car / pedestrian 3D AP50 | 22.43% / 0.0394% |
| Depth MAE against quantized LiDAR radial-bin targets | 1.627 m |

Conditional geometry/temporal metrics and distance breakdowns remain in the full report. They do not describe missed objects; pedestrian geometry has only three matched observations.

The RTX 5070 benchmark used one four-camera 704Ã—256 scene per call, three cached previous observations, 32 cached validation inputs, 50 accepted full-history warmups and 500 timed updates in each of three repetitions. Selected processing throughput is 28.79â€“28.94/s, median 34.19â€“34.28 ms and p95 37.35â€“37.69 ms. Model-only CUDA medians are 19.88â€“19.95 ms. The peak reported PyTorch allocations are 228.86 MiB allocated/334 MiB reserved; these are not total device VRAM usage.

Processing includes host/device transfer, model execution, supplied-pose/history handling, road decoding, NMS and tracking. It excludes processed-image decoding, native capture/resizing, pose estimation, road mesh construction and display. Processed PNG decode median was 10.92 ms, with RGB/calibration conversion measured separately. The cold first call and history-reset timings are preserved in [latency.json](../../artifacts/final-measurements/reference/latency.json); steady-state results are not cold-start or full live-camera latency.

Combined cached lifting and NMS reached 44.22â€“44.64/s in the matched workload. All 13 available accuracy guards passed, but four pedestrian geometry guards were unavailable because validation had zero matched pedestrians. Complete optimization acceptance is therefore unverifiable: caches stay experimental, and the final runtime uses original lifting/NMS, eager FP16, no compilation. See [runtime-decision.json](../../artifacts/final/runtime-decision.json).

## Final replay inspection and delivery

The final viewer replays exported test predictions with supplied dataset poses at port 8772. Display threshold 0.55 was selected on validation before test; normal metric cutoff remains 0.01. It does not hide ground-truth geometry in rendering.

Operational checks and assistant visual review completed: synchronized camera panels, frame changes, controls, prediction/API parity and recording were checked. The inspection found implausible vehicle clusters/overlap, coarse road boundaries and unstable adjacent-frame arrangements. Even default confidence 0.55 showed 28 cars at the inspected first frame; lower-threshold diagnostic views showed 100. Peripheral predictions often lack annotation coverage, so visual concerns are not automatically numerical false-positive labels.

[Automated QA](../../artifacts/final/replay-qa/qa-report.json) retains its historical visual-review-pending status; the subsequent [assistant review receipt](../../artifacts/final/replay-qa/visual-review.json) completes that inspection with adverse findings. It covers 11 screenshots, sampled recording frames and a separate default-threshold view, not every test scene. The [24-scene recording](../../artifacts/final/replay-qa/selected-replay.webm) is recorded playback; CPU SwiftShader submission/rAF timing does not measure hardware display or fresh inference.

The [final report](../../artifacts/final/RESULTS.md) and [hashed inventory](../../artifacts/final/handoff-manifest.json) identify the completed deliverables. No additional training or test execution is authorized. Use validation reproduction and the demo startup commands in [README](../../README.md); preserve v0.1, all baselines/revisions and the once-only test evidence.
