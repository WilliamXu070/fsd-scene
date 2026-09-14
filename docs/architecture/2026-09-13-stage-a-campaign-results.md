# Stage A continuation: measured outcomes

The user authorized a bounded continuation campaign after the checkpoint-branching repair. The retained model remains the original nuScenes epoch 18. No architecture addition earned adoption, and Stage B remains untrained. This entry records executed experiments; the earlier rationale remains the record of proposed mechanisms.

Parent checkpoint SHA256: `28d570f3688ce1dd0b2f52c59ae92568abfb1a0ca634147642e5c88139013b50`. Dataset: the existing six-camera nuScenes cache, 28,130 training frames from 700 scenes and 6,019 validation frames from 150 scenes. Original geometry, both object classes and ground outputs remain in scope.

## What was actually tried

| Track | Implemented test | Outcome and boundary |
|---|---|---|
| A1 | 100 proposals per class; separate pedestrian 3x3 peak suppression | Quotas barely changed accuracy. Peak suppression damaged pedestrian recall. Neither adopted. |
| A2 | Per-class rotated-BEV-IoU quality head; score enabled versus disabled on identical trained geometry | Enabling the current scorer worsened precision at matched recall. No adoption. This does not reject every quality-aware detector. |
| A3 | Double the existing circular yaw-loss contribution | Better than the regressed high-LR control, worse than epoch 18. Corner supervision and other geometry losses remain untested. |
| A4 | Zero-gated C2 adapter into P3, followed by an RMS-normalized repair | Raw adapter contribution was nearly absent. Normalization increased activity but did not produce a better model. This was not a stride-4 depth/lifting experiment. |
| A5 | Training-only object-surface depth availability/timing audit | Measured object points are available, but a verified instance/motion/visibility overlay is still required. No object-depth training intervention was run. |
| Appearance | Moderate brightness/contrast, shared across six views | No gain in the short high-LR screen. Broader augmentation remains untested. |
| Recipe amendment | Global restart LR `1e-4` versus `1e-5` | The lower rate avoided the large high-rate regression, but did not improve overall full-validation accuracy over epoch 18. This was not the proposed backbone-only LR multiplier test. |

The architecture/augmentation screens used 600 successful optimizer updates, each seeing the same 2,400 frames from 60 training scenes. Both LR conditions reached 2,000 updates / 8,000 frames from 199 scenes. Exact parent initialization, fresh branch optimizer state, resume lineage and finite weights were independently verified. All runs had zero AMP overflows; the 34 refiner tensors remained unchanged.

## Why the screening winner was not adopted

The 24-scene screening panel favored the lower LR: car AP reached 38.45% versus the parent's 37.78%. The selected checkpoint was frozen before evaluating the full validation set.

| Full validation | Car BEV AP50 | Pedestrian BEV AP50 | Road IoU | Sidewalk IoU |
|---|---:|---:|---:|---:|
| Epoch 18 | 37.46% | 2.02% | 85.58% | 57.70% |
| LR `1e-5`, 2,000 updates | 37.50% | 1.92% | 85.51% | 57.68% |
| LR `1e-4`, 2,000 updates | 31.50% | 1.80% | 84.13% | 53.65% |

The candidate failed the predeclared mean object AP improvement guard. Outside the current screening panel, car AP also fell slightly, from 37.39% to 37.30%. These small differences do not establish statistical degradation, but they do not justify a replacement. The lower rate is a better tested continuation condition than `1e-4`, not a newly superior model.

At approximately 30% car recall, candidate precision improves slightly, while pedestrian precision near 10% recall worsens. Pooled matched-track size residual variation rises about 12.9%, with changing matched cohorts. There is no uniform false-positive or temporal-stability improvement.

These are custom two-class validation metrics, not official nuScenes AP/NDS or a sealed test. The full set includes the screening panel, and the historical parent was selected using all 150 validation scenes. Single-seed partial-epoch continuation does not prove convergence or an optimal architecture.

## Handoff

[Complete results, matched latency measurements and commands](../../experiments/nuscenes/artifacts/stage-a-campaign-02/RESULTS.md). [Campaign implementation and gates](../../experiments/nuscenes/STAGE_A_CAMPAIGN.md). [Independent review](../../experiments/nuscenes/artifacts/campaign-verifier-01/full-results-review.md).

All experiment code is opt-in and all rollback checkpoints remain available. Original v0.1/v0.4 architecture records, epoch 18, canonical configuration and existing replay remain preserved. The next candidate intervention is audited object-surface depth supervision, compared against a matched low-rate continuation control. More epochs, larger features or Stage B attention remain hypotheses rather than established fixes.
