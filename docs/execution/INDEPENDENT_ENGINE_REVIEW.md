# Independent engine/metrics integration review

Read-only review on 2026-09-11; no GPU task or sealed-test evaluation performed. These findings describe the source snapshot at review time, before parent fixes.

## Correct integration

- Evaluation receives decoded model-generated proposals after class-aware NMS; no ground-truth centers are passed to decode. Geometry targets are used only by loss/metrics.
- Road logits are argmaxed before comparison; unknown targets remain ignored. GT identity and pose are supplied to residual temporal metrics, while tracker IDs are separately exported for visualization.
- Depth diagnostics compare expected radial depth with quantized LiDAR bin centers and explicitly label that limitation. Pooled RMSE correctly aggregates squared per-frame RMSE weighted by valid-pixel counts, rather than averaging RMSE directly.
- Custom score protocol records confidence cutoff0.01, oriented-BEV NMS0.5 and max100 retained detections. Geometry and temporal summaries remain conditional on matched detections, so recall must be reported alongside them.
- Learnability evaluates actual proposals on the selected training subset, with separate car/person recall and road IoU >=0.9. This proves tiny-set fit only; it is not validation generalization. Lower-confidence extra detections can still exist because the stated gate is recall-based.
- Source/manifest snapshots exist, overfit data uses a named manifest, and the fresh-process restore diagnostic verifies actual model/optimizer/scheduler/scaler/RNG/history restoration. That diagnostic explicitly does not execute `engine.train`'s resume control flow.

## Ordered fixes / safeguards

1. **Mid-epoch update-limit resume loses remaining samples.** In `engine.train`, reaching max_updates saves `(epoch,next_step=step+1)` then breaks the inner loop. The epoch-end path then steps the scheduler, validates and overwrites latest/best with `(epoch+1,next_step=0)`. Resuming skips the unconsumed remainder. Preserve next-step/history/current-epoch and do not step the epoch scheduler until the loader is exhausted. Add an engine-level interrupted-versus-uninterrupted control; checkpoint restoration alone does not cover this.
2. **Sealed-test guard is only an explicit flag.** `scripts/evaluate.py` rejects test by default but accepts `--unseal-test` repeatedly, accepts diagnostic variants on test and does not bind to immutable checkpoint/manifest hashes. Before the final run, persist a frozen-selection receipt and a one-logical-run ledger with resumability after interruption; prohibit model-selection diagnostics on sealed test. This is a procedural gap, not evidence that the test has already leaked.
3. **Resume provenance compatibility is not enforced.** Existing run provenance is not rewritten (good), but incoming config/data pointers can differ from that provenance or checkpoint without rejection. Validate immutable model/data/selection fields on resume, allowing only explicitly documented run-control changes. Full training must use a named frozen manifest, not the actively replaced default pointer.
4. **Nonfinite depth predictions need a clear failure record.** `depth_metrics` intentionally returns null numerical summaries when any valid target has an invalid prediction. The engine's final weighted aggregation multiplies these nulls by counts, causing TypeError. Fail with the invalid-prediction count and diagnostic context rather than an incidental aggregation error; do not silently drop bad depths.
5. **Name the image control accurately.** Current `shuffled` diagnostic permutes camera images while keeping their calibration order. This is a camera-association falsification, not unrelated-frame shuffling. Report it as such; blank images provide the complementary image-dependence control.

These issues were reported to the parent for fixes in parent-owned code. Metrics/viewer source was not used to inspect final-test predictions.
