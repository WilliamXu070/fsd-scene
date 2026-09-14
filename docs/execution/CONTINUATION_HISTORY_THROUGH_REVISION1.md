# Active revision 1

Revision1 is running in exec session23526 (read-only observer15551) via scripts/run_refiner_revision.py, after exact real recovery and production50-update gates passed. Run artifacts/runs/revision01-refiner-only; log artifacts/revision01-refiner-only.log; stage09_revision01_refiner_only. Nine complete epochs, original24-epoch cosine horizon, same bestA initialization; only refiner parameters train, base parameter/buffer digest checked each epoch. No source edits or second GPU job while it runs. A gated follow-on diagnostics queue is waiting in execsession36634; afterrevision completion it ownsGPU for best/latest validation, bestimage/history/proposalcontrols and best/latest fixedtraining-fit probes. Queue status: artifacts/revision01-evaluation/status.json.

Production gate is preserved at artifacts/revision01-gate/production-backend.json:50updates,200frames,zero AMP skips, frozen base exact, onlyrefinergradients, checkpoint4737a14fb5a05fe08100a6f4f9b19a350ccd25741688b7c10efd7b239aa1afa4. First epoch was split: add11.7253359s gate work to resumed raw first-epoch timing for approximate work total. Actual current source SHA401e51afc29afe4072d7d196adc7108537a56e54d0342599125ef1acd07bfffa. Recovery report binds source, original config SHA706869c93fa43d31706cbb3854dd630e72315ea997d832f966d8fa05f4ce6e1b and exactA initialization. Allstate/progress and ninebaseoutput tensors passed bitwise under deterministic math-attention control; production gate is separate.

Baseline validation and training-fit queues79651/8722 both completed and exited; all earlier baseline launcher/observer sessions also exited. Training-fit A carAP.79386/pedAP.13780 (10/59matched), B carAP.64581/pedAP.004237 (1/59), on selected96trainobservations. FullBinitial-only carAP.33539 versusrefined.36731; emptyhistory.36342/mismatchedhistory.36728. Depth and base metrics fell duringjointB. This justified frozen-base revision1, NOT positive-loss rebalancing. The latter remains deferred. Revisioncount1/max3, ledger/docs/experiments/REVISION01_REFINER_ONLY.md contain fixed selection/budget.

The baselineA validation viewer is port8770, execsession70681, metadata at artifacts/postbaseline/baseline-a/replay-metadata.json. It renders actual197valpredictionframes. Browser screenshots in artifacts/postbaseline/visual-review/ show major unsupported object clutter; frame630 has100objects at.3, zeroat.95. Parent visually reviewed index121confidence.3. Model worker is CPU-only auditing scored/ignored object distributions, camera/FOV/edge conditions and ground-contact offsets; no newlabels/GTmasking or policy accepted. Metrics worker's recovery and confidence-curve tasks are complete/idle. Confidence-curves evidence: artifacts/diagnostics/confidence-curves/README.md. Camera/FOV/edge/groundcontact/road-only rules do not explain away clutter; no displaydefault or filteringrule was selected. Modelworker then ran a bounded96frameCPU initial-proposal matcher audit: artifactdirectory willbe provided in its final message. Savedraw-proposals.npz supports a single counterfactual +2*(1-BEVIoU) matchingcost analysis, currently inprogress; no second modelpass/sourcechanges or revision2 acceptance.

Continue afterrevision: ordinaryvalidation, fixedtraining-fit/proposal/temporaldiagnosis asneeded; selectevidence-basednextrevision orbestmodel, profile/optimize, freeze/testonce, finalUIQAandhandoff. Finalteststillsealed. Goalactive. Historicalnotesbelow.

---

# Current execution state

Full baseline completed: A24epochs, B9epochs (early stopping), bestB epoch3 SHA f6eec2e72181f0d28b9c99a09215f70998cb7713f01f044164447688286e0da9. B9671successful optimizer updates/4isolated AMP skips. Both completion reports and final six-plot sets exist. Original training launcher14763 and observer83211 exited successfully; do not poll/restart them.

GPU queue: validation session79651, then training-fit session8722. The first runs A/B ordinary validation exports, B image/history controls and full A/B proposal audits. The second waits for its pass and runs the fixed96-observation stratified training-fit diagnostic. Do not start another GPU job or modify project Python source until both queues finish. Queue state/artifacts: artifacts/postbaseline/.

Preliminary ordinary validation: A road.81578/sidewalk.49887/carAP.37422/pedAP0; B road.79422/sidewalk.46449/carAP.36731/pedAP0. B blank images carAP~.000026, shuffled cameras.00864, empty history.36342, mismatched history.36728. These are validation results; final test remains sealed. Full proposal/training-fit conclusions remain pending.

One positive-center-loss revision design is saved under artifacts/diagnostics/pedestrian-revision-design/. It is not implemented/accepted. Revision count0; max3. Before implementation review full B diagnosis, fix the experimental endpoint/selection criteria, and use a deliberate weights-only initialization (existing resume rejects changed loss config). The user's strict numerical-optimization parity criteria must not be misrepresented as measured when geometry has zero matches.

Continue all authorized work: controlled revisions if supported, full inference profiling/optimization, freeze selected model/source, single logical final test, final prediction-only UI/recording QA and truthful handoff. Goal remains active/incomplete. Old continuation below is historical.

---

# Live recovery continuation

Validation-only queue is waiting in exec session79651: `.venv/Scripts/python.exe artifacts/postbaseline/run.py`. It starts only after both baseline stage records pass, then owns the GPU sequentially for ordinary A/B validation exports, B image/history controls, and full A/B proposal diagnostics. Status: artifacts/postbaseline/status.json. Do not launch another GPU job while that queue is running.

Pedestrian depth audit is complete: artifacts/diagnostics/stage-a-pedestrian-depth-cpu/findings.md.39 conservative box-supported pixels in four observations have+3.49m bias/3.62m MAE; same-frame/view/depth-bin controls have-4.63m bias/6.13m MAE. No pedestrian-specific depth failure is established. All six cached depth maps reproduced exactly.

Full baseline is running again: launcher14763, observer83211. StageA PASSED24epochs/25791updates, bestepoch21 score.4222250 (road.8157833,sidewalk.4988938,carAP.3742231,pedestrianAP0). BestA SHA2560c2c8ffc34f951669011b9f00041e2606f627384d64a5f1354367f2ff38326d2. A completion report and final plots are saved in artifacts/runs/baseline-a/.

The first B attempt failed before50updates and is preserved under artifacts/runs/baseline-b-failed-amp-001. Reproduced FP16 atan2 reciprocal overflow was corrected only inside decode_proposals, with original epsilon placement/output dtype retained. Tested forward proposals were bitwise equal.27targeted model/loss/proposal tests passed, plus240real frames/60updates with zero skips and a separate actual-engine60-update gate. Source SHA at recovery:7332f451f9c1ea0745a8629abe4df82052de87bc54fae618db1762b2df9ba40d. Numerical correctness fix; substantive revision count remains0.

B resumed at epoch0,next_step240,updates60,AMPscale1024,history3; model/optimizer/history states all finite. Full B is incomplete. Its first epoch timing is split: the recovery gate did240frames in20.9814seconds of reported epoch work; resumed epoch_seconds excludes that portion and includes loader restart overhead. Add the saved portion for an approximate first-epoch work total, or use uninterrupted epoch2 for stable ETA. Do not relabel the raw log timing as a whole uninterrupted epoch.

GPU is owned by the running baseline. All workers are idle. Do not change model/training/data source or run another GPU job until it finishes. Model/parameter checkpointing and source snapshots are active.

Six-frame CPU pedestrian diagnosis (one identity): artifacts/diagnostics/stage-a-pedestrian-cpu/findings.md. Four GT neighborhoods have no peaks; other two rank720/677 within pedestrians, all candidate box IoUs0, no source/decoded class disagreement. This does not support quota-only fixes. Production CUDA/AMP full-validation diagnostics remain pending after B.

Independent browser timing instrumentation and regressions passed; docs/execution/RENDER_TIMING.md. Final selected-model render measurements remain pending. Provisional metadata now correctly points to preserved ground_plane.smoke.json; historical timing capture labels are explained by artifacts/render-timing/ground-provenance-note.json.

---

# Resumable execution checkpoint

Continue the full approved goal through training, validation, bounded revisions, optimization, one frozen final test, and inspected prediction-only replay. The active goal remains incomplete. Tiny-set learning and provisional UI QA are prerequisites, not completion.

## Current processes

- Full baseline launcher is exec session 14763: `.venv/Scripts/python.exe scripts/run_baseline.py`. Stage A is complete; Stage B resumed from its verified60-update recovery checkpoint. Old sessions91744/93321 have exited. Both have 24-epoch caps, minimum 8 epochs and patience 6. Do not start a second GPU job or alter model/training/data code during the baseline.
- Lightweight observer is session 83211. Its atomic latest snapshot is `artifacts/baseline-progress.json`; per-run `epochs.jsonl`, `train.jsonl`, best/latest checkpoints and `artifacts/stages/08_train_*.json` are authoritative. Do not restart a healthy process just because a session lookup fails.
- Updated provisional real training-fit viewer is port 8768. Isolated browser regression passed; browser was closed. This replay is explicitly 16 training frames and has visible quality defects.

## Passed prerequisites

- Native isolated Python 3.12.13, PyTorch 2.14.0+cu130 / torchvision 0.29.0+cu130; actual RTX 5070 sm_120 convolution, attention, backward and optimizer update passed. Global environment preserved.
- Full data and geometry gates passed: 4300 train / 197 validation / 1622 sealed-test scenes. Frozen manifest SHA256 `804f711286751ef5ed3bea1f04a9969c8f58d34cfc8ab05490e6b57daa9d9a0c`. 44 missing-fisheye exclusions recorded. 30767 consumed source hashes, 6119 caches, 24476 camera images checked. 54 non-test timestamps visually reviewed. Test inspection was mechanical only.
- Evaluator controls, actual pipeline gradients, GT-input isolation, exact inference reload and interrupted/resumed engine checks passed. Integrated prebaseline controls: 144 passed.
- Tiny-set A passed at 912 updates: road IoU .95082, car recall 1, pedestrian recall .9375. Tiny-set B passed after 59 successful updates: road .95015, car recall .97980, pedestrian recall .9375. These are training-fit results only.
- Four-camera native pinhole/Mei rays, radial depth and IMU-origin ego geometry verified. Car long-axis canonicalization corrected 69 parameterizations with cuboid geometry preserved; source annotations and original caches retained.

## Data and evaluation limitations

- Validation has only six pedestrian observations from one identity, none supported in camera03. Training has pedestrian support in all four views. Drive-held-out validation has limited geographic overlap with training and is not entirely novel geography.
- Ground supervision is sparse: about 6.57% of training BEV cells. Unknown regions are masked; ignored unknown-region predictions can still create visible UI clutter. Never hide them using GT masks in normal rendering.
- Nominal 2 Hz is every fifth source frame: measured median 0.522773862 seconds (~1.913 Hz). Actual timestamps drive alignment, metrics and replay.
- Final training-only render plane is -0.930941852176808 m relative to the chosen IMU origin, from all 4300 train scenes. Old smoke calibration is preserved. Frame5355's diagonal road is an actual right turn, not a projection error.
- Some official cuboids are coarse. Physical front/back heading is unverified. Do not silently change metrics, tighten labels or claim dense/surround accuracy unsupported by labels.
- All AP values use the documented custom KITTI-360 subset protocol, not official leaderboard metrics.

## Next actions

1. Let the baseline finish, monitor failures and checkpoint updates, and measure Stage B epoch time. Preserve A and B baseline artifacts. Isolated AMP overflow skips have recovered; diagnose repeated failures rather than blindly retrying.
2. Evaluate best A/B ordinarily on validation. Run `scripts/run_diagnostics.py` and `scripts/proposal_diagnostics.py` (validation-only, downstream GT) after the GPU is free. Review actual prediction visuals. Current substantive revision count is zero; maximum three.
3. The proposal helper now includes source-class/global-K membership, tied score-rank intervals and hypothetical per-class quota diagnostics; 12 CPU controls passed. It has not yet run on the full trained baseline. The data worker is independently auditing validation pedestrian geometry and training-order distribution; use its evidence when available.
4. Choose each revision from measured failure evidence. Preserve baseline and record hypothesis, controlled comparison, outcome and rollback checkpoint. Do not assume quotas, bigger backbones or altered losses will help without diagnosis.
5. Benchmark the selected candidate with 50 warmups, 500 full-history timed scenes and three repetitions. Profile all model/postprocessing stages plus separate decode/render costs. The old 20.57 ms hooked 30-scene training pilot is not final latency. Validate numerical optimizations against <=1 percentage point road/sidewalk IoU and per-class AP loss, <=5% geometry/temporal worsening; missing metrics do not establish parity.
6. Finish source changes before selection freeze. `scripts/freeze_selection.py` binds weights/config/manifest/source and ordinary validation report. Run final test only once logically through `scripts/evaluate.py --unseal-test`; failed/interrupted runs may resume only the same frozen identity. Do not tune on final test.
7. Export selected predictions, run real replay visual/operational QA, produce recordings, plots, final reports and one-command startup instructions. Deliver best verified model with unmet accuracy/latency targets stated honestly, then mark active goal complete.

## Useful artifacts and commands

- Architecture baseline `docs/architecture/2026-09-11-preliminary-v0.1.md` is immutable; v0.2 documents actual implementation.
- Data evidence `docs/execution/FULL_DATA_GATE.md`, `artifacts/data-audit/full-gate-report.json`.
- `scripts/plot_training.py --run artifacts/runs/baseline-a --evaluation-scope validation` refreshes six plots safely from growing logs. Eight plotting controls and real-log visual QA passed.
- `scripts/status.py` gives compact resumable status. `artifacts/experiments/ledger.json` records revisions and correctness fixes.
- User browser profiles must remain untouched. Viewer QA succeeded with isolated Chromium/CPU SwiftShader. Keep fixture versus real prediction evidence separate.
- Windows normal shell reads may fail the ACL helper. Authorized elevated exec works. Never delete unrelated files; keep 50 GB free. No dataset downloads remain necessary.

## Observed training snapshot

This is a dated snapshot, not live status; refresh the atomic monitor or epoch logs before making decisions.

```json
{
  "updated_at": 1789118607.6526022,
  "active_stage": "a",
  "stage_status": {
    "a": "running",
    "b": "pending"
  },
  "training": {
    "event": "train",
    "epoch": 18,
    "step": 3764,
    "total_steps": 4300,
    "updates": 19210,
    "elapsed_s": 200.8766643999843,
    "amp_scale": 2048.0,
    "amp_overflow_skips": 6
  },
  "latest_validation": {
    "a": {
      "epoch": 17,
      "seconds": 271.39639650000026,
      "score": 0.4054718638928383,
      "best": 0.40813873194367,
      "road_iou": {
        "road": 0.8178558505213988,
        "sidewalk": 0.497458920965122,
        "other_ground": 0.5702969577500613
      },
      "objects": {
        "car": {
          "ap": 0.3065726840848325,
          "recall": 0.42213114754098363,
          "gt_count": 244
        },
        "pedestrian": {
          "ap": 0.0,
          "recall": 0.0,
          "gt_count": 6
        }
      },
      "peak_allocated_mb": 1763.7802734375
    },
    "b": {
      "epoch": null,
      "seconds": null,
      "score": null,
      "best": null,
      "road_iou": {},
      "objects": {},
      "peak_allocated_mb": null
    }
  }
}
```
