# Validation and backend optimization comparison

`scripts/compare_runs.py` reads two ordinary validation reports and optional paired benchmark reports. It never loads model weights, images, targets, or test data and never selects, overwrites, or deletes checkpoints. Report hashes record the exact JSON inputs. Use a fresh output path for every comparison.

```powershell
.venv/Scripts/python.exe scripts/compare_runs.py --baseline artifacts/evaluation/baseline-val.json --candidate artifacts/evaluation/candidate-val.json --output artifacts/comparison/accuracy.json
.venv/Scripts/python.exe scripts/compare_runs.py --baseline artifacts/evaluation/baseline-val.json --candidate artifacts/evaluation/candidate-val.json --baseline-benchmark artifacts/latency/baseline.json --candidate-benchmark artifacts/latency/candidate.json --output artifacts/comparison/backend.json
.venv/Scripts/python.exe -m pytest tests/test_comparison.py -q
```

The report-only Python API is `compare_reports(baseline, candidate, baseline_benchmark=None, candidate_benchmark=None)`. Results serialize with strict JSON, including when input metrics contain NaN/infinity: invalid values remain disclosed as strings and cannot pass.

## Accuracy evidence and limits

Both evaluation reports must identify a checkpoint and its SHA-256, dataset manifest SHA-256, configuration SHA-256, and source SHA-256. Only ordinary `val` evaluation is comparable. The manifest, metric and geometry/temporal matching protocols, frame counts, score cutoff, NMS threshold, detection cap, per-class ground-truth counts, and valid road-pixel count must agree. Changed checkpoint/configuration/source hashes are disclosed, because candidate weights or optimized code can legitimately change. A matching path alone cannot establish immutable identity.

Gated metrics are road and sidewalk IoU; separate car and pedestrian custom BEV AP at IoU 0.5; per-class 3D-center, ground-center, dimension and yaw errors; and all five temporal residual-change/rate errors. The allowed AP/IoU decrease is 0.01 on the fraction scale (one percentage point). Error metrics may rise by at most 5% relative to their baseline. An exactly zero error baseline allows only 1e-9 absolute error. One floating-point representable step at a calculated boundary handles arithmetic rounding without introducing another scientific tolerance.

Geometry and temporal metrics require nonzero matched coverage in both reports. Missing, null, nonfinite, out-of-range, or unsupported values cannot pass. Reports show every metric's baseline, candidate, delta, threshold and result. Geometry matched counts and temporal transition counts show changes explicitly without inventing another coverage threshold. These are conditional errors; a lower value from fewer matches must be read with the coverage and per-class recall. 3D AP25/AP50 are retained as nongating diagnostics. The approved primary object acceptance metric is explicitly BEV AP50, not a mixture of incomparable protocols.

Accuracy status is `not_comparable` for a known protocol/data mismatch, `not_verifiable` for missing comparison metadata, `fail` for a proven tolerance violation, `not_verifiable` for incomplete required metric evidence, and `pass` only when all required checks pass. Each missing field remains visible even if another field proves failure. No training or test report can be used for optimization acceptance.

## Timing evidence and interpretation

Without benchmarks the result can pass the accuracy gate but cannot establish a speed improvement. Supplied timing needs both reports. Each benchmark must bind to its own evaluation checkpoint/manifest/source hashes and exact `runtime_backend` values (`precision`, `compiled`, `stage`, `refinement_enabled`). This prevents accepting compiled/FP16 latency against accuracy measured only with different code or a different backend. Source hashes can differ between the baseline and candidate, but each candidate's timing and accuracy must describe the same source. Effective configuration hashes are reported rather than equated because the benchmark applies explicit loader/pretrained/runtime overrides.

Paired benchmarks need the same reported GPU, measurement scope, stage, refinement/history, image dimensions, four-camera input, one-scene batch, and decoding score/NMS/cap protocol. The replay score threshold may differ from the validation AP cutoff, provided both timing runs use the same disclosed threshold. Each of at least three paired repetitions needs 50 full-history warmup updates and 500 timed updates. Missing bindings, a shortened pilot, invalid latency numbers, or a profile-only report cannot establish performance acceptance.

The primary speed comparison uses uninstrumented combined wall latency, with each repetition's mean, median and p95 retained. A repeatable speed gain means the candidate has lower mean latency in every paired repetition; there is no invented minimum speedup. A descriptive aggregate uses the median of repetition means and its sequential FPS. The 30 FPS target is reported separately and does not force rejection of a verified improvement. GPU-name equality alone does not establish identical clocks, power, contention, or other physical benchmark conditions.

`optimization_eligible` requires passing accuracy, complete/comparable timing, and the repeated speed gain. This is an evidence result, not automatic model selection. Both configurations remain available for an explicitly documented manual speed/accuracy tradeoff. The CLI exits 2 when required accuracy/timing evidence fails or is incomplete; it exits 0 for a completed passing accuracy comparison with valid or omitted timing. Therefore automation must inspect `optimization_eligible`, not infer acceptance from exit code alone.

The test suite uses clearly synthetic reports only. It falsifies every accuracy boundary, wrong splits/protocols/manifests/counts, absent pedestrian or temporal evidence, NaN serialization, zero-error behavior, changed matched coverage, incorrect timing-to-model/backend binding, short timing pilots, and speed gains that hide accuracy loss.
