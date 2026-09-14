# Camera-based 3D scene experiment

## Current expanded-data study

The additional-drive study is complete. **Expanded-height** is the selected model: 9,971 training scenes, 32.60% car AP / 79.53% road IoU on expanded validation, and 44.06 prepared-input scene predictions/s. Pedestrian detection remains unreliable. See the [current final report](experiments/expanded_training/FINAL_RESULTS.md), [commands](experiments/expanded_training/README.md), and [paired camera/source-data replay](http://127.0.0.1:8780/).

All three revisions in the additional study and its single fresh test on drive 0007 are complete. Original R2, original source and drive 0004 results remain preserved. The section below records the **historical four-drive study**; its selected model, metrics, timing and completed training budget apply to that earlier study.

## Historical study

Four calibrated cameras â†’ ResNet-50/FPN â†’ depth-based BEV â†’ road/object heads â†’ late spatial/temporal box refinement â†’ prediction-only 3D replay.

**Training, all three revisions, final testing, and replay QA are complete. R2 is the selected checkpoint; no further training is authorized.** The workflow runs, but pedestrian detection and faithful scene reconstruction remain poor. The [final report](artifacts/final/RESULTS.md) and [hashed handoff inventory](artifacts/final/handoff-manifest.json) identify the deliverables and unmet targets.

## Results and evidence

| Metric | Validation: 197 scenes | Final test: 1,622 scenes |
| --- | ---: | ---: |
| Road IoU | 81.58% | **87.91%** |
| Sidewalk IoU | 49.89% | **61.39%** |
| Car BEV AP at IoU 0.5 | 37.75% | **36.05%** |
| Pedestrian BEV AP at IoU 0.5 | 0.00% | **0.355%** |

Final-test pedestrian recall is **3/195 observations (1.54%)**. Scores use the custom KITTI-360 subset protocol and valid supervision only; they are not official benchmark scores. Test ran once after freezing the model/runtime. Use the [archived test report](artifacts/final/test.json) and [completion ledger](artifacts/final/test_ledger.json); do not rerun or retune against test.

Selected RTX 5070 eager-FP16 processing measures **28.79â€“28.94 predictions/s**, with **34.19â€“34.28 ms median** and **37.35â€“37.69 ms p95** across three repetitions of 500 timed updates after 50 full-history warmups. This is one four-camera scene per call on 32 cached validation inputs, including transfer, pose/history handling, road decoding, NMS and tracking. Image decoding, native camera capture/resizing, pose estimation, rendering and display are separate. **The 30 FPS target is unmet by the selected pipeline.**

Combined projection/NMS caches reached 44.22â€“44.64 predictions/s on the same benchmark, but remain experimental: four validation pedestrian geometry guards are unavailable, so full accuracy acceptance cannot be established. The frozen runtime keeps the original eager FP16 path. See [latency evidence](artifacts/final-measurements/reference/latency.json) and [runtime decision](artifacts/final/runtime-decision.json).

- [Frozen model selection](artifacts/final/selection.json), [selected checkpoint](artifacts/runs/revision02-overlap-match/best.pt), [frozen configuration](artifacts/final/configuration.json).
- [Training/validation diagnosis](artifacts/final-diagnosis/TRAINING_VALIDATION_DIAGNOSIS.md), [revision assessment](artifacts/revision03-assessment/assessment.json), [experiment ledger](artifacts/experiments/ledger.json).
- [Original architecture v0.1](docs/architecture/2026-09-11-preliminary-v0.1.md), [implemented v0.2](docs/architecture/2026-09-11-implementation-v0.2.md), [current handoff status](docs/execution/CONTINUATION.md).

## Open the recorded demo

If the server is already running, open [http://127.0.0.1:8772](http://127.0.0.1:8772). Otherwise, from this project directory:

```powershell
.venv/Scripts/python.exe scripts/serve_demo.py --predictions artifacts/final/test-scenes.jsonl --metadata artifacts/final/replay-metadata.json --data-root data/kitti360 --port 8772
```

Pause, scrub, step, orbit/top view, camera panels and confidence controls passed operational checks. [Assistant visual review](artifacts/final/replay-qa/visual-review.json) found implausible overlapping vehicle clusters, coarse road boundaries and unstable arrangements. QA completion is **not** a faithful-reconstruction pass. The [recording](artifacts/final/replay-qa/selected-replay.webm) covers the first 24 recorded scenes; the review sampled the video and fixed scene indices.

The viewer uses exported predictions and supplied dataset poses, without live inference or ground-truth substitution. Display confidence 0.55 was selected on validation; evaluation retains cutoff 0.01. Lanes, physical camera integration, driving control and Raspberry Pi deployment are excluded.

## Reproduce validation and timing

Use the project-local Python 3.12 / PyTorch 2.14 CUDA 13 environment. Dependencies are in [requirements.lock.txt](requirements.lock.txt); actual GPU checks are in [gpu_smoke.json](artifacts/environment/gpu_smoke.json). To recreate an environment with installed `uv`, use `powershell -File scripts/setup.ps1`. The working global Python was preserved.

The verified [dataset manifest](data/kitti360/manifest.full.json) contains train drives 0000/0002, validation 0003 and test 0004. See [data preparation](docs/execution/DATA.md). Validation commands below create fresh reproduction outputs; choose unused output paths if rerunning. They do not rerun final test.

```powershell
.venv/Scripts/python.exe scripts/evaluate.py --checkpoint artifacts/runs/revision02-overlap-match/best.pt --split val --output artifacts/reproduction/validation.json --predictions artifacts/reproduction/validation-scenes.jsonl

$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:NUMEXPR_NUM_THREADS='2'
.venv/Scripts/python.exe scripts/benchmark.py --checkpoint artifacts/runs/revision02-overlap-match/best.pt --split val --output artifacts/reproduction/latency.json --threshold 0.01 --warmup 50 --updates 500 --repeats 3 --cache-frames 32 --profile
```

CUDA scatter/ranking variability can cause small numeric differences. The frozen receipt and archived reports are the authoritative completed result. Display refresh and replay playback rate are not perception throughput.

<details>
<summary>Recorded training commands and initialization semantics â€” reference only</summary>

The authorized learning budget is exhausted. These commands document reproducibility; they are not instructions to start another run. Training also writes stage records, so use a separately authorized reproduction checkout to preserve the completed evidence.

```powershell
.venv/Scripts/python.exe scripts/train.py --config configs/baseline.yaml --stage a --run artifacts/runs/baseline-a

# Existing baseline engine convention: cross-stage --resume loads A weights and starts fresh B state.
.venv/Scripts/python.exe scripts/train.py --config configs/baseline.yaml --stage b --run artifacts/runs/baseline-b --resume artifacts/runs/baseline-a/best.pt

# Recorded R2 experiment: weights-only initialization, all training state fresh.
.venv/Scripts/python.exe scripts/train.py --config configs/revision02-overlap-match.yaml --stage b --run artifacts/runs/revision02-overlap-match --initialize artifacts/runs/baseline-a/best.pt

# Same-stage interruption recovery restores optimizer, scheduler, scaler, RNG, progress and history.
.venv/Scripts/python.exe scripts/train.py --config configs/revision02-overlap-match.yaml --stage b --run artifacts/runs/revision02-overlap-match --resume artifacts/runs/revision02-overlap-match/latest.pt
```

`--initialize` requires an empty run directory and an `experiment.id`/`experiment.initialization` matching the supplied checkpoint. Baseline configuration lacks those experiment fields; its existing launcher deliberately uses cross-stage `--resume`. Same-stage `--resume` restores full state. See [train CLI](scripts/train.py), [engine](src/fsd/engine.py), and [baseline launcher](scripts/run_baseline.py).

</details>
