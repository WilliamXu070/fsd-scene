# Paired scene inspection

The visual review now includes the actual source/compiled evidence, synchronized with the existing 3D prediction. This is a supplement to the completed experiment, using its **197 validation observations**. The original final test, source, checkpoint and replay remain unchanged.

Start from `C:/Users/William/projects/fsd`:

```powershell
.venv/Scripts/python.exe inspection/serve.py --port 8773
```

Open [the inspector](http://127.0.0.1:8773). If it is already running, just open the link.

The left panel is the existing prediction-only 3D replay. The right panel follows its exact frame and shows a selectable camera input with native pinhole/fisheye box projections, optional original LiDAR samples, compiled ground-truth ground cells and predicted ground/boxes on the same metric grid. Green denotes supported labels; orange denotes predictions. Unknown ground stays unknown. Camera images are the exact processed 704×256 model inputs, not the original full-resolution photographs. Projected cuboid edges are not silhouettes or occlusion masks.

Below, the table exposes dataset object IDs, predicted track IDs, confidence, dimensions, heading, center error and best overlap. A failure selector opens the before/after frames. The downloads contain the original compiled NPZ, source LiDAR BIN and full paired JSON with predictions, labels, road/depth targets, validity masks, calibration and source paths/hashes. JSON boxes follow shape N×7 (an empty list means zero boxes); NPZ preserves array shapes/dtypes. LiDAR overlays are uniformly decimated to at most 3,000 points within 80 m for display; the download is the original full point frame.

The comparison cutoff controls the inspector and 3D view together. Labels never enter the original prediction API. While a new reference loads, old reference panels are dimmed and marked pending. Play/pause, step and scrub are synchronized; this is saved-data inspection, not new model inference.

## Car consistency findings

Across 193 adjacent pairs with the same labeled car supported in both frames:

| Diagnostic | Metric cutoff 0.01 | Display cutoff 0.55 |
|---|---:|---:|
| Matched in both frames | 65 | 56 |
| Match lost | 26 | 28 |
| Match regained | 30 | 32 |
| Missed in both | 72 | 77 |
| Predicted track ID changed, among pairs matched in both | 36/65 | 30/56 |

A lost match means the prediction no longer meets the existing class-consistent BEV IoU≥0.5 criterion. It can reflect a missing detection, movement of the predicted box, or filtering; it is not automatically proof the object visually vanished. Missing annotation times are never bridged. Track-ID changes here are an observation-pair diagnostic, not official tracking metrics or unique physical-object counts.

At the display cutoff, matched-pair mean error changes are **0.571 m** for world-aligned centers, **0.145 m** for dimensions and **21.56°** for directed heading. These account for actual labeled motion and ego rotation; they describe only the 56 successfully matched pairs. Lowering the cutoff leaves substantial instability.

One concrete example is dataset car **13040**, source frames **740→745**, about **0.523 s** apart:

| Value | Frame 740 | Frame 745 |
|---|---:|---:|
| Label heading | −2.18° | −3.85° |
| Predicted heading | 156.26° | −27.00° |
| Predicted track ID | 3267 | 3279 |
| Label length | 3.988 m | 3.989 m |
| Predicted length | 5.253 m | 4.887 m |
| Confidence | 0.746 | 0.859 |

The directed heading-error change is **178.42°**. Treating the box axis as equivalent after a 180° turn reduces this particular change to **1.58°**: this example is largely a front/back flip, alongside an ID change and size error. A cuboid can retain good overlap while a rendered vehicle suddenly reverses. Annotation yaw does not independently prove physical vehicle facing.

## What this changes in the plan

1. Every visual review must retain source images, compiled targets and raw predicted numbers for the same frame. [Plan supplement](PLAN.md).
2. Report availability and ID stability alongside geometry errors; matched-only averages hide dropouts.
3. Investigate proposal misses/extra detections, association errors and heading ambiguity separately. The existing tracker uses class-consistent distance matching with an 8 m car gate and velocity from the last displacement. It returns IDs for current detections; it does not emit a maintained box during a missed observation or stabilize dimensions/headings. These are relevant design limitations, not a proven complete explanation of the errors.
4. A next implementation should first test stronger association against these labeled sequences, then consider short-gap persistence and track-conditioned dimension/heading filtering. Evaluate retained false objects as well as reduced flicker. A temporal filter cannot recover reliable detections from uniformly poor proposals. This is a proposal only: **no tracker changes, output smoothing or additional training were made.**

## Evidence and verification

All 197 paired payloads were compared with the original prediction JSONL and compiled NPZ arrays. Existing 114 car matches, 65 matched transitions and all three temporal means were reproduced exactly. Analytic dropout/ID/gap checks passed. Six live frames were checked for prediction API parity, byte-identical cache/LiDAR downloads and all 24 camera image hashes. Browser checks cover synchronization, camera/LiDAR toggles, thresholds, failure pairs, playback and the last frame.

- [Data and metric verification](../artifacts/paired-inspection/verification.json)
- [Car consistency numbers/events](../artifacts/paired-inspection/consistency.json)
- [Concrete heading case](../artifacts/paired-inspection/heading-case.json)
- [Browser QA](../artifacts/paired-inspection/qa-verified/report.json)
- [Actual visual review](../artifacts/paired-inspection/visual-review.json)

Recheck existing evidence with `.venv/Scripts/python.exe inspection/verify.py` while the server runs. `inspection/build.py` creates the paired validation evidence from saved artifacts and refuses to overwrite an already published bundle. No GPU or training is required. `inspection/qa.cjs` uses the installed isolated Chromium fallback after the integrated browser connection failed; its software rendering is not a performance benchmark.
