# Replay viewer

The local viewer consumes **prediction-only JSONL exports** from `fsd.engine.evaluate`. It never opens target NPZ files or model checkpoint pickles. It renders predicted flat road regions, car/person assets scaled to predicted boxes, and a fixed reference vehicle. Lane marks and inferred hidden objects are not invented.

## Run with actual predictions

```powershell
.venv/Scripts/python.exe scripts/serve_demo.py --predictions runs/baseline/val_predictions.jsonl --data-root data/kitti360 --metadata runs/baseline/replay_metadata.json --port 8765
```

Open `http://127.0.0.1:8765`. The server listens only on loopback. It can start before predictions exist; the page explicitly waits, then detects appended complete frames. It indexes byte offsets and reads one scene at a time, so the full road export is not retained in server memory.

Optional provenance JSON:

```json
{"provenance":"trained","checkpoint":"baseline/best.pt","note":"Recorded validation replay","bev_min":-40,"bev_step":0.5,"ground_z":-1.5,"ground_source":"Provisional mount assumption"}
```

Write `trained` only for a real trained checkpoint export. Otherwise use `untrained` or omit metadata, which displays **Training status unverified**. A checkpoint filename does not by itself prove training. `--metadata` accepts JSON only; `.pt` is never deserialized. `--fixture` overrides provenance to visibly synthetic geometry.

## Controls and representation

- Play/pause, single-frame previous/next, scrubber and 0.5/1/2x playback. Keyboard space toggles play; arrows advance frames.
- Orbit by dragging, zoom by scrolling, and select perspective or top view.
- Confidence slider filters objects for display only; it does not modify evaluation metrics.
- Ground and fixed reference-vehicle base use inference export `road_ground_z_m`, falling back to metadata `ground_z`; default -1.5m is an explicit provisional mounting assumption. Replace it with a static calibration derived from training data only, never per-test annotations. Predicted object coordinates remain unmodified.
- Metric grid uses 5m cells; x is forward, y is left, z is up. The scene displays original coordinates without snapping cars onto roads or lanes.
- Road colors: road gray, sidewalk light gray, other known ground muted green. Unknown cells, false `road_visible` cells (when supplied by the inference exporter), and predicted road confidence below 0.45 are transparent. `road_visible` must come from calibrated inference geometry, never ground-truth target masks. A camera-FOV mask indicates potential coverage, not guaranteed absence of occlusion. This is a display threshold, not ground-truth masking.
- Four camera panels load corresponding images from exported `image_paths`, or the dataset manifest's `samples[].images` mapping. Missing images stay explicitly unavailable. Camera image paths must resolve inside `--data-root` and have an image extension.
- Model-update timing is the recorded `inference_ms` from the prediction export. Playback speed and rendering cadence are not presented as perception FPS.

Normal scene API fields are explicitly whitelisted; accidentally present `ground_truth`, labels/target caches or arbitrary paths are never forwarded. `labels` in that interface means **predicted object classes**, not annotation targets. Geometry contains metric boxes, confidence, predicted classes, track IDs and road predictions.

## UI fixture and tests

```powershell
.venv/Scripts/python.exe scripts/serve_demo.py --predictions viewer/fixtures/ui.jsonl --data-root data/kitti360 --metadata viewer/fixtures/metadata.json --fixture --port 8766
.venv/Scripts/python.exe -m pytest tests/test_viewer.py -q
node --check viewer/app.js
```

The fixture contains 12 original synthetic frames and no camera photos. It is UI-development data, never a trained-model demo or validation result. Twelve passing server controls cover waiting, incremental/partial exports, prediction-field filtering, manifest image lookup, root traversal and file-type rejection, fixture/provenance separation, and real HTTP endpoint/static-file behavior.

Browser-plugin visual QA was attempted but its Node tool kernel failed before connection with `windows sandbox failed: helper_unknown_error: apply deny-read ACLs`. HTTP tests and JavaScript syntax checks pass. Final browser/computer inspection of actual trained predictions remains required; no screenshot-based visual pass is claimed.

## Design and assets

Cool neutral, prediction-first scene matching the reference's simplified road and stock objects. Original car/person meshes, three-quarter ego-relative view, quiet camera column, bottom replay timeline; no dashboard chrome pretending to be a driving controller. Color system: scene #F2F4F6, road #AEB6BD, sidewalk #D5D9DC, ground #B9C4BA, ink #30383E, accent #526C80. Bahnschrift with Segoe UI fallback uses installed Windows fonts. Three.js 0.180.0 and OrbitControls are bundled locally under their MIT license; see `viewer/ATTRIBUTION.md`.

Ground-frame correction: an independent training-data audit found that the pose origin is above road level (e.g. an annotated pedestrian base at approximately -1.34m). The previous display plane at z=0 was inconsistent with metric ego coordinates; metadata now controls the plane and reference asset together without snapping predicted boxes.

UI correctness follow-up: metadata `note` is now displayed prominently over the scene, so training-fit-only scope is visible. Playback uses elapsed seconds from the first scene and retains the original timestamp in a tooltip. Exact recorded inference times remain unchanged; a visible note distinguishes cold frames from the separate warmed benchmark. The road legend explicitly states that calibrated camera coverage does not prove observed ground.

Scalability follow-up: the server now retains at most two decoded scenes, and rapid scrubbing debounces/cancels superseded requests. A synthetic 1,622-frame, 1.10 GB stress probe retained about 44 MiB server RSS and measured ~40 ms median HTTP/JSON/four-image scrubs. This excludes browser rendering and perception. See [REPLAY_SCALABILITY.md](REPLAY_SCALABILITY.md) for reproducible evidence and limits.
