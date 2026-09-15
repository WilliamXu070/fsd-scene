# Recorded model viewer

A static adaptation of the existing Three.js viewer. It uses the same procedural car/person meshes, road colours, metric axes, and orbit/top-view conventions. It adds synchronized image loading, camera selection, predicted-cuboid image projection, object inspection, on-demand rendering, and article embedding. There is no browser inference, server API, localhost dependency or training connection.

## Pinned recording

- Retained nuScenes epoch 18, Stage A. Spatial/temporal refinement is bypassed; newer experiment branches are not adopted.
- Checkpoint SHA256: `28d570f3688ce1dd0b2f52c59ae92568abfb1a0ca634147642e5c88139013b50`.
- First 40 chronological validation keyframes, all from scene-0003, chosen before inspecting predictions. Six images per frame, copied byte-for-byte from the model's prepared inputs. About 19.5 seconds of source time; roughly 14.4 MiB for the full recording.
- Boxes, confidence, class, track ID and road output are actual checkpoint predictions. Road display RLE is lossless relative to the existing renderer's predicted confidence >=0.45 and calibration-FOV mask. It is not ground truth. Confidence 0.55 is the initial display filter; lower-scoring exported predictions remain available.
- Camera projection uses each frame's intrinsic and camera-to-ego matrices, clips cuboid edges at the near plane and image boundary, and does not assume a visible silhouette. Ground-truth arrays never enter the public recording.
- The earlier article metrics describe KITTI-360. This clip does not update those results or establish a new benchmark. Pedestrian detection and scene reconstruction remain unreliable.

Metadata and image/frame SHA256 checks are in `recording/manifest.json` and `recording/asset-integrity.json`. Dataset media and derived data are covered by `recording/LICENSE.txt`. Three.js 0.180.0 and OrbitControls retain their MIT notice in `vendor/THREE-LICENSE.txt`; procedural meshes are original project code. No vendor library was downloaded or changed for this integration.

## Reproduce or replace deliberately

The retained local prediction export is `artifacts/website-replay/nuscenes-epoch18-export/`. It was generated using the existing `experiments/nuscenes/export_replay.py` with the retained checkpoint/configuration, `--limit 40`, and an explicit diagnostic note. No training was started or modified. The export remains ignored; only the allowlisted public recording is committed.

Package a completed export into a NEW review directory before replacing the public recording:

```powershell
.venv/Scripts/python.exe scripts/package_website_replay.py --export artifacts/website-replay/nuscenes-epoch18-export --config experiments/nuscenes/artifacts/full-baseline/stage-a/config.json --output artifacts/website-replay/new-public-bundle --checkpoint-sha256 28d570f3688ce1dd0b2f52c59ae92568abfb1a0ca634147642e5c88139013b50 --model-label "nuScenes - retained epoch 18"
```

The packager verifies the source export identity, dataset manifest, per-frame timestamp, image paths, calibration-cache hashes and byte-for-byte camera copies. It refuses to reuse an output directory. The public allowlist excludes model weights, optimizer state, labels/targets, raw LiDAR, absolute local paths and source authentication.

Run `node --test tests/website_replay.test.mjs tests/website_animations.test.cjs` and `.venv/Scripts/python.exe -m pytest -q tests/test_website_replay_package.py`. Browser checks should cover all camera buttons, play/pause, scrubbing/rapid scrubs, speed, confidence, projections, object inspection, orbit/top/reset, narrow layouts and offscreen/hidden pause. Test both this standalone page and the parent article iframe. On-demand rendering means a paused, untouched viewer does not keep redrawing.

For later updates, use a new recording/version and bump asset query versions. Do not silently replace this with an unselected training branch, relabel an old mini-checkpoint replay, or present replay refresh as inference throughput.
