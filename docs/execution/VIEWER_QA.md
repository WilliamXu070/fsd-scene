# Viewer visual and interaction QA

Status: renderer fixture QA passed after one visually discovered bug was fixed.
This is **not final model QA**. Only synthetic scene geometry and procedural camera
charts were used. No sealed-test imagery or labels were inspected.

The Browser and Computer Use runtime failed to initialize because of the Windows
`apply deny-read ACLs` helper error. After reading both applicable skills and trying
their supported entrypoints, the separate Playwright MCP browser testing interface
successfully opened the local viewer. Evidence comes from the actual rendered page,
DOM-backed interaction, screenshots, and image decode state.

## Verified behavior

| Behavior | Observed result |
|---|---|
| Initial perspective | Road/sidewalk mesh, three detected cars, one person and distinct ego asset render coherently |
| Play/pause | Frame0 advanced to3; pausing held frame3 through an additional0.7s |
| Next/previous | Frame0 ->1 ->0 after awaiting the asynchronous frame load |
| Timeline | Home selects0; End selects11 |
| End of replay | 2x playback stops at frame11 and returns to Play |
| Confidence | Threshold1.0 removes all4 detections; threshold0.0 restores4; ego remains a reference asset |
| View/grid | Perspective/top view and metric-grid toggle change the scene as expected |
| Orbit/zoom | Drag140x40px and wheel-350 change angle/scale while keeping frame0 |
| Resize | Desktop1440x1000 and mobile390x844 layouts remain usable; mobile hides playback speed |
| Four camera panels | All4 decoded704x256 charts; next-frame URLs change together from index0 to1 |
| Provenance | Synthetic fixture banner and explanatory note remain visible; missing model timing is an em dash |

## Issue found and verified fixed

The first four-camera screenshot showed **No image available** over images that had
successfully loaded. DOM inspection found `span.camera-placeholder[hidden]`; an
author display rule overrode the browser's default hidden styling.

The metrics agent added `[hidden]{display:none!important}`. Reloaded browser checks
confirm all four placeholder elements have computed `display:none`, all four images
remain loaded, and the corrected screenshot has no placeholder overlay.

- Before: `artifacts/viewer-qa/06-four-camera-panels.png`
- After: `artifacts/viewer-qa/07-four-camera-panels-fixed.png`
- Detailed machine-readable checks: `artifacts/viewer-qa/report.json`
- Final accessibility snapshot: `artifacts/viewer-qa/final-dom.md`

The only remaining first-navigation console error was favicon.ico404. This does not
affect replay. The mobile playback-speed control is hidden; primary desktop controls
are fully accessible. These checks establish renderer plumbing, not perception
accuracy, real-time model latency, temporal stability, or camera/geometry alignment.

## Reproduce the camera-panel fixture

```
.venv/Scripts/python.exe scripts/serve_demo.py --predictions artifacts/viewer-qa/camera-fixture.jsonl --data-root artifacts/viewer-qa --metadata artifacts/viewer-qa/camera-metadata.json --fixture --port 8767
```

All chart images and fixture copies stay under `artifacts/viewer-qa`. They are
original procedural test data, clearly labeled, and excluded from model outputs.

## Actual training-fit replay (separate from fixture QA)

The viewer on port8765 was subsequently inspected with real predictions from
`artifacts/runs/overfit16-b/learned.pt` on the same16 training frames5355-5430.
Renderer controls and four-camera synchronization pass. This is **not held-out
validation**, and the displayed perception is **not yet a faithful reconstruction**.

At confidence0.3 frame5355 displays23 objects and frame5430 displays29, including
extra and overlapping meshes. Increasing to0.7 leaves6 and14 respectively, while
0.9 leaves3 and4; this is a precision/recall tradeoff, not validated threshold tuning.
Broad ground patches cover unsupported regions. Conservative known-cell target
metrics alone therefore do not establish good full-scene predictions. Box bottoms
relative to the fixed measured ground vary by up to about0.37m in these two frames;
there is no evidence for a global units or coordinate rotation error.

The diagonal road in frame5355 is consistent with the actual45degree right turn
measured from source poses; see GEOMETRY_AUDIT.md. All64 camera images across the
16-frame replay decode and follow the selected sourceframe. Play advances toindex3
in1.25s, Pause holds for0.65s, and End selectsindex15 at7.83s elapsed. The provenance
note fits both1440x1000 desktop and390x844 mobile without clipping.

Initially the UI omitted its detailed metadata note and used ambiguous timing and
absolute epoch labels. The metrics agent fixed these. Screenshot10 verifies the
full training-fit disclaimer, elapsed timeline, Recorded model call label, and
explicit road-camera-coverage caveat. The recorded566.0ms firstframe includes
coldstart; the later23.9ms value is not a warmedbenchmark or proof of30FPS.

- `artifacts/viewer-qa/10-training-fit-labeled.png`: real prediction screenshot
- `artifacts/viewer-qa/11-training-fit-mobile.png`: mobile provenance check
- `artifacts/viewer-qa/training-fit-replay.gif`:16 browser screenshots assembled at
  approximately source cadence (522ms/frame), not a liveFPS recording
- `artifacts/viewer-qa/training-fit-report.json`: controls, counts and limitations
- `artifacts/viewer-qa/training-fit-geometry-observations.json`: box-ground offsets

Playwright did not have native video recording configured. Original screenshots
are preserved under training-fit-frames. Rendering cached predictions does not
verify temporalattention effectiveness, tracking stability or inference throughput.


## Updated request/cache regression — actual training-fit replay

The updated server on **http://127.0.0.1:8768** was verified against the unchanged
16-frame `overfit16-b-scenes.jsonl` prediction export. Port8765 and existing browser
sessions were left untouched. No model inference, annotation targets, sealed-test
content, or hardware-GPU workload was used.

The supported browser connection still failed with the Windows ACL helper error;
the separate browser MCP reported its profile already in use. An isolated headless
Chromium149 session using the existing installed Playwright runtime successfully
rendered the page with hardware GPU disabled. Its actual WebGL renderer identified
itself as ANGLE/SwiftShader. That isolated browser was closed after the check.

- `/replay-requests.js` loaded with HTTP200.
- 100 rapid timeline input events produced one final request, selecting index11.
- A real index4 response delayed350ms for the regression test was cancelled;
  index12 remained displayed after the stale response would have arrived.
- Play advanced from index0 to3 in1.25s; Pause held index3 for0.7s. Next/previous
  returned to the expected frame.
- All64 camera images across16 frames decoded704×256 and matched each frame index.
- There were no page/console errors. The one aborted request was the deliberately
  superseded index4 request.
- Visually reviewed screenshots confirm the prominent unclipped training-fit/no
  held-out-accuracy disclaimer, four clean camera panels, controls and elapsed time.
  Existing extra/overlapping object predictions remain visible.

Evidence: `artifacts/viewer-qa/request-regression/report.json`,
`01-updated-training-fit.png`, `02-latest-scrub-result.png`, and
`03-final-regression.png`. Report includes source/export/metadata SHA-256 values.
Reproduction uses `scripts/qa_replay_browser.cjs` with `--playwright-core` pointing
to the installed runtime, `--browser` to the installed headless-shell executable,
`--url http://127.0.0.1:8768`, and `--output` to a fresh project artifact directory.

This closes the functional browser regression for the request/cache update. It
establishes neither held-out perception quality nor hardware rendering/model FPS.
