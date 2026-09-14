# Separate browser rendering and replay timings

`viewer/replay-timing.js` adds bounded CPU-side QA telemetry, with no normal-UI widgets and no changes to recorded model inference values. The installed viewer exposes a non-writable, frozen `window.sceneReplayTiming` interface:

```javascript
window.sceneReplayTiming.snapshot() // a detached, read-only observation of telemetry/state
window.sceneReplayTiming.reset()    // clear timing buffers/counters only; preserve replay/scene
```

Each of eight metrics retains its latest 600 numeric samples in a fixed Float64 ring. Numeric storage is 38,400 bytes total, plus fixed counters/objects. Snapshot calculations allocate at most one bounded sorted copy per metric. Reports distinguish total observed samples from retained samples, use linear-interpolated percentiles, and reject nonfinite/negative durations. A long session cannot retain one measurement object per browser frame indefinitely.

## What each number measures

| Metric | Meaning and boundary |
|---|---|
| `scene_update_cpu_ms` | Synchronous road/object/DOM application, including assigning image URLs. Camera decoding, completed layout/paint, and GPU completion are excluded. |
| `render_submit_cpu_ms` | CPU wall duration of `renderer.render()`, potentially including driver blocking. This is not GPU-complete rendering latency. |
| `raf_interval_ms` | Interval between browser animation callbacks. This measures scheduling; it neither proves compositor/display completion nor fresh scene predictions. |
| `request_headers_ms` | Accepted scene request start to response headers. Includes server/network scheduling. |
| `response_body_json_ms` | Response headers to `response.json()` completion. Remaining transfer and JSON decode are combined; this is not isolated CPU parse time. |
| `request_to_decoded_ms` | Accepted request start through response JSON completion. |
| `request_to_scene_apply_ms` | Scene load start through synchronous scene/DOM update. Excludes asynchronous camera-image decoding and GPU/display completion. |
| `recorded_model_call_ms` | Historical `inference_ms` copied from the export. It can include cold calls and is never a newly measured inference. |

Several boundaries overlap. Do not add their medians to claim end-to-end latency. Rendering continues when replay is paused, so animation callback frequency must never be called fresh perception FPS. No WebGL completion fence, compositor timestamp, physical-display measurement, camera capture, or model call is introduced.

Only accepted requests contribute request latency samples. Superseded and failed requests have separate counters. Reset clears the previous animation timestamp so the first subsequent callback does not produce a false long interval. Snapshot context includes replay provenance/note, selected frame, playing/visibility state, viewport/pixel ratio, and the last reported renderer draw/geometry/texture counts.

## Provisional real-replay capture

`artifacts/render-timing/capture-render-timing.cjs` launches a fresh headless Chromium context with hardware GPU disabled and explicitly asserts a SwiftShader software renderer. It reads the existing authorized 16-frame training-fit replay on 8768, leaves user profiles untouched, warms for 1.5 seconds, measures a 2.5 second paused window, then captures three complete 15-update playback windows. Returning to frame 0 is completed before each reset, avoiding contamination from a queued scrub. The helper deliberately verifies the 16-frame training-fit disclaimer; a final selected-model capture must deliberately update that scope/protocol rather than silently relabel this evidence.

```powershell
node artifacts/render-timing/timing-controls.mjs
node artifacts/render-timing/capture-render-timing.cjs --playwright-core C:/Users/William/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules/playwright-core --browser C:/Users/William/AppData/Local/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-win64/chrome-headless-shell.exe --url http://127.0.0.1:8768 --output artifacts/render-timing/fresh-capture.json
```

Runtime paths above identify the existing local installation. Use a fresh report filename. No new package, checkpoint, data, training configuration or Python script was changed for this task.

The three repetitions in `training-fit-provisional.json` measured:

| Measurement | Median across each repetition | p95 across each repetition |
|---|---|---|
| Scene update CPU | 1.00 / 1.00 /0.90ms | 1.33 /1.13 /1.13ms |
| Rendering call CPU submission | 0.70 /0.70 /0.70ms | 1.10 /1.20 /1.00ms |
| Browser animation callback interval | 16.70 /16.70 /16.70ms | 16.80 /16.80 /16.80ms |
| Request through synchronous scene apply | 52.10 /38.20 /39.70ms | 68.11 /69.37 /68.59ms |

Each repetition includes 15 actual accepted scene loads and 469–472 rendering calls. Some animation intervals reached33–50ms; medians alone do not describe every frame. Renderer resource counts were stable at two geometries and one texture at each window end. These counts are renderer bookkeeping, not total browser memory. Historical exported model-call median23.883 ms is retained separately and is not measured by this browser run.

All timings remain provisional: they use a short training-fit replay, a warm local filesystem cache, instrumented software rendering, and no live model inference. They do not establish hardware-rendering latency, held-out prediction quality, or 30 fresh predictions per second.

## Regression evidence

Seven timing controls pass: ring eviction/arithmetic, invalid durations, detached snapshot isolation, superseded-request exclusion, interval arithmetic, and reset semantics. Existing three asynchronous viewer-request tests and 12 server tests pass. The actual-browser regression was rerun after instrumentation:100 rapid scrubs produce one latest request, delayed stale responses remain excluded, play/pause/stepping work, and all 64 camera images decode. There were no browser/page errors. The screenshot was visually inspected: existing controls and the training-fit disclaimer remain visible without new timing clutter.

Evidence lives in `artifacts/render-timing/`: `timing-controls.json`, `training-fit-provisional.json`, its screenshot, and `viewer-regression/report.json` with screenshots. The isolated capture/regression browsers were closed afterward; existing sessions and servers remain available.
