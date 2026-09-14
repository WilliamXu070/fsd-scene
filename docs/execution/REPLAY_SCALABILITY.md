# Replay scalability audit — synthetic fixture only

The 1,622-scene replay path is bounded by a byte-offset index and a two-scene decoded cache. It does not load the entire road export into memory. No model, dataset annotation, sealed-test image, or GPU computation was used in this audit.

## Measured probe

The explicitly labeled synthetic fixture contains 1,622 repeated original scenes, each with three 160×160 road/class-confidence/visibility arrays, 100 geometric boxes, and four image references. JSONL size is **1,098,310,018 bytes (1.10 GB)**. Each synthetic 704×256 camera PNG is approximately 470 KB. The images contain original random pixels and a visible synthetic label; they are not recordings or model evidence.

Forty reproducible random scrubs include the first, last, and middle scene. Each cycle requests one scene JSON, parses it in a Python HTTP client, and transfers all four PNGs concurrently. The server runs as a separate loopback process. Its process-tree RSS includes the Windows virtual-environment launcher and actual child Python server.

| Observation | Before | After |
|---|---:|---:|
| Startup through first indexed status, warm filesystem cache | 1.340 s | 1.166 s |
| Median full scrub: HTTP + client JSON + four PNG transfers | 89.06 ms | 39.74 ms |
| p95 full scrub | 118.23 ms | 61.45 ms |
| Median scene JSON request alone | 51.01 ms | 29.71 ms |
| Median four-camera transfer phase | 30.39 ms | 3.68 ms |
| Full JSON decodes per scene + four image requests | 9 | 1 |
| Maximum observed server process-tree RSS | 43.66 MiB | 44.35 MiB |
| Scene response size | ~755 KB | ~677 KB |

Evidence: `artifacts/viewer-stress/baseline-process-tree.json` and `artifacts/viewer-stress/optimized-process-tree.json`. Each report preserves server-source SHA-256, frame count, payload sizes, sampled indices, and timing details. The earlier `baseline.json` measured only the Windows launcher RSS and is superseded for memory conclusions by `baseline-process-tree.json`.

## Necessary changes

- `scripts/serve_demo.py` caches only the two most recently accessed decoded scenes. Camera-availability checks and image requests reuse them. Cache invalidates when the prediction file is removed, truncated, or replaced; append-only exports retain previously complete frames. Incomplete trailing records stay hidden.
- Compact JSON formatting removes whitespace without changing prediction values or fields.
- `viewer/replay-requests.js` aborts superseded scene fetches and discards stale responses before JSON decoding where possible. Late decode completion cannot replace the newest scene. Slider input is debounced by 70 ms; play/pause or frame-step controls cancel pending scrub work.
- Cancelled client connections are handled without producing a server traceback. Existing path confinement, image-only dataset serving, prediction-field whitelist, and visible fixture provenance remain intact.

The UI retains one scene record and its existing bounded object pool. Road textures and geometry already dispose their preceding allocations; no unbounded road-frame cache was added. A maximum of 100 exported objects remains the model/export contract. The server does not parse all 1,622 records at startup; it scans line boundaries and retains integer offsets.

## Verification and reproduction

```powershell
.venv/Scripts/python.exe -m pytest tests/test_viewer.py -q
node --test tests/test_viewer_requests.mjs
node --check viewer/app.js
.venv/Scripts/python.exe scripts/stress_replay.py --output artifacts/viewer-stress/fresh-probe.json
```

Twelve server controls and three asynchronous request controls pass. They cover two-entry cache eviction, reuse across camera requests, append/replacement/truncation/removal behavior, and all prior isolation/traversal tests. The request tests verify that 100 rapid slider events schedule one final request, that superseded network responses avoid JSON decode, and that stale decode completion or failures cannot overwrite current results.

The probe creates or reuses `artifacts/viewer-stress/synthetic-1622/` and starts a temporary loopback server with `--fixture`; it shuts down its own server process tree afterward. It checks the 50 GiB free-space reserve before generating the approximately 1.1 GB fixture. The synthetic fixture is expendable project-owned test data, separate from KITTI-360 inputs and predictions.

## Limits of the evidence

These timings are replay I/O measurements, not perception FPS or model accuracy. They exclude browser PNG decoding, WebGL rendering, and model inference. The filesystem cache is warm and the data are repeated synthetic geometry. The manifest-image fallback was covered by correctness tests but this stress probe uses explicit image paths; loading a real large manifest may add memory. RSS was sampled between requests, so transient within-request peaks can be higher. Only one before/after probe was performed: observed gains are diagnostic, not a statistical performance benchmark.

The 70 ms debounce is an intentional responsiveness tradeoff while dragging, separate from the measured HTTP latency. Actual browser layout, long-duration playback, GPU texture allocation, rendering cadence, and interaction during real exports still need final operational QA. Existing actual training-fit viewer QA remains documented separately in `VIEWER_QA.md`; it is not replaced by this synthetic stress test.
