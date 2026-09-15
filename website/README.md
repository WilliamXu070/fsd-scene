# fsd-scene project article

Static HTML/CSS/JavaScript. The article documents the custom perception attempt, its measured results, and the later Sparse4Dv3 comparison. Eight self-contained SVG figures provide playback, scrubbing, representation choices, architecture steps, loss selection, synthetic diagnostics, query refinement and a measured timing breakdown.

The opening prose and the ten-part IPFormer outline follow William's reference report. The IPFormer section expands the outline with explanations checked against the papers and `src/fsd/model.py`. Later article sections are preserved. The embedded opening image is the image supplied in the report; it illustrates the intended scene and is not a project prediction.

From the project root:

```powershell
.venv/Scripts/python.exe -m http.server 8790 --bind 127.0.0.1 --directory website
```

Open http://127.0.0.1:8790/. Edit article text in `index.html`, visual styling in `style.css`, and chapter navigation in `app.js`.

Edit the interactive figures in `animations.js` and their isolated surface styles in `animations.css`. Each `data-animation` hook in the HTML selects a figure definition containing geometry, controls, explanatory readouts and provenance. No runtime dependency, remote asset request or model inference is required.

Every figure starts paused. Play/Pause, Restart and a keyboard-accessible slider work independently; architecture and query panels also offer Previous/Next. Leaving the viewport or hiding the tab pauses playback. The shared animation loop is capped near 30 updates per second and stops when idle. With reduced motion enabled, Play becomes Step and advances still frames. Figures recompose for mobile; captions and article text remain available without JavaScript.

## Animation provenance and checks

- Camera, object, feature-map and query visuals are original deterministic schematics, not recorded checkpoint output. The diagnostic viewer deliberately uses authored errors and scores. Its dashed reference boxes are synthetic, not dataset ground truth.
- Architecture shapes follow `src/fsd/model.py` and `configs/baseline.yaml`. Repeated sparse refinement is conceptual; the earlier baseline uses one spatial-and-temporal block.
- The ray uses an ideal pinhole projection. Storage counts include feature values only. Scalar training uses squared error, not the complete project loss.
- Latency means come from the local, excluded `artifacts/final-measurements/reference/latency.json`, `stage_profile.timings`: eager execution, 30 updates, three history frames. FPN is already included in the image encoder. The grey remainder is the instrumented total minus listed module means. Separate cached-processing medians already include transfer, pose/history handling, road decoding, score filtering/NMS and tracking; they exclude native capture/resize, pose estimation, queues, road mesh construction and display. See `scripts/benchmark.py` for the scope. Pipeline boxes are not drawn to a duration scale.
- No dataset images, model weights, inference dumps, report snapshots or private recordings are bundled for these animations.

Run `node --test tests/website_animations.test.cjs` from the project root for dependency-free calculation and asset checks. Browser QA should cover every playback/slider, mode and loss choice, reference and confidence controls, narrow and desktop layouts, keyboard stepping, offscreen pause and reduced motion. Inspect console errors and SVG label clipping. Keep QA outputs under ignored `artifacts/website-animations/`. Bump animation asset query versions in `index.html` for later published changes to avoid stale caches.

Use `<figure class="media-surface">` with an image and caption for embedded media. Store site images under `assets/` with filenames containing no spaces, and use relative URLs so they work under the GitHub Pages project path. Use `<figure class="code-surface">` with a caption and `<pre><code>` for a code panel; escape HTML characters in code and label pseudocode as such. The code panels scroll horizontally within the article and can be focused with the keyboard. Neither surface changes the existing article grid, sidebar, or typography.

Push website edits to `main` to deploy through `.github/workflows/deploy-pages.yml`. Only `website/` is published. Keep source-report snapshots, imports, and verification outputs in the ignored `artifacts/website-content/` directory.

Desktop uses a sticky chapter index; small screens use an expandable contents menu. Keyboard focus, skip navigation, reduced-motion and print styling are included. Training remains stopped.
