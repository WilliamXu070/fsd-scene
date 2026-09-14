# Paired data inspection and car consistency

This supplements the completed experiment at the user's request. It does not reopen learning or the sealed test. Use the 197 validation observations and the existing selected-model prediction export.

1. Bind each prediction to its exact sequence, timestamp, four processed camera inputs, compiled NPZ supervision and source-file provenance. Verify hashes and pose agreement before displaying comparisons.
2. Present the existing prediction-only 3D replay beside calibrated camera overlays and a top-down comparison with compiled ground/box labels. Green means supported annotation, orange means prediction. Unknown ground stays unknown. A label overlay is a diagnostic layer, never an inference input.
3. Expose the actual numbers: geometric centers, dimensions, heading, confidence, predicted track IDs, dataset instance IDs, road/depth arrays, calibration, masks and exact local source paths. Provide downloadable compiled targets and paired JSON for inspection.
4. Measure consecutive-frame car availability, disappear/reappear events, track-ID changes and changes in geometry error. Compare the same supported dataset identity at adjacent eligible timestamps; account for ego and actual object motion. Report both the original 0.01 metric cutoff and the existing 0.55 display cutoff. Do not interpolate missing labels or count true motion as jitter.
5. Inspect ordinary and numerically flagged failure pairs. Check synchronization during scrubbing/playback, raw-array parity, calibrated overlays and threshold behavior. Preserve screenshots and adverse findings.

The existing temporal metric only covers successful matches, so it can omit exactly the dropouts the user notices. This extension adds availability and identity diagnostics; it does not claim official tracking scores or diagnose the upstream cause from flicker alone.

Design: a synchronized laboratory comparison, with the current 3D view at left and source/compiled evidence at right. Use the existing Segoe UI typography, white #ffffff and slate #243640 text, pale gray #edf2f6 surfaces, green #087f5b annotations, orange #c45a13 predictions, and blue #376db3 depth points. Numeric tables use tabular figures. No decorative animations; motion follows replay controls.

Next intervention depends on evidence: preserve tracks through short misses if availability is the problem; stabilize dimensions if size error fluctuates; distinguish directed 180-degree heading flips from geometric box-axis changes. Any new training, tracker modification or output smoothing is a separate proposed change. First quantify the failure while preserving current raw outputs.
