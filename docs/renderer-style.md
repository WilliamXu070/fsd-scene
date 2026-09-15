# Prediction output renderer

This is post-detection presentation. It does not change training, road classes, object predictions, confidence, boxes, poses, calibration, tracking or source replay images.

## Design and assets

The user's Tesla screenshot guides silhouettes, matte neutral traffic, a bright restrained ground plane and clean road edges. This is not a Tesla asset extraction or a claim to reproduce its perception capabilities.

- Traffic: Quaternius's CC0 sedan, https://poly.pizza/m/Cz6yDaUcM9, from https://quaternius.com/packs/cars.html. Selected after comparing its more realistic silhouette with the first Kenney prototype. Source GLB SHA256: `bf00f2f0386a25aa310abc0424d22586e46a59ee6c737e6b375c97c9f01bd462`.
- Ego: Kenney Car Kit 3.1 `sedan-sports.glb`, https://kenney.nl/assets/car-kit. Source GLB SHA256: `2889c428ca1dd9c975c3ff760eb8757883490410555ab60b818f712f583826d6`. Original red paint, expressive eyes, gold stripes and side flashes are composed in the renderer. It is a humorous race-car character, not a downloaded Disney/Pixar model.
- Both source packs are CC0. The selected sources and changes are also attributed inside each viewer. Blender refines edges/normals, normalizes geometry to the prediction coordinate convention, and exports compact triangle buffers. Quaternius's original wheels are retained; Kenney's wheels are replaced with round geometry. The public render meshes have 7,120 and 6,199 triangles respectively; GPU geometry/materials are shared across detections.

## Road geometry and honesty boundary

`cellsFromRecord` preserves the original calibration-FOV and predicted-confidence >= 0.45 policy. The public replay decodes its existing lossless display RLE instead. `traceRegions` follows directed cell boundaries, preserving separate islands and holes, then `smoothRing` applies two local averaging passes capped at half a grid cell relative to every original vertex. The result is triangulated as surfaces, not a blurred pixel texture.

White represents predicted road; neutral gray groups sidewalk/other-ground. The visible-ground underlay prevents tiny seams between smoothed class borders. Edge smoothing can move the displayed class boundary by up to 0.25 m on the present 0.5 m grid; it is not improved localization. No components are deliberately filtered and no temporal lag/interpolation is added. Unknown regions are not assigned new prediction values. Original road grid mode restores the exact pixel colours/mask for auditing. The retained model does **not** predict lane markings, so none are drawn. Camera cuboid projections and numeric object inspection still use the untouched predicted boxes.

## Rebuild

The input asset files and editable output scenes remain under ignored `artifacts/renderer-polish/`. Run from the project root using installed Blender 5.1:

```powershell
& 'C:/Program Files/Blender Foundation/Blender 5.1/blender.exe' --background --factory-startup --python scripts/build_viewer_vehicle_assets.py -- --source artifacts/renderer-polish/quaternius-sedan.glb --output artifacts/renderer-polish/quaternius-build --name traffic-sedan --creator quaternius
& 'C:/Program Files/Blender Foundation/Blender 5.1/blender.exe' --background --factory-startup --python scripts/build_viewer_vehicle_assets.py -- --source 'artifacts/renderer-polish/kenney-car-kit/Models/GLB format/sedan-sports.glb' --output artifacts/renderer-polish/vehicle-build --name ego-racer
```

Review generated `.blend` scenes before copying their selected `.mesh.js` geometry modules to `viewer/scene-style/assets/`. Run `.venv/Scripts/python.exe scripts/sync_scene_style.py` to mirror the eight allowlisted files into `website/replay/scene-style/`. The static modules work with the existing local server's JavaScript allowlist, without changing the replay API or restarting training services. No ML model/checkpoint is copied. All mesh/road changes should be made in the canonical local modules, not just the Pages copy.

## Ego pivot and optional local character

`orbit-frame.js` centres perspective/top/reset on the ego mesh bounding-box centre, replacing the previous target eight metres ahead. Ground changes translate camera and target together, preserving deliberate user pan. Reset restores the ego pivot.

The user supplied `lightning-mcqueen.zip` containing a Blender scene and textures, but no redistribution license. Its derived mesh is explicitly ignored in both viewer directories and excluded from the sync allowlist. Public/default views retain the CC0 ego. Only localhost previews opt in using `?ego=mcqueen`; no download of the unlicensed mesh is attempted by public URLs.

The local character retains the original body, eye and tyre textures. Blender runs with `--disable-autoexec`, evaluates the rest pose without source lights/cameras, aligns the long axis toward the eyes to +X, and normalizes the centred geometry. Uniform physical scaling preserves the original proportions at 4.5 × 2.2804 × 1.5084 m (11,025 triangles). Tyre bottoms sit on the model ground plane. Schema 2 stores UVs, embedded texture data and physical dimensions; schema 1 remains supported for public CC0 assets. No detected vehicle size/yaw is changed.

Rebuild the local asset with `scripts/build_mcqueen_asset.py` using Blender's `--background --factory-startup --disable-autoexec` flags and the supplied scene, then `--python scripts/build_mcqueen_asset.py -- --textures <extracted-textures> --output <ignored-build-directory>`. Copy only the resulting `ego-mcqueen.mesh.js` into the two ignored `scene-style/assets/` paths for local preview. Public redistribution remains pending the source page/license or explicit rights confirmation.

## Verification

`node --test tests/scene_style.test.mjs tests/website_replay.test.mjs tests/website_animations.test.cjs tests/test_viewer_requests.mjs`

The renderer tests exhaust every 3x3 binary mask, check raw occupied area, hole/island topology, bounded smoothing, metric coordinates, all 40 real frames, exact raw-grid colours, buffer validity, disposable per-frame geometry and identical shared assets. Browser QA must also exercise replay, confidence, raw/smooth mode, orbit/top views, object selection, mobile and the embedded article. Model-image integrity checks remain separate and unchanged. Server tests cover the static JavaScript/text assets without broadening the server allowlist, permitting traversal outside `viewer/`, or allowing checkpoint reads.
