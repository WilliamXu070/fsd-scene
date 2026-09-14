# Teaching Figure Brief

## Lesson Instant
- **Brief ID:** `F01-C`
- **Node:** `M01`
- **Layer:** `components`
- **Learner question:** How does our original model transform camera images into scene features and geometry?
- **Intended insight:** Image features are lifted into a shared ground grid before separate road and object predictions.
- **Persistent example:** One synchronized six-camera scene.
- **Primary role:** `expand`
- **Figure count:** `1`
- **Delivery mode:** `static`

## Essential Information Inventory
- **Required input:** Six calibrated camera images.
- **Required output:** Road categories and metric object boxes.
- **Required components:** ResNet50, FPN, depth/context heads, lift, BEV encoder, prediction heads, optional refinement.
- **Required order:** Encode images, lift features, encode ground grid, predict, optionally refine.
- **Required transformations:** Images to image features to metric ground features to geometry.
- **Required representation states:** Image pyramid, depth probabilities, context, BEV, boxes.
- **Required group boundaries:** Image encoding, geometric lifting, scene prediction, optional refinement.
- **Required invariants:** One synchronized scene; shared camera backbone; calibrated metric coordinates.
- **Must-cover IDs:** `I01, I02, I03, I04, I05, I06`
- **Excluded details:** Losses, optimizer settings, numerical performance claims.

## Reference Plan
- **Reference search completed:** `yes`
- **Factual sources:** experiments/nuscenes/code/src/fsd/model.py and configs/full.json; training engine Stage A/B control.
- **Composition references:** TinyTPU editorial diagrams previously inspected in this conversation.
- **Online references analyzed:** https://www.tinytpu.com/
- **Access date:** `2026-09-13`
- **Reuse mode:** `project_derived`
- **Rights status:** Original diagram; no copied external artwork.
- **Reference findings:** Use generous whitespace, direct labels and representations that change along the pathway.
- **Why the reference fits:** The user is documenting an engineering learning project.
- **Why custom work remains:** Our exact model wiring differs from all external diagrams.

## Composition Blueprint
- **Figure grammar:** `single_panorama`
- **Reading direction:** `left_to_right`
- **Persistent anchor:** One scene and its progressively transformed feature tensors.
- **Operation labels:** Inside directly connected modules.
- **Representation objects:** Camera tiles, feature stacks, ground grid and structured output cards.
- **State labels:** Directly below each representation.
- **Grouping devices:** Numbered phases and shaded regions.
- **Geometry meaning:** Arrows encode actual dependencies; dashed outlines identify optional Stage B.
- **Non-color encoding:** All routes carry direct labels; optional route also uses dashes.
- **Main path:** Images through lift and BEV to road and object branches.
- **Maximum label bands:** `4`
- **Interaction:** `none`
- **Duplicate overview rail:** `no`

## Information Coverage Matrix
| ID | Required information | Priority | Visible carrier | Direct label | Source | Depth fit |
|---|---|---|---|---|---|---|
| I01 | Input and backbone | must | Camera tiles and encoder | Six views; ResNet-50 | model.py | pass |
| I02 | Pyramid and branch input | must | Three pyramid levels | P3 feeds depth and context | model.py | pass |
| I03 | Geometric transformation | must | Lift module and calibration input | Camera rays and weighted pooling | model.py | pass |
| I04 | Shared scene features | must | Ground grid and BEV encoder | 128 x 160 x 160 | model.py | pass |
| I05 | Predictions | must | Road and object branch cards | Road classes; centers; box geometry | model.py | pass |
| I06 | Optional late refinement | must | Dashed lower route | Implemented, not reached in full training | engine.py | pass |

## Depth Boundary
- **Include now:** Components, feature sizes, data routes, output meanings and optional execution boundary.
- **Exclude until later:** Loss derivations and internal attention computations.
- **Why exclusions protect focus:** This figure explains architecture rather than teaching each operator.

## Static Delivery
- **Format:** `svg`
- **Canvas:** 2400 by 1320 pixels.
- **Minimum display size:** Open full image for detailed labels.
- **Smallest label:** 18 source pixels for camera identifiers; at least 22 for component descriptions.
- **Alt text:** Six camera images pass through shared ResNet50 and an FPN. P3 yields depth probabilities and context, lifted into BEV using calibrated rays. A BEV CNN feeds road, center and geometry heads. An optional late refinement route uses camera features and previous object features.
- **Caption:** Original custom nuScenes pipeline; solid path is Stage A, dashed extension is Stage B.
- **Source note:** Verified against local model implementation on 2026-09-13; tensor sizes omit batch dimension.

## QA Gates
- **Coverage gate:** `pass`
- **Reference gate:** `pass`
- **Depth gate:** `pass`
- **Static-first gate:** `pass`
- **Evidence gate:** `pass`
- **Render QA:** `pass`
