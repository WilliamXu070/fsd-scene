# Model implementation notes

Implemented in `src/fsd/model.py` and `src/fsd/losses.py`; shared interfaces are in CONTRACTS.md. This implements the approved baseline, not the original IPFormer/SuperOcc architectures.

## Actual network pathway

- Current RGB values in [0,1] are ImageNet-normalized inside the shared ResNet-50. V2 ImageNet weights are requested when pretrained=true; download errors are fatal rather than silently falling back to random initialization. Backbone BN running statistics remain frozen in train mode.
- ResNet C3/C4/C5 (strides8/16/32) feed a torchvision FPN, 128 channels by default. P3 predicts64 radial-depth bins and64 contextual channels. The pyramid incorporates coarse features into P3 and all three scales are sampled during refinement.
- DepthLift uses calibrated unit camera rays and camera-to-ego transforms. Radial bins are1..80m; geometry and pooling accumulate in FP32 even under autocast. Eight bins at a time are multiplied by context and scatter-added into the160x160 BEV. Support mass above1 is normalized; sparse probability amplitude remains. No persistent rich XYZ feature volume is constructed.
- Context passes through a128-channel projection and three residual BEV blocks. Heads predict three ground classes, two class-center heatmaps, and eight raw geometry values: sigmoid-decoded subcell offsets2, vertical center1, logdimensions3, sin/cos heading2.
- Local center peaks select up to200 initial boxes; supervision never replaces these proposals. Geometry can represent arbitrary subcell centers. Classification-independent geometry at one ground cell has an acknowledged collision limit for overlapping centers.
- One late object refinement block reads the box center and eight corners from every valid image and FPN level. Spatial cross-attention combines these samples; temporal cross-attention reads detached historical object features with ego-aligned box-position and elapsed-time embeddings. Up to three prior states are used; no temporal BEV stage exists.
- Box corrections update center, dimensions and yaw; classes receive residual logits. If every spatial sample is invalid, the original box and logits are preserved exactly. Fully masked historical state is equivalent to empty history. Sequence changes, nonpositive times and gaps beyond2s invalidate memory; parent runtime also resets at discontinuities.
- Historical timestamps are subtracted in FP64 before converting elapsed seconds to FP32, preserving subsecond timing for Unix timestamps. No current or future labels enter inference.

## Supervision

- Depth categorical cross-entropy uses only valid measured radial-bin targets, valid rays and valid cameras.
- Road cross-entropy and Dice ignore unknown ground. Dice averages represented target classes.
- Center focal loss uses Gaussian center targets with size-dependent radius. Only supported negatives are supervised; exact annotated centers always remain positive. Positive cells regress subcell location, vertical center, logdimensions and circular heading.
- Refined predictions are Hungarian-matched one-to-one using metric center distance, logdimension distance, circular angle and class confidence. Classification negatives outside detection_valid are ignored. Matched boxes receive center, logdimension and circular-heading losses.
- All task weights default1 and are configurable in the top-level loss mapping. Trainer logs individual task components; total sums each task once. An image with no valid supervision has zero loss.

## Verification performed

10 CPU behavioral tests pass, using small synthetic configurations but the real ResNet/FPN/lifting/refinement network. Cases include known radial rays and metric offsets, nonzero image/backbone/neck/depth/head gradients, history pose rotation, one-to-one assignment, circular yaw, unknown-label masking, target-isolated inference, invalid-image fallback, foreign/future/stale history rejection, Unix-time precision and differentiable local refinement sampling. These tests do not claim data accuracy or replace the real-data gates.

One synthetic CUDA13/FP16 compatibility pass at the full default size (four704x256 images,160BEV,200 proposals) completed forward/backward with finite gradients and1.545GB peak allocated GPU memory. Cold forward/backward took1.37s; one cold history-enabled evaluation took89.5ms. These are compatibility observations, NOT a latency benchmark, performance target achievement or training result.

## Remaining practical limits

The baseline chunked scatter operator may be a latency bottleneck and includes dynamic-index GPU synchronization; profile before choosing optimizations. Center proposals must recover objects before refinement can help them. Temporal features permit learned motion handling but do not guarantee correspondence or smoothness; held-out empty/mismatched-history controls remain necessary. Sparse road supervision is not a visibility predictor and must not be described as dense reconstruction accuracy.
