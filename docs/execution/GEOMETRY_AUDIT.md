# Ego geometry audit: frame5355

The apparent diagonal road is supported by an actual right turn. No fixed45-degree
IMU/body-axis correction is warranted.

The rectified front-camera optical axis in the existing ego frame is
[0.994109,0.037795,-0.101578], with horizontal yaw+2.177degrees. The Velodyne forward
axis differs by only-0.333degrees. Consecutive motion initially follows ego forward:
frame5355 to5360 translates[2.4281,-0.2442,-0.0023]metres.

The vehicle then turns right. Relative yaw from frame5355 is-12.984degrees at5360,
-32.560at5370,-44.835at5400 and-45.330at5430. Position at5430 in the original5355
frame is[36.577,-29.994,-0.187]metres, matching the diagonal ground geometry.

As an independent composition check, pose * camera-to-ego reproduces the official
cam0_to_world record with0.000394m translation difference (rounded text-file
precision). GT object centers project into the right half of the front image at
u=548,472,628and514pixels, consistent with the photographed street and objects.

Evidence and reproducible CPU-only script:
- artifacts/geometry-diagnosis/frame5355-diagnosis.json
- artifacts/geometry-diagnosis/frame5355-turn-geometry.png
- artifacts/geometry-diagnosis/diagnose_frame5355.py

This audit used training drive0000 and official calibration/poses only. No model
inference, sealed-test inspection or geometry implementation changes were performed.
