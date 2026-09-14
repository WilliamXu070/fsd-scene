# Execution contracts — 2026-09-11

Goal: trained/evaluated four-camera KITTI-360 3D scene model and measured replay demo. Baseline architecture in docs/architecture/2026-09-11-preliminary-v0.1.md remains preserved. User approved at most three evidence-driven revisions; never fabricate validation or silently remove required classes/cameras.

Ownership: data agent owns src/fsd/data.py, geometry.py, acquisition.py, scripts/prepare_data.py, tests/test_geometry.py and data artifacts. Model agent owns src/fsd/model.py, losses.py, tests/test_model.py. Parent owns trainer/evaluator/config/runtime/docs and integration. No worker changes another worker's files without coordinating.

Common batch contract (torch tensors unless metadata): images [B,V,3,256,704] float RGB 0..1 (model normalizes ImageNet); camera_valid [B,V] bool; camera_to_ego [B,V,4,4]; intrinsics [B,V,3,3] for RESIZED images; distortion [B,V,4] k1,k2,p1,p2; xi [B,V] (0 for pinhole); camera_model [B,V] 0 pinhole, 1 Mei; rays [B,V,32,88,3] normalized CAMERA-coordinate unit vectors at stride-8 feature centers; ray_valid [B,V,32,88]; ego_to_world [B,4,4]; timestamp [B] seconds; sequence list[str]; frame_id [B].

Geometry choice: use RADIAL range 1..80m for all depth distributions so native fisheye rays beyond 90 degrees remain valid. This is an explicit clarification of the open depth convention. Calibration/targets/lifting must agree. Depth target [B,V,32,88] int64 bin 0..63, -1 unknown. Depth bins centers linearly from 1..80 (torch.linspace); target nearest center.

Targets: road [B,160,160] int64 0 road,1 sidewalk,2 other KNOWN ground,-1 unknown. boxes list[Tensor[N,7]] x,y,z,l,w,h,yaw geometric centers in ego (x forward,y left,z up); labels list[Tensor[N]] 0 car,1 pedestrian; instance_ids list[Tensor[N]]; detection_valid [B,2,160,160] bool to mask unsupported negatives; geometry positives are always valid. grid row corresponds y, col x, range [-40,40); source labels untouched.

geometry.py public functions: project_points_torch(points [B,N,3] ego, batch) -> (grid [B,V,N,2] normalized align_corners=False coordinates, valid [B,V,N]); box_corners(boxes [...,7]) -> [...,8,3]; dataset collate_fn stacks tensors, preserves ragged labels/boxes/IDs and sequence metadata.

data.py: Kitti360Dataset(root, split, config=None) exposes ordered frame samples with getitem individual current frame. config may be plain dict. Splits train0000/0002 val0003 test0004. No implicit ground truth supplied to network. Create dataset manifest and cache, use prepare_data CLI. Temporal memory handled sequentially by parent/model; no automatic future sampling.

model.py: SceneModel(config:dict), forward(batch, history=None, refine=True) -> dict: road_logits [B,3,160,160], depth_logits [B,V,64,32,88], center_logits [B,2,160,160], geometry_map [B,8,160,160] (raw offset2,z,logdims3,sin,cos), initial_boxes [B,K,7], initial_logits [B,K,2], refined_boxes [B,K,7], refined_logits [B,K,2], proposal_features [B,K,128], spatial_valid [B,K], state dict(detached features, boxes, ego_to_world, timestamp, sequence). K default200 <= allowed cells. history list previous state dicts max3; never cross sequences. Network consumes geometry/calibration/images only, not targets. Teacher labels must not be used in proposal generation.

losses.py: compute_losses(outputs,batch,config:dict,refine=True)-> dict scalar tensors with 'total' and task components. Match predicted boxes one-to-one to targets, ignore unsupported negatives and empty labels. Torch CPU tests permitted; no prolonged GPU work by workers while parent installs or trains.

Config schema: data(root,train_drives,val_drives,test_drives,stride,max_sync_ms), model(image_size[256,704],fpn_channels128,context_channels64,bev_channels128,bev_size160,bev_min-40.,bev_step.5,depth_bins64,depth_min1.,depth_max80.,num_proposals200,history_length3,pretrained true), train(seed42,lr1e-4,weight_decay.01,accumulate4,epochs_a24,epochs_b24,patience6,min_epochs8), runtime(device cuda,mixed_precision true). Defaults required so omitted keys work.
