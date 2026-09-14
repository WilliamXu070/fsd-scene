"""Geometric contract tests. These do not substitute for real-data audit gates."""
import numpy as np
import torch
import pytest
from fsd.geometry import box_corners, project_points_torch, project_numpy, camera_rays, transform_points, load_calibration
from fsd.data import depth_supervision, road_supervision, object_supervision, collate_fn
from pathlib import Path


def camera():
    return {'camera_to_ego':np.eye(4,dtype=np.float32)[None], 'intrinsics':np.array([[[100.,0.,100.],[0.,100.,100.],[0.,0.,1.]]],np.float32), 'distortion':np.zeros((1,4),np.float32),'xi':np.zeros(1,np.float32),'camera_model':np.zeros(1,np.int64)}


def batch(calibration):
    result={k:torch.as_tensor(v)[None] for k,v in calibration.items()}
    result['images']=torch.zeros(1,len(calibration['xi']),3,200,200)
    result['camera_valid']=torch.ones(1,len(calibration['xi']),dtype=torch.bool)
    return result


def test_project_known_pinhole():
    points=torch.tensor([[[0.,0.,10.],[1.,0.,10.],[0.,0.,-10.]]])
    grid,valid=project_points_torch(points,batch(camera()))
    assert torch.allclose(grid[0,0,0],torch.tensor([.005,.005]),atol=1e-6)
    assert torch.allclose(grid[0,0,1],torch.tensor([.105,.005]),atol=1e-6)
    assert valid.tolist()==[[[True,True,False]]]


def test_projection_backprop_invalid_points_finite():
    points=torch.tensor([[[0.,0.,10.],[100.,100.,0.],[0.,0.,0.],[-1e4,1e4,-1.]]],requires_grad=True)
    grid,valid=project_points_torch(points,batch(camera()))
    grid.square().sum().backward()
    assert torch.isfinite(grid).all()
    assert torch.isfinite(points.grad).all()


def test_extrinsic_translation():
    cal=camera();cal['camera_to_ego'][0,:3,3]=[2,3,4]
    uv,valid,ranges=project_numpy(np.array([[2.,3.,14.]]),cal,(200,200))
    assert np.allclose(uv[0,0],[100,100])
    assert np.allclose(ranges,10)


def test_oriented_box_extents():
    boxes=torch.tensor([[2.,3.,1.,4.,2.,2.,np.pi/2]])
    corners=box_corners(boxes)
    assert torch.allclose(corners.mean(1),boxes[:,:3])
    assert torch.allclose(corners.max(1).values-corners.min(1).values,torch.tensor([[2.,4.,2.]]),atol=1e-5)


def test_mei_inverse_roundtrip():
    cal=camera();cal['xi'][:]=1.5;cal['camera_model'][:]=1
    cal['distortion'][:]=[.02,.2,.001,-.002]
    rays,valid=camera_rays(cal,25,25,200,200)
    uv,proj_valid,_=project_numpy(rays[0].reshape(-1,3)*10,cal,(200,200))
    x,y=np.meshgrid((np.arange(25)+.5)*8-.5,(np.arange(25)+.5)*8-.5)
    expected=np.stack((x,y),-1).reshape(-1,2)
    assert np.max(abs(uv[0,valid.ravel()]-expected[valid.ravel()]))<.002


def test_real_calibration_axes_and_projection():
    root=Path('data/kitti360')
    if not (root/'calibration').exists():pytest.skip('Official calibration not yet acquired')
    cal=load_calibration(root)
    # Forward optical axis of front camera faces positive ego x.
    assert cal['camera_to_ego'][0,0,2]>.9
    for v in range(4):
        rays=cal['rays'][v].reshape(-1,3);ego=transform_points(rays*10,cal['camera_to_ego'][v])
        uv,valid,_=project_numpy(ego,cal)
        x,y=np.meshgrid((np.arange(88)+.5)*8-.5,(np.arange(32)+.5)*8-.5)
        expected=np.stack((x,y),-1).reshape(-1,2)
        assert np.max(abs(uv[v,cal['ray_valid'][v].ravel()]-expected[cal['ray_valid'][v].ravel()]))<.002


def test_ragged_collate_keeps_object_counts_and_paths():
    samples=[{'boxes':torch.zeros(n,7),'labels':torch.zeros(n,dtype=torch.long),'images':torch.zeros(4,3,8,8),'sequence':'seq','image_paths':['a','b','c','d']} for n in [0,2]]
    result=collate_fn(samples)
    assert result['images'].shape==(2,4,3,8,8)
    assert [len(x) for x in result['boxes']]==[0,2]
    assert result['image_paths'][0]==['a','b','c','d']

def test_ground_fov_is_inference_only_and_front_facing():
    from fsd.geometry import ground_fov_mask
    cal=camera()
    # Camera right=-ego left, down=-ego up, forward=ego forward.
    cal['camera_to_ego'][0,:3,:3]=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    b=batch(cal);b['ray_valid']=torch.ones(1,1,25,25,dtype=torch.bool)
    mask=ground_fov_mask(b,ground_z=-1.6)
    assert mask[0,80,100]
    assert not mask[0,80,60]
    b['road']=torch.full((1,160,160),12345);b['depth_target']=torch.rand(1,1,25,25)
    assert torch.equal(mask,ground_fov_mask(b,ground_z=-1.6))


def test_car_axis_canonicalization_preserves_corners_and_ious():
    from fsd.geometry import canonicalize_car_box
    from fsd.metrics import pairwise_ious
    originals=np.array([[12.,-3.,1.,1.9,4.2,1.5,3.10],[-4.,8.,.3,2.0,5.2,1.8,-2.8],[.2,-.4,2.,1.7,4.5,1.6,.2]],np.float64)
    unchanged=originals.copy()
    canonical=np.stack([canonicalize_car_box(b,0) for b in originals])
    assert np.array_equal(originals,unchanged)
    assert (canonical[:,3]>=canonical[:,4]).all()
    assert ((canonical[:,6]>=-np.pi)&(canonical[:,6]<np.pi)).all()
    old=box_corners(torch.from_numpy(originals)).numpy()
    new=box_corners(torch.from_numpy(canonical)).numpy()
    distances=np.linalg.norm(old[:,:,None]-new[:,None,:],axis=-1)
    assert distances.min(1).max()<1e-10 and distances.min(2).max()<1e-10
    bev,three_d=pairwise_ious(originals,canonical)
    assert np.allclose(np.diag(bev),1,atol=1e-10)
    assert np.allclose(np.diag(three_d),1,atol=1e-10)
    assert np.array_equal(canonical[:,[0,1,2,5]],originals[:,[0,1,2,5]])


def test_car_axis_canonicalization_preserves_normal_car_and_pedestrian():
    from fsd.geometry import canonicalize_car_box
    car=np.array([1.,2.,3.,4.2,1.9,1.5,-.7],np.float32)
    pedestrian=np.array([1.,2.,3.,.7,2.4,1.9,3.1],np.float32)
    assert np.array_equal(canonicalize_car_box(car,0),car)
    assert np.array_equal(canonicalize_car_box(pedestrian,1),pedestrian)


def test_existing_real_overfit_targets_unchanged_by_car_axis_convention():
    from fsd.geometry import canonicalize_car_box
    import json
    root=Path('data/kitti360');path=root/'manifest.smoke.json'
    if not path.exists():pytest.skip('Frozen real overfit targets not acquired')
    manifest=json.loads(path.read_text());cars=people=0
    for row in manifest['samples']:
        if row['drive']!='0000' or not 5355<=row['frame_id']<=5430:continue
        with np.load(root/row['cache']) as cache:
            for box,label in zip(cache['boxes'],cache['labels']):
                assert np.array_equal(canonicalize_car_box(box,int(label)),box)
                cars+=int(label==0);people+=int(label==1)
    assert cars==99 and people==16
