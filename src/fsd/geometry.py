"""KITTI-360 metric geometry. Ego axes: forward, left, up; depth is radial."""
from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import torch
import yaml

def transform_points(points, transform):
    return np.asarray(points) @ np.asarray(transform)[:3,:3].T + np.asarray(transform)[:3,3]

def matrix44(values):
    out=np.eye(4); out[:3,:]=np.asarray(values).reshape(3,4); return out

def read_keyed(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            key,value=line.split(':',1)
            try:result[key]=np.fromstring(value,sep=' ')
            except ValueError:continue # Human-readable calibration timestamp.
    return result

def load_calibration(root,image_size=(256,704)):
    root=Path(root)/'calibration'; cams=read_keyed(root/'calib_cam_to_pose.txt'); perspective=read_keyed(root/'perspective.txt')
    flip=np.diag([1.,-1.,-1.,1.]); height,width=image_size
    output={k:[] for k in ('camera_to_ego','intrinsics','distortion','xi','camera_model')}
    for cam in range(4):
        camera_to_ego=flip@matrix44(cams[f'image_{cam:02d}'])
        if cam<2:
            rect=np.eye(4); rect[:3,:3]=perspective[f'R_rect_{cam:02d}'].reshape(3,3)
            camera_to_ego=camera_to_ego@np.linalg.inv(rect)
            k=perspective[f'P_rect_{cam:02d}'].reshape(3,4)[:,:3].copy()
            original_w,original_h=perspective[f'S_rect_{cam:02d}']
            dist=np.zeros(4); xi=0.; model=0
        else:
            raw=(root/f'image_{cam:02d}.yaml').read_text().splitlines()[1:]
            config=yaml.safe_load('\n'.join(raw)); p=config['projection_parameters']; d=config['distortion_parameters']
            k=np.array([[p['gamma1'],0,p['u0']],[0,p['gamma2'],p['v0']],[0,0,1.]])
            original_w,original_h=config['image_width'],config['image_height']
            dist=np.array([d['k1'],d['k2'],d['p1'],d['p2']]); xi=config['mirror_parameters']['xi']; model=1
        # Pixel-center preserving resize (PIL/OpenCV half-pixel convention).
        sx,sy=width/original_w,height/original_h
        k[0,:]*=sx; k[1,:]*=sy
        k[0,2]+=(sx-1)/2; k[1,2]+=(sy-1)/2
        for name,value in zip(output,(camera_to_ego,k,dist,xi,model)): output[name].append(value)
    output={k:np.asarray(v,dtype=np.int64 if k=='camera_model' else np.float32) for k,v in output.items()}
    output['rays'],output['ray_valid']=camera_rays(output,height//8,width//8,height,width)
    velo=matrix44(np.loadtxt(root/'calib_cam_to_velo.txt').flatten())
    output['velo_to_ego']=(flip@matrix44(cams['image_00'])@np.linalg.inv(velo)).astype(np.float32)
    return output

def distort_xy(x,y,d):
    k1,k2,p1,p2=d; r2=x*x+y*y; scale=1+k1*r2+k2*r2*r2
    return x*scale+2*p1*x*y+p2*(r2+2*x*x), y*scale+p1*(r2+2*y*y)+2*p2*x*y

def camera_rays(calibration,h=32,w=88,image_h=256,image_w=704):
    u,v=np.meshgrid((np.arange(w)+.5)*image_w/w-.5,(np.arange(h)+.5)*image_h/h-.5)
    all_rays=[]; all_valid=[]
    for k,d,xi,model in zip(calibration['intrinsics'],calibration['distortion'],calibration['xi'],calibration['camera_model']):
        xd=(u-k[0,2])/k[0,0]; yd=(v-k[1,2])/k[1,1]
        x=xd.copy(); y=yd.copy()
        # Newton inversion is stable for the strong radial distortion of these lenses.
        for _ in range(25):
            fx,fy=distort_xy(x,y,d); ex=fx-xd; ey=fy-yd
            k1,k2,p1,p2=d; r2=x*x+y*y; s=1+k1*r2+k2*r2*r2
            dsx=2*x*(k1+2*k2*r2); dsy=2*y*(k1+2*k2*r2)
            a=s+x*dsx+2*p1*y+6*p2*x; b=x*dsy+2*p1*x+2*p2*y
            c=y*dsx+2*p1*x+2*p2*y; e=s+y*dsy+6*p1*y+2*p2*x
            det=a*e-b*c; det=np.where(abs(det)>1e-9,det,1e-9)
            x-=np.clip((e*ex-b*ey)/det,-.1,.1); y-=np.clip((-c*ex+a*ey)/det,-.1,.1)
        r2=x*x+y*y; discriminant=1+(1-xi*xi)*r2
        scale=(xi+np.sqrt(np.maximum(discriminant,0)))/(1+r2)
        rays=np.stack((scale*x,scale*y,scale-xi),axis=-1)
        rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-9)
        xcheck,ycheck=distort_xy(x,y,d)
        valid=(discriminant>0)&(np.hypot(xcheck-xd,ycheck-yd)<1e-5)
        all_rays.append(rays); all_valid.append(valid)
    return np.array(all_rays,np.float32),np.array(all_valid,bool)

def project_numpy(points,calibration,image_size=(256,704)):
    height,width=image_size; grids=[]; validities=[]; ranges=[]
    for extr,k,d,xi,model in zip(*(calibration[n] for n in ['camera_to_ego','intrinsics','distortion','xi','camera_model'])):
        p=transform_points(points,np.linalg.inv(extr)); radial=np.linalg.norm(p,axis=-1)
        denom=p[:,2]+xi*radial
        safe=np.where(abs(denom)>1e-8,denom,1e-8)
        x,y=distort_xy(p[:,0]/safe,p[:,1]/safe,d)
        u=k[0,0]*x+k[0,2]; v=k[1,1]*y+k[1,2]
        valid=(denom>1e-5)&(radial>1e-5)&np.isfinite(u)&np.isfinite(v)&(u>=-.5)&(u<width-.5)&(v>=-.5)&(v<height-.5)
        if model and xi>1: valid &= p[:,2]/np.maximum(radial,1e-8)>-1/xi
        grids.append(np.stack((u,v),axis=-1)); validities.append(valid); ranges.append(radial)
    return np.array(grids),np.array(validities),np.array(ranges)

def project_points_torch(points,batch):
    """Project ego points to native pinhole/Mei views, align_corners=False grids."""
    points=points.float(); dtype=points.dtype; extr=batch['camera_to_ego'].to(dtype)
    # Rigid inverse avoids torch.linalg.inv mixed precision restrictions.
    centered=points[:,None,:,:]-extr[:,:,:3,3][:,:,None,:]
    p=torch.einsum('bvnj,bvjk->bvnk',centered,extr[:,:,:3,:3])
    radial=torch.linalg.vector_norm(p,dim=-1).clamp_min(1e-8)
    xi=batch['xi'].to(dtype)[:,:,None]
    denom=p[...,2]+xi*radial; safe=torch.where(denom>1e-5,denom,torch.ones_like(denom))
    x,y=(p[...,0]/safe).clamp(-10,10),(p[...,1]/safe).clamp(-10,10)
    d=batch['distortion'].to(dtype); k1,k2,p1,p2=[d[:,:,i,None] for i in range(4)]
    r2=x*x+y*y; scale=1+k1*r2+k2*r2*r2
    xd=x*scale+2*p1*x*y+p2*(r2+2*x*x); yd=y*scale+p1*(r2+2*y*y)+2*p2*x*y
    k=batch['intrinsics'].to(dtype)
    u=k[:,:,0,0,None]*xd+k[:,:,0,2,None]; v=k[:,:,1,1,None]*yd+k[:,:,1,2,None]
    h,w=batch['images'].shape[-2:]
    grid=torch.stack((2*(u+.5)/w-1,2*(v+.5)/h-1),dim=-1)
    valid=(denom>1e-5)&torch.isfinite(grid).all(-1)&(grid.abs()<=1).all(-1)
    mei=(batch['camera_model']==1)[:,:,None] & (xi>1)
    valid &= (~mei)|(p[...,2]/radial>-1/xi.clamp_min(1e-6))
    if 'camera_valid' in batch: valid &= batch['camera_valid'][:,:,None]
    return torch.nan_to_num(grid,nan=2.,posinf=2.,neginf=-2.).clamp(-100,100),valid

def canonicalize_car_box(box,class_id):
    """Use a car's longer horizontal box axis as length; preserve occupied geometry.

    This axis convention does not certify which end is the physical vehicle front.
    Other classes and already canonical cars remain bit-identical.
    """
    result=np.asarray(box).copy()
    if class_id==0 and result[4]>result[3]:
        result[3],result[4]=result[4],result[3]
        result[6]=(result[6]+np.pi/2+np.pi)%(2*np.pi)-np.pi
    return result


def box_corners(boxes):
    signs=boxes.new_tensor(list(itertools.product((-1,1),repeat=3)))
    local=boxes[...,None,3:6]*signs/2
    c=boxes[...,6].cos()[...,None]; s=boxes[...,6].sin()[...,None]
    x=c*local[...,0]-s*local[...,1]; y=s*local[...,0]+c*local[...,1]
    return torch.stack((x,y,local[...,2]),dim=-1)+boxes[...,None,:3]


def ground_fov_mask(batch, ground_z=-1.6, bev_size=160, bev_min=-40., bev_step=.5, max_range=80.):
    """Potential ground visibility from calibration only; does NOT resolve occlusion.

    Never reads labels, road targets, depth targets, or detection_valid. Covers native
    ray domain and range in any live camera; marks no ground under the ego chassis.
    """
    device=batch['camera_to_ego'].device
    b=batch['camera_to_ego'].shape[0]
    axis=torch.arange(bev_size,device=device,dtype=torch.float32)*bev_step+bev_min+bev_step/2
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    points=torch.stack((xx,yy,torch.full_like(xx,float(ground_z))),-1).reshape(1,-1,3).expand(b,-1,-1)
    grids,valid=project_points_torch(points,batch)
    extr=batch['camera_to_ego'].float()
    radial=torch.linalg.vector_norm(points[:,None]-extr[:,:,:3,3][:,:,None],dim=-1)
    valid &= radial<=max_range
    if 'ray_valid' in batch:
        import torch.nn.functional as F
        domain=batch['ray_valid'].float();views=domain.shape[1]
        support=F.grid_sample(domain.reshape(b*views,1,*domain.shape[-2:]),grids.reshape(b*views,1,-1,2),mode='nearest',align_corners=False,padding_mode='zeros').reshape(b,views,-1)>.5
        valid &= support
    visible=valid.any(1).reshape(b,bev_size,bev_size)
    ego_footprint=(xx.abs()<2.5)&(yy.abs()<1.)
    return visible & ~ego_footprint[None]
