"""Verified KITTI-360 caches, conservative visible-target preparation and loader."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, OrderedDict
import json
import hashlib
import importlib.metadata
import math
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial import cKDTree
from fsd.acquisition import sequence, DRIVES, atomic_json, ensure_space
from fsd.geometry import load_calibration, transform_points, project_numpy, box_corners, project_points_torch, canonicalize_car_box

SPLITS={'train':('0000','0002'),'val':('0003',),'test':('0004',)}
RAGGED={'boxes','labels','instance_ids','image_paths'}

def collate_fn(samples):
    if not samples: raise ValueError('Cannot collate empty samples')
    return {key:([s[key] for s in samples] if key in RAGGED or isinstance(samples[0][key],str) else torch.stack([s[key] for s in samples])) for key in samples[0]}

class Kitti360Dataset(Dataset):
    def __init__(self,root,split,config=None):
        self.root=Path(root); self.split=split; self.config=config or {}
        data_config=self.config.get('data',self.config)
        manifest_path=self.root/data_config.get('manifest','manifest.json')
        if not manifest_path.exists(): raise FileNotFoundError(f'{manifest_path} missing: run scripts/prepare_data.py --mode all')
        self.manifest=json.loads(manifest_path.read_text()); drives=SPLITS[split]
        self.samples=[x for x in self.manifest['samples'] if x['drive'] in drives]
        self.samples.sort(key=lambda x:(x['sequence'],x['frame_id']))
        self.calibration=load_calibration(self.root)
        if not self.samples: raise ValueError(f'No prepared {split} samples')
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        row=self.samples[index]
        images=np.stack([np.asarray(Image.open(self.root/p).convert('RGB')).transpose(2,0,1) for p in row['images']])
        sample={'image_paths':row['images'],'images':torch.from_numpy(images.copy()).float()/255.,'camera_valid':torch.ones(4,dtype=torch.bool),'sequence':row['sequence'],'frame_id':torch.tensor(row['frame_id']),'timestamp':torch.tensor(row['timestamp'],dtype=torch.float64)}
        for key in ('intrinsics','camera_to_ego','distortion','xi','camera_model','rays','ray_valid'):
            sample[key]=torch.from_numpy(self.calibration[key].copy())
        with np.load(self.root/row['cache'],allow_pickle=False) as cache:
            for key in ('ego_to_world','road','depth_target','boxes','labels','instance_ids','detection_valid'):
                array=cache[key].copy()
                if key in ('road','depth_target','labels','instance_ids'):array=array.astype(np.int64)
                sample[key]=torch.from_numpy(array)
        return sample

def timestamps(path):
    # Dataset timestamps have nanosecond precision; epoch seconds remain float64.
    result=[]
    for text in Path(path).read_text().splitlines():
        text=text.strip()
        result.append(float(np.datetime64(text.replace(' ','T'),'ns').astype(np.int64))/1e9 if text else np.nan)
    return np.asarray(result)

def xml_matrix(node):
    return np.fromstring(node.findtext('data'),sep=' ').reshape(int(node.findtext('rows')),int(node.findtext('cols')))

def load_boxes(path):
    result=[]
    for element in ET.parse(path).getroot():
        class_id=int(element.findtext('semanticId','-1'))
        if class_id not in (13,19):continue
        transform=xml_matrix(element.find('transform')); local=xml_matrix(element.find('vertices'))
        world=transform_points(local,transform)
        result.append({'class':0 if class_id==13 else 1,'id':class_id*1000+int(element.findtext('instanceId')),'timestamp':int(element.findtext('timestamp')),'start':int(element.findtext('start_frame')),'end':int(element.findtext('end_frame')),'world':world,'heading':transform[:3,0]})
    return result

def ego_box(annotation,world_to_ego):
    vertices=transform_points(annotation['world'],world_to_ego)
    heading=world_to_ego[:3,:3]@annotation['heading']; yaw=np.arctan2(heading[1],heading[0])
    c,s=np.cos(yaw),np.sin(yaw); rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    local=vertices@rotation; lo,hi=local.min(0),local.max(0)
    center=((lo+hi)/2)@rotation.T
    return canonicalize_car_box(np.r_[center,hi-lo,yaw].astype(np.float32),annotation['class'])

def grid_indices(points):
    ij=np.floor((np.asarray(points)[:,:2]+40)/.5).astype(np.int64)
    valid=((ij>=0)&(ij<160)).all(1)
    return ij,valid

class StaticWindows:
    def __init__(self,root,seq):
        self.files=[]; self.cache=OrderedDict()
        for p in sorted((Path(root)/'data_3d_semantics/train'/seq/'static').glob('*.ply')):
            start,end=map(int,p.stem.split('_')); self.files.append((start,end,p))
    def load(self,path):
        if path in self.cache:
            self.cache.move_to_end(path);return self.cache[path]
        raw=PlyData.read(path)['vertex'].data
        sem_key='semanticID' if 'semanticID' in raw.dtype.names else 'semantic'
        inst_key='instanceID' if 'instanceID' in raw.dtype.names else 'instance'
        required={'x','y','z',sem_key,inst_key}
        if not required.issubset(raw.dtype.names):raise RuntimeError(f'Semantic PLY lacks labels: {path}')
        xyz=np.column_stack([raw[k] for k in ('x','y','z')]).astype(np.float64)
        semantics=raw[sem_key].astype(np.int16); instances=raw[inst_key].astype(np.int32)
        confidence=raw['confidence'] if 'confidence' in raw.dtype.names else np.ones(len(raw))
        valid=(confidence>=.5)&np.isin(semantics,[7,8,9,10,22])
        # Retain actual points at 10 cm density; no interpolation of unknown ground.
        xyz=xyz[valid]; semantics=semantics[valid]
        if len(xyz):
            _,indices=np.unique(np.floor(xyz/.1).astype(np.int64),axis=0,return_index=True)
            xyz=xyz[indices]; semantics=semantics[indices]
        item=(xyz,semantics,cKDTree(xyz[:,:2]) if len(xyz) else None)
        self.cache[path]=item
        while len(self.cache)>2:self.cache.popitem(last=False)
        return item
    def points(self,frame,ego_to_world):
        choices=[p for a,b,p in self.files if a<=frame<=b]
        out=[]; labs=[]
        for path in choices:
            xyz,semantic,tree=self.load(path)
            if tree is None:continue
            indices=tree.query_ball_point(ego_to_world[:2,3],60)
            out.append(xyz[indices]); labs.append(semantic[indices])
        return (np.concatenate(out),np.concatenate(labs)) if out else (np.empty((0,3)),np.empty(0,np.int16))

def depth_supervision(points,calibration):
    uv,valid,radial=project_numpy(points,calibration)
    depth=np.full((4,32,88),-1,np.int16); zbuffer=np.full((4,32,88),np.inf,np.float32)
    for v in range(4):
        mask=valid[v]&(radial[v]>=1)&(radial[v]<=80)
        xy=np.floor((uv[v,mask]+.5)/8).astype(int); d=radial[v,mask]
        np.minimum.at(zbuffer[v],(xy[:,1],xy[:,0]),d)
        good=np.isfinite(zbuffer[v])&calibration['ray_valid'][v]
        depth[v,good]=np.clip(np.rint((zbuffer[v,good]-1)*63/79),0,63).astype(np.int16)
    return depth,zbuffer

def currently_visible(points,calibration,zbuffer,tolerance=1.):
    uv,valid,radial=project_numpy(points,calibration)
    visible=np.zeros(len(points),bool)
    for view in range(4):
        idx=np.flatnonzero(valid[view])
        if not len(idx):continue
        xy=np.floor((uv[view,idx]+.5)/8).astype(int)
        reference=zbuffer[view,xy[:,1],xy[:,0]]
        # No LiDAR return -> unknown visibility, not a free-space observation.
        visible[idx]|=np.isfinite(reference)&(radial[view,idx]<=reference+tolerance)&calibration['ray_valid'][view,xy[:,1],xy[:,0]]
    return visible

def road_supervision(world_points,semantics,world_to_ego,calibration,zbuffer):
    road=np.full((160,160),-1,np.int8)
    if not len(world_points):return road
    ego=transform_points(world_points,world_to_ego); ij,in_range=grid_indices(ego)
    keep=np.flatnonzero(in_range); visible=currently_visible(ego[keep],calibration,zbuffer,tolerance=.75)
    keep=keep[visible]
    if not len(keep):return road
    mapped=np.where(semantics[keep]==7,0,np.where(semantics[keep]==8,1,2))
    cells=ij[keep,1]*160+ij[keep,0]
    counts=np.bincount(cells*3+mapped,minlength=160*160*3).reshape(160,160,3)
    total=counts.sum(-1); winners=counts.argmax(-1)
    valid=(total>=2)&(counts.max(-1)>=.75*total)
    road[valid]=winners[valid].astype(np.int8)
    return road

def mark_footprint(mask,box,value):
    x,y,z,l,w,h,yaw=box; radius=np.hypot(l,w)/2+.5
    lo=np.maximum(np.floor((np.array([x,y])-radius+40)/.5).astype(int),0)
    hi=np.minimum(np.ceil((np.array([x,y])+radius+40)/.5).astype(int)+1,160)
    if np.any(hi<=lo):return
    xs=-40+(np.arange(lo[0],hi[0])+.5)*.5; ys=-40+(np.arange(lo[1],hi[1])+.5)*.5
    xx,yy=np.meshgrid(xs-x,ys-y); c,s=np.cos(yaw),np.sin(yaw)
    inside=(abs(c*xx+s*yy)<=l/2+.5)&(abs(-s*xx+c*yy)<=w/2+.5)
    area=mask[lo[1]:hi[1],lo[0]:hi[0]]; area[inside]=value

def object_supervision(annotations,frame,world_to_ego,lidar,calibration,zbuffer,road):
    boxes=[];labels=[];ids=[];excluded=Counter()
    valid=np.broadcast_to(road>=0,(2,160,160)).copy()
    # Known ground cells are conservative negatives. Missing labels elsewhere stay ignored.
    for obj in annotations:
        if not obj['start']<=frame<=obj['end']:continue
        box=ego_box(obj,world_to_ego)
        if not (-40<=box[0]<40 and -40<=box[1]<40):continue
        # All unsupported temporal boxes reserve unknown footprint rather than becoming negatives.
        if obj['timestamp'] not in (-1,frame):
            mark_footprint(valid[obj['class']],box,False);excluded['dynamic_not_at_frame']+=1;continue
        c,s=np.cos(box[6]),np.sin(box[6]); delta=lidar-box[:3]
        local=np.stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1],delta[:,2]),axis=1)
        returns=(abs(local)<=box[3:6]/2+.1).all(1)
        supported=returns.any() and currently_visible(lidar[returns],calibration,zbuffer).any()
        if not supported:
            mark_footprint(valid[obj['class']],box,False);excluded['no_current_lidar_camera_support']+=1;continue
        if np.any(box[3:6]<=0) or not np.isfinite(box).all():
            excluded['invalid_box']+=1;continue
        boxes.append(box);labels.append(obj['class']);ids.append(obj['id'])
    for box,cls in zip(boxes,labels):
        ix,iy=np.floor((box[:2]+40)/.5).astype(int)
        valid[cls,max(0,iy-2):min(160,iy+3),max(0,ix-2):min(160,ix+3)]=True
    return np.asarray(boxes,np.float32).reshape(-1,7),np.asarray(labels,np.int64),np.asarray(ids,np.int64),valid,dict(excluded)

def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def preprocessing_source():
    package=Path(__file__).resolve().parent
    sources={str(p.relative_to(package.parent.parent)).replace('\\','/'):file_sha256(p) for p in (package/'data.py',package/'geometry.py',package/'acquisition.py',package.parent.parent/'scripts/prepare_data.py')}
    packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','pillow','plyfile','pyyaml')}
    record={'cache_schema':2,'car_axis_convention':'length_ge_width_geometric_v1_front_back_unverified','files':sources,'packages':packages,'image_size':[256,704],'bev_range_m':[-40,40],'bev_cell_m':.5,'max_sync_span_s':.06,'depth':'64 radial bins1..80m','label_policy':'supported_timestamp_current_lidar_visible_ground_v1'}
    record['sha256']=stable_fingerprint(record)
    return record


class VerifiedSources:
    """Hash stored inputs against acquisition provenance before any cache reuse."""
    def __init__(self,root):
        self.root=Path(root); path=self.root/'source_manifest.json'
        self.manifest_sha256=file_sha256(path)
        manifest=json.loads(path.read_text())
        self.expected={r['path']:(key,r) for key,r in manifest['members'].items()}
        self.verified={}
    def verify(self,path):
        path=Path(path);relative=path.relative_to(self.root).as_posix()
        if relative in self.verified:return self.verified[relative]
        if relative not in self.expected:raise RuntimeError(f'No acquisition provenance for {relative}')
        key,source=self.expected[relative]
        size=path.stat().st_size;digest=file_sha256(path)
        if size!=source['stored_bytes'] or digest!=source['stored_sha256']:
            raise RuntimeError(f'Source integrity mismatch: {relative}; reacquire before preparation')
        record={'stored_bytes':size,'stored_sha256':digest,'source_member':key,'source_sha256':source['source_sha256'],'zip_crc32':source['zip_crc32']}
        self.verified[relative]=record
        return record
    def fingerprint(self,paths):
        return {Path(p).relative_to(self.root).as_posix():self.verify(p)['stored_sha256'] for p in paths}


def verified_cached_row(path,fingerprint):
    """Only reuse complete caches with identical inputs/code and a verified payload."""
    path=Path(path);sidecar=path.with_suffix('.json')
    if not path.exists() or not sidecar.exists():return None
    try:
        row=json.loads(sidecar.read_text())
        if row.get('preprocessing_fingerprint')!=fingerprint:return None
        if path.stat().st_size!=row['cache_bytes'] or file_sha256(path)!=row['cache_sha256']:return None
        with np.load(path,allow_pickle=False) as arrays:
            expected={'ego_to_world':(4,4),'depth_target':(4,32,88),'road':(160,160),'detection_valid':(2,160,160)}
            if any(arrays[key].shape!=shape for key,shape in expected.items()):return None
            boxes=arrays['boxes'];labels=arrays['labels'];ids=arrays['instance_ids']
            if boxes.shape!=(len(labels),7) or ids.shape!=(len(labels),):return None
            if not np.isfinite(boxes).all() or not np.isfinite(arrays['ego_to_world']).all():return None
            if not np.isin(labels,[0,1]).all() or not np.isin(arrays['road'],[-1,0,1,2]).all():return None
            if not ((arrays['depth_target']>=-1)&(arrays['depth_target']<64)).all():return None
            if int((arrays['road']>=0).sum())!=row['road_valid_cells']:return None
            if [int((d>=0).sum()) for d in arrays['depth_target']]!=row['depth_valid_cells']:return None
            if int((labels==0).sum())!=row['cars'] or int((labels==1).sum())!=row['pedestrians']:return None
        return row
    except (OSError,ValueError,KeyError,EOFError):return None


def prepare_cache(root,limit=0,drives=DRIVES,publish=True):
    started=time.perf_counter();root=Path(root);ensure_space(root)
    calibration=load_calibration(root);source=VerifiedSources(root);processor=preprocessing_source()
    # Fingerprinted namespace preserves all caches referenced by frozen smoke manifests.
    cache_root=root/'cache'/('v2_'+processor['sha256'][:16])
    calibration_paths=sorted((root/'calibration').glob('*'))
    calibration_fingerprint=source.fingerprint([p for p in calibration_paths if p.is_file()])
    selection_path=root/('selection_smoke.json' if limit else 'selection_full.json')
    selections=json.loads(selection_path.read_text())
    progress_path=root/('manifest.preparing.json' if publish else 'manifest.warmup.'+'-'.join(drives)+'.json')
    manifest={'version':2,'subset':'smoke' if limit else 'full_2hz','sample_stride':5,'image_size':[256,704],'radial_depth':True,'splits':SPLITS,'samples':[],'exclusions':[],'preprocessing_source':processor,'source_manifest':{'path':'source_manifest.json','sha256':source.manifest_sha256},'selection':{'path':selection_path.name,'sha256':file_sha256(selection_path)},'calibration_sources':calibration_fingerprint,'drive_sources':{},'verified_sources':source.verified,'publication':'standard' if publish else 'warm_cache_only','preparation':{'generated':0,'reused_verified':0,'complete':False},'label_policy':'Official XML boxes at supported timestamps and observed LiDAR surfaces. Current-visible static semantic ground only. Unlabeled cells ignored. Test drive not used for model selection.'}
    for drive in drives:
        seq=sequence(drive);pose_path=root/'data_poses'/seq/'poses.txt'
        box_path=root/'data_3d_bboxes/train'/f'{seq}.xml'
        timestamp_paths=[root/'data_2d_raw'/seq/f'image_{v:02d}'/'timestamps.txt' for v in range(4)]
        velo_timestamp_path=root/'data_3d_raw'/seq/'velodyne_points/timestamps.txt'
        shared=source.fingerprint([pose_path,box_path,*timestamp_paths,velo_timestamp_path])
        manifest['drive_sources'][seq]=shared
        rows=np.loadtxt(pose_path)
        poses={int(r[0]):np.r_[r[1:].reshape(3,4),[[0,0,0,1]]]@np.diag([1.,-1.,-1.,1.]) for r in rows}
        anns=load_boxes(box_path);static=StaticWindows(root,seq)
        ts=[timestamps(p) for p in timestamp_paths];velo_ts=timestamps(velo_timestamp_path)
        for i,frame in enumerate(selections[seq]):
            try:
                images=[root/'images'/seq/f'image_{v:02d}'/f'{frame:010d}.png' for v in range(4)]
                velo=root/'data_3d_raw'/seq/'velodyne_points/data'/f'{frame:010d}.bin'
                if not all(p.exists() for p in images) or not velo.exists():raise ValueError('missing_image_or_lidar')
                times=np.array([v[frame] for v in ts]+[velo_ts[frame]])
                if not np.isfinite(times).all() or np.ptp(times)>.06:raise ValueError(f'sync_error_{np.ptp(times):.3f}s')
                semantic_paths=[p for a,b,p in static.files if a<=frame<=b]
                inputs=source.fingerprint([*images,velo,*semantic_paths])
                fingerprint=stable_fingerprint({'processor':processor['sha256'],'calibration':calibration_fingerprint,'shared':shared,'inputs':inputs,'frame':frame})
                path=cache_root/seq/f'{frame:010d}.npz'
                row=verified_cached_row(path,fingerprint)
                if row is not None:
                    manifest['samples'].append(row);manifest['preparation']['reused_verified']+=1
                else:
                    pose=poses[frame];inv=np.linalg.inv(pose)
                    lidar=np.fromfile(velo,np.float32).reshape(-1,4)[:,:3]
                    lidar=transform_points(lidar,calibration['velo_to_ego'])
                    depth,zbuffer=depth_supervision(lidar,calibration)
                    world,semantics=static.points(frame,pose)
                    road=road_supervision(world,semantics,inv,calibration,zbuffer)
                    ground_ego=transform_points(world,inv)
                    ground_radius=np.hypot(ground_ego[:,0],ground_ego[:,1])
                    height_points=ground_ego[(semantics==7)&(ground_radius>2)&(ground_radius<10),2]
                    ground_summary={'median':float(np.median(height_points)),'points':int(len(height_points)),'p05':float(np.percentile(height_points,5)),'p95':float(np.percentile(height_points,95))} if len(height_points)>=20 else None
                    boxes,labels,ids,det_valid,excluded=object_supervision(anns,frame,inv,lidar,calibration,zbuffer,road)
                    if (depth>=0).sum()==0:raise ValueError('no_valid_depth')
                    if (road>=0).sum()==0:raise ValueError('no_valid_road')
                    ensure_space(root,1024*1024);path.parent.mkdir(parents=True,exist_ok=True)
                    temporary=path.with_suffix('.npz.tmp')
                    with temporary.open('wb') as output:
                        np.savez_compressed(output,ego_to_world=pose.astype(np.float32),depth_target=depth,road=road,boxes=boxes,labels=labels,instance_ids=ids,detection_valid=det_valid)
                    temporary.replace(path)
                    row={'drive':drive,'sequence':seq,'frame_id':frame,'timestamp':float(times[0]),'sync_span_ms':float(np.ptp(times)*1000),'images':[p.relative_to(root).as_posix() for p in images],'cache':path.relative_to(root).as_posix(),'cache_sha256':file_sha256(path),'cache_bytes':path.stat().st_size,'preprocessing_fingerprint':fingerprint,'source_paths':list(inputs),'ground_height':ground_summary,'cars':int((labels==0).sum()),'pedestrians':int((labels==1).sum()),'road_valid_cells':int((road>=0).sum()),'depth_valid_cells':[int((d>=0).sum()) for d in depth],'excluded_objects':excluded}
                    atomic_json(path.with_suffix('.json'),row)
                    manifest['samples'].append(row);manifest['preparation']['generated']+=1
            except (ValueError,KeyError) as exc:
                manifest['exclusions'].append({'sequence':seq,'frame_id':frame,'reason':str(exc)})
            if i%20==0:
                manifest['preparation']['elapsed_s']=time.perf_counter()-started
                print(f'PREPARE {seq} {i+1}/{len(selections[seq])} usable={len(manifest["samples"])} excluded={len(manifest["exclusions"])} reused={manifest["preparation"]["reused_verified"]} elapsed_s={manifest["preparation"]["elapsed_s"]:.1f}',flush=True)
                atomic_json(progress_path,manifest)
        atomic_json(progress_path,manifest)
    manifest['counts']={split:{'frames':sum(s['drive'] in drives for s in manifest['samples']),'cars':sum(s['cars'] for s in manifest['samples'] if s['drive'] in drives),'pedestrians':sum(s['pedestrians'] for s in manifest['samples'] if s['drive'] in drives)} for split,drives in SPLITS.items()}
    manifest['prepared_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    manifest['preparation']['elapsed_s']=time.perf_counter()-started
    manifest['preparation']['complete']=True
    if publish:
        frozen_path=root/('manifest.smoke.json' if limit else 'manifest.full.json')
        # Full manifest is published only after all requested drives finish.
        if not limit or not frozen_path.exists():atomic_json(frozen_path,manifest)
        atomic_json(root/'manifest.json',manifest)
    else:
        # Warmup never touches active, final, smoke, or full-preparation manifests.
        atomic_json(progress_path,manifest)
    print(json.dumps({'counts':manifest['counts'],'preparation':manifest['preparation']},indent=2),flush=True)
    return manifest


def audit_cache(root):
    root=Path(root); destination=Path('artifacts/data-audit');destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text()); calibration=load_calibration(root)
    rows=[r for r in manifest['samples'] if r['drive']!='0004']
    if not rows:raise RuntimeError('No non-test samples available for projection audit')
    indices=np.unique(np.linspace(0,len(rows)-1,min(50,len(rows)),dtype=int)); max_error=0.; audit_rows=[]
    # True camera ray -> ego -> image round trip, including both fisheyes.
    for view in range(4):
        rays=calibration['rays'][view].reshape(-1,3); mask=calibration['ray_valid'][view].flatten()
        ego=transform_points(rays*10,calibration['camera_to_ego'][view]);uv,valid,_=project_numpy(ego,calibration)
        u,v=np.meshgrid((np.arange(88)+.5)*8-.5,(np.arange(32)+.5)*8-.5)
        error=np.linalg.norm(uv[view]-np.stack((u,v),-1).reshape(-1,2),axis=-1)
        max_error=max(max_error,float(error[mask].max()))
    for number,index in enumerate(indices):
        row=rows[index]; audit_rows.append({'sequence':row['sequence'],'frame_id':row['frame_id'],'boxes':f'{number:02d}_{row["sequence"]}_{row["frame_id"]:010d}.jpg','road':f'{number:02d}_road.png','depth':f'{number:02d}_depth.jpg'}); tile=Image.new('RGB',(1408,512)); draw=ImageDraw.Draw(tile)
        with np.load(root/row['cache']) as cache: boxes=cache['boxes'];road=cache['road'];depth=cache['depth_target']
        corners=box_corners(torch.from_numpy(boxes)).numpy() if len(boxes) else np.zeros((0,8,3))
        uv,valid,_=project_numpy(corners.reshape(-1,3),calibration)
        for view in range(4):
            image=Image.open(root/row['images'][view]).convert('RGB'); d=ImageDraw.Draw(image)
            vuv=uv[view].reshape(-1,8,2); vok=valid[view].reshape(-1,8)
            for points,ok in zip(vuv,vok):
                for a,b in ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)):
                    if ok[a] and ok[b]:d.line([tuple(points[a]),tuple(points[b])],fill=(255,90,40),width=2)
            d.text((6,6),f'{row["sequence"]} frame {row["frame_id"]} camera {view} GT audit',fill=(255,255,255))
            tile.paste(image,((view%2)*704,(view//2)*256))
        tile.save(destination/f'{number:02d}_{row["sequence"]}_{row["frame_id"]:010d}.jpg',quality=90)
        # Independent road/depth diagnostic panes, never normal replay predictions.
        colors=np.array([[95,100,105],[80,160,220],[150,125,80]],np.uint8)
        ground=np.full((160,160,3),20,np.uint8);known=road>=0;ground[known]=colors[road[known]]
        road_image=Image.fromarray(ground).resize((640,640),Image.Resampling.NEAREST)
        rd=ImageDraw.Draw(road_image)
        for box in boxes:
            c,sn=np.cos(box[6]),np.sin(box[6]);pts=[]
            for dx,dy in ((-1,-1),(1,-1),(1,1),(-1,1)):
                x=box[0]+c*dx*box[3]/2-sn*dy*box[4]/2;y=box[1]+sn*dx*box[3]/2+c*dy*box[4]/2
                pts.append(((x+40)*8,(y+40)*8))
            rd.line(pts+[pts[0]],fill=(255,100,50),width=2)
        rd.ellipse((316,316,324,324),fill=(255,255,0))
        rd.text((5,5),'GT road(gray), sidewalk(blue), other(brown), unknown(black); x right,y down',fill='white')
        road_image.save(destination/f'{number:02d}_road.png')
        depth_tile=Image.new('RGB',(1408,512))
        for view in range(4):
            image=Image.open(root/row['images'][view]).convert('RGB');d=ImageDraw.Draw(image)
            for iy,ix in zip(*np.where(depth[view]>=0)):
                if (ix+iy)%2:continue
                value=int(depth[view,iy,ix]);color=(255-int(value*3),min(255,60+value*3),int(value*4))
                x,y=(ix+.5)*8-.5,(iy+.5)*8-.5;d.ellipse((x-1,y-1,x+1,y+1),fill=color)
            d.text((6,6),'GT nearest radial depth (red near, blue far)',fill='white')
            depth_tile.paste(image,((view%2)*704,(view//2)*256))
        depth_tile.save(destination/f'{number:02d}_depth.jpg',quality=90)
    report={'frames':audit_rows,'status':'numerical_pass_visual_review_required' if max_error<.05 else 'fail','projection_roundtrip_max_pixels':max_error,'audit_frames':len(indices),'test_policy':'No test imagery in qualitative audit; mechanical preparation only','counts':manifest['counts'],'subset':manifest['subset'],'sync_span_ms_max':max(s['sync_span_ms'] for s in manifest['samples']),'limitations':['Sparse visible LiDAR-supervised road coverage, not dense ground completion','Dynamic object boxes used only at exact annotated timestamps','Static objects restricted to their annotation windows and present surface evidence','KITTI-360 boxes are annotation primitives; pedestrians can have coarse dimensions','Custom benchmark metrics apply only to supported validity regions, not official KITTI scores']}
    atomic_json(destination/'report.json',report); print(json.dumps(report,indent=2));return report


def estimate_ground_plane(root, manifest_name='manifest.smoke.json'):
    """Estimate a single display-plane height from TRAIN ground evidence only."""
    import hashlib
    root=Path(root); path=root/manifest_name; manifest=json.loads(path.read_text())
    cal=load_calibration(root); windows={}; medians=[]; counts=[]; records=[]
    for row in manifest['samples']:
        if row['drive'] not in SPLITS['train']:continue
        seq=row['sequence']
        if row.get('ground_height'):
            g=row['ground_height'];medians.append(g['median']);counts.append(g['points'])
            records.append({'sequence':seq,'frame_id':row['frame_id'],'median_ground_z_m':g['median'],'points':g['points'],'p05_m':g['p05'],'p95_m':g['p95']})
            continue
        if seq not in windows:windows[seq]=StaticWindows(root,seq)
        with np.load(root/row['cache']) as cache:pose=cache['ego_to_world'].astype(np.float64)
        world,sem=windows[seq].points(row['frame_id'],pose)
        ego=transform_points(world,np.linalg.inv(pose));keep=(sem==7)&(np.hypot(ego[:,0],ego[:,1])<10)&(np.hypot(ego[:,0],ego[:,1])>2)
        if keep.sum()<20:continue
        vals=ego[keep,2];median=float(np.median(vals));medians.append(median);counts.append(int(len(vals)))
        records.append({'sequence':seq,'frame_id':row['frame_id'],'median_ground_z_m':median,'points':int(len(vals)),'p05_m':float(np.percentile(vals,5)),'p95_m':float(np.percentile(vals,95))})
    if not medians:raise RuntimeError('No train road evidence for ground-plane estimation')
    result={'ground_z_m':float(np.median(medians)),'frame_median_p05_m':float(np.percentile(medians,5)),'frame_median_p95_m':float(np.percentile(medians,95)),'frames':len(medians),'points':sum(counts),'source_split':'train_only','source_manifest':manifest_name,'source_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'estimator':'Median of per-frame median ego road heights in2-10m annulus; actual official semantic points, no flattening of training targets','limitations':'Fixed render-plane approximation. ground_fov_mask gives potential camera coverage, not occlusion certainty. Egos chassis footprint masked geometrically.','per_frame':records}
    atomic_json(root/'ground_plane.json',result);return result
