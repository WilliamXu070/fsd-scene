"""CPU-only paired validation evidence and consecutive-car diagnostics."""
from pathlib import Path
import json,itertools,datetime
from collections import Counter
import numpy as np
from fsd.geometry import load_calibration,project_numpy,transform_points
from fsd.metrics import pairwise_iou,detection_match,wrap_angle,Evaluator
from fsd.sealing import sha256,source_hash

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/kitti360'
OUT=ROOT/'artifacts/paired-inspection'
PRED=ROOT/'artifacts/final-measurements/reference/scenes.jsonl'
REPORT=ROOT/'artifacts/final-measurements/reference/validation.json'
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,allow_nan=False,separators=(',',':')),encoding='utf8')

def corners(box):
    local=np.array(list(itertools.product((-1,1),repeat=3)))*np.asarray(box[3:6])/2
    c,s=np.cos(box[6]),np.sin(box[6]);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return local@rot.T+box[:3]

EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
def projection(boxes,cal):
    # Sample the edges: straight world edges can curve in native fisheyes.
    result=[]
    for box in boxes:
        co=corners(np.asarray(box));pts=np.concatenate([np.linspace(co[a],co[b],9) for a,b in EDGES])
        uv,valid,_=project_numpy(pts,cal)
        result.append([[[uv[c,e*9+k].round(3).tolist() if valid[c,e*9+k] else None for k in range(9)] for e in range(12)] for c in range(4)])
    return result

def frame_matches(raw,cache,threshold):
    p=np.asarray(raw['boxes'],float).reshape(-1,7);g=cache['boxes'];labs=np.asarray(raw['labels']);scores=np.asarray(raw['scores']);support=Evaluator()._support(p,labs,cache['detection_valid'])
    rows=[];pred_status=['below_threshold']*len(p)
    for cls in (0,1):
        pi=np.flatnonzero((labs==cls)&(scores>=threshold));gi=np.flatnonzero(cache['labels']==cls)
        iou=pairwise_iou(p[pi],g[gi]);status,matched=detection_match(iou,scores[pi],.5,support[pi])
        for k,idx in enumerate(pi):pred_status[idx]={1:'matched',0:'scored_unmatched',-1:'unassessed'}[int(status[k])]
        for k,idx in enumerate(gi):
            candidates=np.flatnonzero(matched==k);chosen=int(pi[candidates[0]]) if len(candidates) else None
            nearest=int(pi[np.argmax(iou[:,k])]) if len(pi) else None
            row=dict(gt_index=int(idx),gt_id=int(cache['instance_ids'][idx]),label=cls,matched_prediction=chosen,best_iou=float(iou[:,k].max()) if len(pi) else 0.,best_prediction=nearest)
            if chosen is not None:
                err=p[chosen]-g[idx];world_error=cache['ego_to_world'][:3,:3]@err[:3]
                row.update(track_id=int(raw['track_ids'][chosen]),score=float(scores[chosen]),center_error_world=world_error.tolist(),dimension_error=err[3:6].tolist(),yaw_error_rad=float(wrap_angle(err[6])))
            rows.append(row)
    return {'targets':rows,'prediction_status':pred_status,'threshold':threshold}

def summarize(frames,key):
    counts=Counter();events=[];changes=[]
    for i in range(1,len(frames)):
        prev,cur=frames[i-1],frames[i]
        dt=cur['timestamp']-prev['timestamp']
        if prev['sequence']!=cur['sequence'] or not 0<dt<=1.01:continue
        a={r['gt_id']:r for r in prev['matches'][key]['targets'] if r['label']==0}
        b={r['gt_id']:r for r in cur['matches'][key]['targets'] if r['label']==0}
        for identity in a.keys()&b.keys():
            x,y=a[identity],b[identity];hitx=x['matched_prediction'] is not None;hity=y['matched_prediction'] is not None
            counts['supported_adjacent_pairs']+=1
            tag=('matched_both' if hitx and hity else 'disappeared' if hitx else 'reappeared' if hity else 'missed_both')
            counts[tag]+=1
            event=dict(index=i,previous_index=i-1,gt_id=identity,kind=tag,dt=dt)
            if hitx and hity:
                switched=x['track_id']!=y['track_id'];counts['track_id_changes']+=int(switched)
                change=dict(center_error_change_m=float(np.linalg.norm(np.array(y['center_error_world'])-x['center_error_world'])),dimension_error_change_m=float(np.abs(np.array(y['dimension_error'])-x['dimension_error']).mean()),yaw_error_change_deg=float(abs(np.degrees(wrap_angle(y['yaw_error_rad']-x['yaw_error_rad'])))))
                changes.append(change);event.update(change,previous_track=x['track_id'],track_id=y['track_id'],track_id_changed=switched)
                if switched or change['yaw_error_change_deg']>45 or change['center_error_change_m']>1:events.append(event)
            elif hitx!=hity:events.append(event)
    coverage=[r for f in frames for r in f['matches'][key]['targets'] if r['label']==0]
    counts['supervised_car_observations']=len(coverage);counts['matched_car_observations']=sum(r['matched_prediction'] is not None for r in coverage)
    stats={k:{'mean':float(np.mean([x[k] for x in changes])),'p95':float(np.percentile([x[k] for x in changes],95))} for k in changes[0]} if changes else {}
    return dict(counts=dict(counts),conditional_error_changes=stats,events=events,definition='Adjacent observations with the same supervised dataset car identity and gap<=1.01s. Geometry conditioned on both IoU0.5 matches. ID changes are a local diagnostic, not official tracking metrics.')

def main():
    assert not (OUT/'manifest.json').exists(),'Use a new output version rather than overwrite evidence'
    report=json.loads(REPORT.read_text());assert report['split']=='val' and report['source_sha256']==source_hash()
    manifest=json.loads((DATA/'manifest.full.json').read_text());assert sha256(DATA/'manifest.full.json')==report['dataset_manifest_sha256']
    samples={(x['sequence'],x['frame_id']):x for x in manifest['samples'] if x['drive']=='0003'}
    rows=[json.loads(l) for l in PRED.open() if l.strip()];assert len(rows)==len(samples)==197
    cal=load_calibration(DATA);frames=[];index=[]
    for i,raw in enumerate(rows):
        row=samples[(raw['sequence'],raw['frame_id'])];assert abs(raw['timestamp']-row['timestamp'])<1e-6;assert raw['image_paths']==row['images'];assert sha256(DATA/row['cache'])==row['cache_sha256']
        with np.load(DATA/row['cache'],allow_pickle=False) as z:cache={k:z[k] for k in z.files}
        np.testing.assert_allclose(raw['ego_to_world'],cache['ego_to_world'],atol=1e-6,rtol=0)
        matches={str(t):frame_matches(raw,cache,t) for t in (.01,.55)}
        f=dict(index=i,sequence=raw['sequence'],frame_id=raw['frame_id'],timestamp=raw['timestamp'],matches=matches)
        frames.append(f);index.append(dict(index=i,frame_id=raw['frame_id'],timestamp=raw['timestamp'],cars=int(sum(cache['labels']==0)),pedestrians=int(sum(cache['labels']==1))))
        lidar_path=next(p for p in row['source_paths'] if p.endswith('.bin'))
        lidar=np.fromfile(DATA/lidar_path,np.float32).reshape(-1,4)[:,:3];ego=transform_points(lidar,cal['velo_to_ego']);good=np.isfinite(ego).all(1)&(np.linalg.norm(ego,axis=1)<80);ego=ego[good];step=max(1,int(np.ceil(len(ego)/3000)));sample=ego[::step]
        uv,valid,radial=project_numpy(sample,cal)
        payload=dict(**f,predictions=raw,targets={k:cache[k].tolist() for k in ('boxes','labels','instance_ids','ego_to_world','road','depth_target','detection_valid')},calibration={k:v.tolist() for k,v in cal.items() if k not in ('rays','ray_valid')},provenance=dict(sample=row,cache_path=row['cache'],cache_sha256=row['cache_sha256'],image_sha256=[sha256(DATA/p) for p in row['images']],lidar_path=lidar_path,lidar_sha256=sha256(DATA/lidar_path),lidar_total_points=len(lidar),lidar_display_points=len(sample),lidar_decimation_stride=step,projection='Native pinhole/Mei; box edges sampled at9worldpoints peredge; projected curves are not silhouettes. Image coordinates are704x256processedinputs.',input_image_scope='Exact compiled704x256RGB tensors before normalization; original source resolution is not shown.'),projection=dict(predictions=projection(raw['boxes'],cal),targets=projection(cache['boxes'],cal),lidar=[np.c_[uv[c,valid[c]],radial[c,valid[c]]].round(3).tolist() for c in range(4)]))
        write(OUT/'frames'/f'{i:04d}.json',payload)
    summary={k:summarize(frames,k) for k in ('0.01','0.55')}
    checks=dict(frame_count=len(rows),cache_hashes_verified=len(rows),image_hashes_recorded=4*len(rows),timestamps_and_pose_agreement=True,final_test_read=False,new_inference=False,training=False)
    write(OUT/'consistency.json',dict(scope='Saved validation predictions only; existing display thresholds, no model selection or learning',thresholds=summary,checks=checks))
    # Bind summary to the new paired payloads and original files.
    write(OUT/'manifest.json',dict(status='complete',split='val',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_sha256=report['checkpoint_sha256'],source_sha256=source_hash(),dataset_manifest_sha256=sha256(DATA/'manifest.full.json'),predictions_sha256=sha256(PRED),validation_report_sha256=sha256(REPORT),builder_sha256=sha256(Path(__file__)),frames=index,frame_sha256={str(i):sha256(OUT/'frames'/f'{i:04d}.json') for i in range(len(rows))},consistency_sha256=sha256(OUT/'consistency.json'),checks=checks))
    write(OUT/'replay-metadata.json',dict(provenance='trained',checkpoint=report['checkpoint'],checkpoint_sha256=report['checkpoint_sha256'],note='Validation drive0003. Existing frozen R2 predictions. Ground-truth diagnostics are separate; no new inference or training.',ground_z=-.930941852176808,ground_source='Training-only ground calibration',bev_min=-40,bev_step=.5))
    print(json.dumps({k:v['counts'] for k,v in summary.items()},indent=2))
if __name__=='__main__':main()
