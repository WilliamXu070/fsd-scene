"""Acquire official KITTI-360 members and build audited training caches."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from fsd.acquisition import Acquirer, BASE, URLS, DRIVES, sequence, atomic_json, RemoteZip

def selected_drive(name,drives): return any(sequence(d) in name for d in drives)

def acquire(root,limit,stride,drives):
    acq=Acquirer(root); root=Path(root)
    acq.acquire(URLS['calibration'],lambda n:True,lambda n:root/n)
    acq.acquire(URLS['poses'],lambda n:selected_drive(n,drives),lambda n:root/'data_poses'/n)
    acq.acquire(URLS['boxes'],lambda n:selected_drive(n,drives),lambda n:root/n)
    for source,kind in [('perspective','data_2d_raw'),('fisheye','data_2d_raw'),('velodyne','data_3d_raw')]:
        acq.acquire(BASE+kind+f'/data_timestamps_{source}.zip',lambda n:selected_drive(n,drives),lambda n:root/kind/n)
    with RemoteZip(URLS['semantics']) as archive:
        semantic_names=[i.filename for i in archive.infos() if '/static/' in i.filename and i.filename.endswith('.ply')]
    selection={};eligibility={}
    for drive in drives:
        seq=sequence(drive)
        poses=np.loadtxt(root/'data_poses'/seq/'poses.txt')
        candidates=poses[:,0].astype(int); candidates=candidates[candidates%stride==0]
        before=len(candidates)
        windows=[tuple(map(int,Path(n).stem.split('_'))) for n in semantic_names if seq in n]
        candidates=np.array([f for f in candidates if any(a<=f<=b for a,b in windows)],dtype=int)
        eligibility[seq]={'posed_2hz':before,'inside_official_semantic_windows':len(candidates),'excluded_no_ground_annotation_window':before-len(candidates)}
        if limit and len(candidates)>limit:
            candidates=candidates[np.linspace(0,len(candidates)-1,limit,dtype=int)]
            if drive in ('0000','0002'):
                # Add real pedestrian timestamps for training smoke coverage only.
                xml=ET.parse(root/'data_3d_bboxes/train'/f'{seq}.xml').getroot()
                available=set(poses[:,0].astype(int)); ped=sorted({int(x.findtext('timestamp')) for x in xml if x.findtext('semanticId')=='19' and int(x.findtext('timestamp','-1')) in available and int(x.findtext('timestamp','-1'))%stride==0})
                all_frames=set(poses[:,0].astype(int))
                best=[];best_score=-1
                for center in ped:
                    block=list(range(max(0,center-5*stride),center+11*stride,stride))
                    if not all(f in all_frames for f in block):continue
                    score=len(set(block)&set(ped))
                    if score>best_score:best,best_score=block,score
                candidates=np.unique(np.r_[candidates,ped[:min(16,len(ped))],best])
        selection[seq]=candidates.tolist()
    atomic_json(root/('selection_smoke.json' if limit else 'selection_full.json'),selection)
    atomic_json(root/('eligibility_smoke.json' if limit else 'eligibility_full.json'),eligibility)
    def resize(data):
        image=Image.open(io.BytesIO(data)).convert('RGB'); image.load()
        image=image.resize((704,256),Image.Resampling.BILINEAR)
        output=io.BytesIO(); image.save(output,format='PNG'); return output.getvalue()
    for drive in drives:
        seq=sequence(drive); frames=set(selection[seq])
        def wanted(n):
            try:return int(Path(n).stem) in frames
            except ValueError:return False
        for cam in range(4):
            kind='data_rect' if cam<2 else 'data_rgb'
            acq.acquire(BASE+f'data_2d_raw/{seq}_image_{cam:02d}.zip',lambda n:wanted(n) and f'/{kind}/' in n,lambda n:root/'images'/seq/f'image_{cam:02d}'/Path(n).name,resize)
        acq.acquire(BASE+f'data_3d_raw/{seq}_velodyne.zip',lambda n:wanted(n) and n.endswith('.bin'),lambda n:root/'data_3d_raw'/n)
    # Static semantic windows are official accumulated labeled geometry, not pseudo labels.
    acq.acquire(URLS['semantics'],lambda n:selected_drive(n,drives) and '/static/' in n and n.endswith('.ply'),lambda n:root/n)
    return selection

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/kitti360')
    parser.add_argument('--mode',choices=['acquire','prepare','all','audit','ground','warm-cache'],default='all')
    parser.add_argument('--limit',type=int,default=0,help='0=all eligible 2 Hz frames; >0 creates explicitly marked smoke subset')
    parser.add_argument('--stride',type=int,default=5)
    parser.add_argument('--manifest',default='manifest.smoke.json',help='Source manifest for --mode ground')
    parser.add_argument('--drives',nargs='+',default=list(DRIVES))
    args=parser.parse_args()
    if args.mode in ('all','acquire'): acquire(args.root,args.limit,args.stride,args.drives)
    if args.mode in ('all','prepare'):
        from fsd.data import prepare_cache
        prepare_cache(args.root,limit=args.limit,drives=args.drives)
        from fsd.data import estimate_ground_plane
        estimate_ground_plane(args.root,'manifest.smoke.json' if args.limit else 'manifest.full.json')
    if args.mode=='warm-cache':
        from fsd.data import prepare_cache
        prepare_cache(args.root,limit=args.limit,drives=args.drives,publish=False)
    if args.mode=='ground':
        from fsd.data import estimate_ground_plane
        estimate_ground_plane(args.root,args.manifest)
    if args.mode in ('all','audit'):
        from fsd.data import audit_cache
        audit_cache(args.root)

if __name__=='__main__':main()
