"""Synthetic replay HTTP stress probe; never reads datasets or model weights."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, random, re, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import psutil
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]

def summary(values):
    ordered=sorted(values)
    return dict(count=len(values),mean_ms=statistics.mean(values),median_ms=statistics.median(values),
                p95_ms=ordered[min(len(ordered)-1,int(.95*len(ordered)))],max_ms=max(values))

def fixture(folder, frames):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'synthetic-repeated-predictions.jsonl'
    metadata=folder/'metadata.json'
    if path.exists():
        return path,metadata
    if psutil.disk_usage(str(folder)).free<52*2**30:
        raise RuntimeError('Stress fixture would violate the 50 GiB free-space reserve')
    rng=random.Random(71622)
    image_paths=[]
    for i in range(4):
        p=folder/f'synthetic-camera-{i}.png'
        im=Image.frombytes('RGB',(704,256),rng.randbytes(704*256*3))
        draw=ImageDraw.Draw(im);draw.rectangle((0,0,703,34),fill='black')
        draw.text((10,10),f'SYNTHETIC STRESS FIXTURE / CAMERA {i} / NO MODEL EVIDENCE',fill='white')
        im.save(p);image_paths.append(p.name)
    road=[[0 if abs(x-80)<25 else 1 if abs(x-80)<30 else 2 for x in range(160)] for y in range(160)]
    confidence=[[rng.random()*.45+.5 for x in range(160)] for y in range(160)]
    visible=[[10<x<150 and 5<y<155 for x in range(160)] for y in range(160)]
    boxes=[[rng.uniform(-35,35),rng.uniform(-35,35),0.,4.,1.8,1.5,rng.uniform(-3.14,3.14)] for i in range(100)]
    record=dict(sequence='SYNTHETIC_REPEATED_STRESS_FIXTURE',boxes=boxes,scores=[.8]*100,labels=[0]*100,
                track_ids=list(range(100)),road=road,road_confidence=confidence,road_visible=visible,
                road_ground_z_m=-1.,image_paths=image_paths,inference_ms=None,
                ego_to_world=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # Hold one JSON line, not all scenes, while generating the deliberately large fixture.
    suffix=json.dumps(record,separators=(',',':'))[1:].encode()
    with path.open('xb') as stream:
        for i in range(frames):
            stream.write(f'{{"frame_id":{i},"timestamp":{i*.5},'.encode()+suffix+b'\n')
    metadata.write_text(json.dumps(dict(provenance='fixture',note='SYNTHETIC SCALABILITY STRESS: repeated original geometry, not model predictions or test results.',ground_z=-1.,ground_source='Synthetic fixture')),encoding='utf8')
    return path,metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fixture-root',type=Path,default=ROOT/'artifacts/viewer-stress/synthetic-1622')
    p.add_argument('--frames',type=int,default=1622)
    p.add_argument('--scrubs',type=int,default=40)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh result path')
    path,metadata=fixture(args.fixture_root.resolve(),args.frames)
    start=time.perf_counter()
    proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/serve_demo.py'),'--predictions',str(path),
                           '--data-root',str(path.parent),'--metadata',str(metadata),'--port','0','--fixture'],
                          cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        line=proc.stdout.readline();match=re.search(r'http://127.0.0.1:\d+',line)
        if not match:raise RuntimeError(f'Server startup failed: {line}')
        base=match.group();server=psutil.Process(proc.pid)
        def tree_rss():
            return sum(p.memory_info().rss for p in [server,*server.children(recursive=True)] if p.is_running())
        before=tree_rss()
        t=time.perf_counter()
        with urlopen(base+'/api/status',timeout=120) as response:status=json.load(response)
        first_index_ms=(time.perf_counter()-t)*1000
        startup_ms=(time.perf_counter()-start)*1000
        assert status['provenance']=='fixture' and status['frames']==args.frames
        indexed=tree_rss()
        scene_ms=[];parse_ms=[];cameras_ms=[];total_ms=[];rss=[];payloads=[]
        rng=random.Random(9);indices=[0,args.frames-1,args.frames//2]+[rng.randrange(args.frames) for _ in range(args.scrubs-3)]
        def fetch_image(url):
            with urlopen(base+url,timeout=30) as response:return len(response.read())
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index in indices:
                t=time.perf_counter()
                with urlopen(base+f'/api/frame?index={index}',timeout=30) as response:body=response.read()
                t1=time.perf_counter();scene_ms.append((t1-t)*1000);payloads.append(len(body))
                record=json.loads(body);t2=time.perf_counter();parse_ms.append((t2-t1)*1000)
                assert record['index']==index and record['sequence']=='SYNTHETIC_REPEATED_STRESS_FIXTURE'
                assert all(record['images'])
                list(pool.map(fetch_image,record['images']))
                end=time.perf_counter();cameras_ms.append((end-t2)*1000);total_ms.append((end-t)*1000)
                rss.append(tree_rss())
        # Verify bounded data access directly, counting full JSON decodes for scene + 4 camera requests.
        spec=importlib.util.spec_from_file_location('stress_server',ROOT/'scripts/serve_demo.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        store=module.ReplayStore(path,path.parent,metadata,fixture=True)
        calls=0;original=module.json.loads
        def counted(*a,**kw):
            nonlocal calls
            calls+=1
            return original(*a,**kw)
        module.json.loads=counted
        try:
            store.frame(args.frames-1)
            for camera in range(4):store.image_path(args.frames-1,camera)
        finally:module.json.loads=original
        report=dict(provenance='synthetic_repeated_fixture_only',frames=status['frames'],fixture=str(path),
                    fixture_bytes=path.stat().st_size,fixture_image_bytes=[p.stat().st_size for p in sorted(path.parent.glob('synthetic-camera-*.png'))],
                    server_source_sha256=hashlib.sha256((ROOT/'scripts/serve_demo.py').read_bytes()).hexdigest(),
                    startup_to_indexed_status_ms=startup_ms,first_index_request_ms=first_index_ms,
                    server_process_tree_rss_mib=dict(before_index=before/2**20,after_index=indexed/2**20,
                                        max_observed=max(rss)/2**20,after_scrubs=rss[-1]/2**20),
                    scene_http=summary(scene_ms),client_json_decode=summary(parse_ms),four_camera_http=summary(cameras_ms),
                    full_scrub_http_and_json=summary(total_ms),scene_response_bytes=dict(min=min(payloads),max=max(payloads)),
                    json_decodes_per_scene_and_four_images=calls,indices=indices,
                    limitations=['Synthetic original repeated predictions, not model or sealed-test evidence.',
                                 'Filesystem cache is warm after fixture generation; startup is not physical cold-disk timing.',
                                 'HTTP + Python client JSON parsing includes four PNG transfers, but excludes browser image decode, WebGL rendering, and GPU model inference.',
                                 'RSS sums the server launcher and child processes, sampled after requests; brief within-request peaks may exceed maximum observed.'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print(json.dumps(report,indent=2))
    finally:
        # Windows venv launchers can own a child Python server; stop this process tree.
        children=psutil.Process(proc.pid).children(recursive=True) if proc.poll() is None else []
        for child in children:
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        _,alive=psutil.wait_procs(children,timeout=5)
        for child in alive:child.kill()

if __name__=='__main__':main()
