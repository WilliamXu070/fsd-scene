"""Replay data isolation, traversal, streaming and provenance controls."""
import importlib.util
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import pytest

spec=importlib.util.spec_from_file_location('serve_demo',Path(__file__).resolve().parents[1]/'scripts/serve_demo.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ReplayStore=module.ReplayStore

def write_frame(path,**extra):
    frame=dict(sequence='drive',frame_id=10,timestamp=1.,boxes=[],scores=[],labels=[],road=[[0]])
    frame.update(extra);path.write_text(json.dumps(frame)+'\n');return frame


def test_missing_prediction_file_reports_waiting(tmp_path):
    store=ReplayStore(tmp_path/'future.jsonl',tmp_path)
    assert store.status()['state']=='waiting'
    assert store.status()['provenance']=='unverified'


def test_frames_indexed_incrementally_and_partial_writer_line_hidden(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path)
    store=ReplayStore(path,tmp_path);assert store.status()['frames']==1
    with path.open('a') as f:f.write('{"frame_id": 11')
    assert store.status()['frames']==1
    with path.open('a') as f:f.write('}\n')
    assert store.status()['frames']==2
    assert store.frame(1)['frame_id']==11


def test_prediction_interface_strips_targets_and_filesystem_paths(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path,ground_truth={'boxes':['secret']},target_boxes=['secret'],image_paths=['private.png'])
    output=ReplayStore(path,tmp_path).frame(0)
    assert 'ground_truth' not in output and 'target_boxes' not in output and 'image_paths' not in output
    assert output['images']==[None]*4


def test_images_resolved_from_manifest_without_opening_target_cache(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path)
    (tmp_path/'photo.png').write_bytes(b'not needed to decode for this path test')
    (tmp_path/'manifest.json').write_text(json.dumps({'samples':[{'sequence':'drive','frame_id':10,'images':['photo.png']*4,'cache':'DO_NOT_OPEN.npz'}]}))
    store=ReplayStore(path,tmp_path)
    assert store.image_path(0,0)==tmp_path/'photo.png'
    assert store.frame(0)['images'][0]=='/api/image?index=0&camera=0'


def test_image_traversal_absolute_and_target_extensions_blocked(tmp_path):
    root=tmp_path/'data';root.mkdir();(tmp_path/'outside.png').write_bytes(b'x')
    path=tmp_path/'pred.jsonl'
    for image in ['../outside.png',str(tmp_path/'outside.png'),'targets.npz']:
        write_frame(path,image_paths=[image])
        with pytest.raises(ValueError):ReplayStore(path,root).image_path(0,0)


def test_fixture_mode_overrides_trained_metadata(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path)
    metadata=tmp_path/'meta.json';metadata.write_text('{"provenance":"trained"}')
    assert ReplayStore(path,tmp_path,metadata,fixture=True).status()['provenance']=='fixture'
    assert ReplayStore(path,tmp_path,metadata).status()['provenance']=='trained'
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'never unpickle me')
    assert ReplayStore(path,tmp_path,checkpoint).status()['provenance']=='unverified'


def test_six_camera_replay_preserves_view_order_and_isolates_targets(tmp_path):
    path=tmp_path/'pred.jsonl';images=[]
    for camera in range(6):
        image=tmp_path/f'camera-{camera}.jpg';image.write_bytes(bytes([camera]));images.append(image.name)
    write_frame(path,image_paths=images,ground_truth={'boxes':['must not enter scene']})
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps(dict(camera_names=['Front','Front left','Front right','Back','Back left','Back right'])))
    store=ReplayStore(path,tmp_path,metadata)
    frame=store.frame(0)
    assert len(frame['images'])==6 and 'ground_truth' not in frame
    assert [store.image_path(0,c).read_bytes() for c in range(6)]==[bytes([c]) for c in range(6)]
    with pytest.raises(IndexError):store.image_path(0,6)


def test_server_serves_assets_and_rejects_traversal(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path)
    server=module.make_server(path,tmp_path,port=0)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base+'/api/status') as r: assert json.load(r)['frames']==1
        with urllib.request.urlopen(base+'/') as r: assert b'Scene replay' in r.read()
        with urllib.request.urlopen(base+'/replay-requests.js') as r: assert b'LatestFrameRequest' in r.read()
        with urllib.request.urlopen(base+'/api/frame?index=0') as r: assert json.load(r)['frame_id']==10
        for invalid in ['/../pyproject.toml','/%2e%2e/pyproject.toml','/api/frame?index=-1','/api/image?index=0&camera=4']:
            with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(base+invalid)
            assert exc.value.code in (400,404)
    finally:
        server.shutdown();server.server_close();worker.join(timeout=3)


def test_ground_plane_is_explicit_metadata_and_does_not_move_boxes(tmp_path):
    path=tmp_path/'pred.jsonl';original=[[10.,2.,-.3,4.,2.,1.6,0.]]
    write_frame(path,boxes=original,scores=[.9],labels=[0])
    store=ReplayStore(path,tmp_path)
    assert store.status()['ground_z']==-1.5
    assert 'Provisional' in store.status()['ground_source']
    meta=tmp_path/'metadata.json';meta.write_text('{"ground_z":-1.3,"ground_source":"Training calibration"}')
    store=ReplayStore(path,tmp_path,meta)
    assert store.status()['ground_z']==-1.3
    assert store.frame(0)['boxes']==original


def test_prediction_derived_road_visibility_is_exposed_without_target_mask(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path,road_visible=[[True,False]],road_ground_z_m=-1.6,road_target_valid=[[True,True]])
    frame=ReplayStore(path,tmp_path).frame(0)
    assert frame['road_visible']==[[True,False]]
    assert frame['road_ground_z_m']==-1.6
    assert 'road_target_valid' not in frame


def test_decoded_frame_cache_is_bounded_and_shared_by_camera_requests(tmp_path,monkeypatch):
    path=tmp_path/'pred.jsonl'
    records=[dict(sequence='synthetic',frame_id=i,boxes=[],image_paths=['photo.png']*4,road=[[0]]) for i in range(8)]
    path.write_text(''.join(json.dumps(record)+'\n' for record in records))
    (tmp_path/'photo.png').write_bytes(b'fixture image bytes')
    store=ReplayStore(path,tmp_path)
    calls=[];original=module.json.loads
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module.json,'loads',counted)
    store.frame(0)
    for camera in range(4):store.image_path(0,camera)
    assert len(calls)==1
    for index in range(1,8):store.frame(index)
    assert len(store.frame_cache)==2
    assert list(store.frame_cache)==[6,7]
    assert len(calls)==8


def test_cache_survives_append_and_invalidates_replaced_file(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path,frame_id=10)
    store=ReplayStore(path,tmp_path);assert store.frame(0)['frame_id']==10
    cached=store.frame_cache[0]
    with path.open('a') as stream:stream.write(json.dumps(dict(frame_id=11,boxes=[]))+'\n')
    assert store.status()['frames']==2 and store.frame_cache[0] is cached
    replacement=tmp_path/'replacement.jsonl'
    write_frame(replacement,frame_id=99,road=[[0]*100]*10)
    replacement.replace(path)
    assert store.status()['frames']==1
    assert store.frame(0)['frame_id']==99


def test_cache_drops_removed_or_truncated_predictions(tmp_path):
    path=tmp_path/'pred.jsonl';write_frame(path,frame_id=10)
    store=ReplayStore(path,tmp_path);store.frame(0)
    path.write_text('{"frame_id":2}\n')
    assert store.frame(0)['frame_id']==2
    path.unlink()
    assert store.status()['state']=='waiting' and not store.frame_cache
