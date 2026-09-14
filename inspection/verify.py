"""Verify real saved references, existing metric agreement, and immutable source."""
from pathlib import Path
import json,numpy as np,urllib.request,hashlib
from fsd.sealing import sha256,source_hash
from fsd.metrics import detection_match,wrap_angle
from build import frame_matches,summarize
R=Path(__file__).resolve().parents[1];O=R/'artifacts/paired-inspection';D=R/'data/kitti360'
m=json.loads((O/'manifest.json').read_text());old=json.loads((R/'artifacts/final/selection.json').read_text());assert source_hash()==old['identity']['source_sha256']
pred=[json.loads(x) for x in (R/'artifacts/final-measurements/reference/scenes.jsonl').open() if x.strip()]
for i,p in enumerate(pred):
    f=O/'frames'/f'{i:04d}.json';assert sha256(f)==m['frame_sha256'][str(i)];r=json.loads(f.read_text());assert r['predictions']==p
    with np.load(D/r['provenance']['cache_path'],allow_pickle=False) as z:
        for k,v in r['targets'].items():
            actual=np.asarray(v).reshape(-1,7) if k=='boxes' else np.asarray(v)
            np.testing.assert_array_equal(actual,z[k])
    assert r['frame_id']==p['frame_id'] and r['timestamp']==p['timestamp']
s=json.loads((O/'consistency.json').read_text());metric=json.loads((R/'artifacts/final-measurements/reference/validation.json').read_text())['metrics']
a=s['thresholds']['0.01'];assert a['counts']['matched_car_observations']==metric['objects']['bev_ap50']['classes']['car']['true_positives']==114
assert a['counts']['matched_both']==metric['temporal']['matched_transition_count']==65
for k in ('center','dimension','yaw'):
    key=k+'_error_change_'+('deg' if k=='yaw' else 'm');official={'center':'center_residual_change_m','dimension':'dimension_residual_change_m','yaw':'yaw_residual_change_deg'}[k]
    np.testing.assert_allclose(a['conditional_error_changes'][key]['mean'],metric['temporal'][official],rtol=0,atol=1e-12)
# Analytic event controls ensure matched-only dropout omissions are now counted,
# and dataset gaps are not silently bridged. No model/target implementation mirroring.
def frame(t,hit=True,track=1):
    row=dict(gt_id=9,label=0,matched_prediction=0 if hit else None,track_id=track,center_error_world=[.2,0,0],dimension_error=[0,0,0],yaw_error_rad=.1)
    return dict(sequence='x',timestamp=t,matches={'k':{'targets':[row]}})
c=summarize([frame(0),frame(.5,False),frame(1),frame(1.5,track=2),frame(4,track=3)],'k')
assert c['counts']['disappeared']==1 and c['counts']['reappeared']==1 and c['counts']['track_id_changes']==1 and c['counts']['supported_adjacent_pairs']==3
assert all(v['mean']==0 for v in c['conditional_error_changes'].values())
u='http://127.0.0.1:8773'
def get(path):return urllib.request.urlopen(u+path).read()
for i in (0,1,98,143,144,196):
    r=json.loads(get(f'/reference/frame?index={i}'));normal=json.loads(get(f'/api/frame?index={i}'))
    assert 'targets' not in normal and all(normal[k]==v for k,v in r['predictions'].items() if k in normal)
    assert hashlib.sha256(get(f'/reference/targets?index={i}')).hexdigest()==r['provenance']['cache_sha256']
    assert hashlib.sha256(get(f'/reference/lidar?index={i}')).hexdigest()==r['provenance']['lidar_sha256']
    for c in range(4):assert hashlib.sha256(get(f'/api/image?index={i}&camera={c}')).hexdigest()==r['provenance']['image_sha256'][c]
out=dict(status='passed',all197_payloads_equal_original_predictions_and_npz=True,existing114_matches_65_transitions_reproduced=True,existing_temporal_means_reproduced=True,dropout_identity_and_gap_controls=True,six_live_api_download_and_24image_hash_checks=True,prediction_api_has_no_targets=True,frozen_source_unchanged=True,bundle_sha256=sha256(O/'manifest.json'),consistency_sha256=sha256(O/'consistency.json'),verification_source_sha256=sha256(Path(__file__)))
(O/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
