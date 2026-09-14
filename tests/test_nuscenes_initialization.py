"""Isolated module load avoids mixing KITTI and nuScenes fsd namespaces."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import torch

SOURCE=Path(__file__).resolve().parents[1]/'experiments/nuscenes/code/src/fsd/initialization.py'
spec=importlib.util.spec_from_file_location('nuscenes_initialization_under_test',SOURCE)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

@pytest.fixture
def parent(tmp_path):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(dataset='nuscenes',version='v1.0-trainval',preparation_complete=True)))
    cfg=dict(data=dict(dataset='nuscenes',version='v1.0-trainval',required_cameras=6,root=str(tmp_path),manifest='manifest.json'),
             model=dict(camera_geometry='pinhole',fpn_channels=128),targets={'depth':'static'},render={'z':0},migration={'version':4})
    ck=dict(config=copy.deepcopy(cfg),model={'weight':torch.tensor([2.])},stage='a',epoch=18,updates=9,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),optimizer={'state':{'old':1}})
    path=tmp_path/'parent.pt';torch.save(ck,path)
    cfg['experiment']=dict(id='branch',initialization=str(path),initialization_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return cfg,path,ck

def test_valid_transfer_returns_only_weights_and_parent_receipt(parent):
    cfg,p,ck=parent;cfg['loss']={'initial':2};cfg['train']={'lr':2e-5}
    payload,receipt=mod.load_branch_parent(cfg,p)
    assert set(payload)=={'model'}
    assert torch.equal(payload['model']['weight'],ck['model']['weight'])
    assert receipt['parent_epoch']==18 and receipt['mode']=='weights_only'

@pytest.mark.parametrize('section,key,value',[
    ('data','dataset','kitti360'),('data','version','v1.0-mini'),('data','required_cameras',4),
    ('data','root','another-cache'),('model','camera_geometry','fisheye'),('model','fpn_channels',256),
    ('targets','depth','different'),('render','z',1),('migration','version',5)])
def test_reject_incompatible_destination(parent,section,key,value):
    cfg,p,_=parent;cfg[section][key]=value
    with pytest.raises(ValueError):mod.load_branch_parent(cfg,p)

@pytest.mark.parametrize('field', ['id','initialization','initialization_sha256'])
def test_requires_explicit_parent_identity(parent,field):
    cfg,p,_=parent;del cfg['experiment'][field]
    with pytest.raises(ValueError):mod.load_branch_parent(cfg,p)

def test_changed_manifest_is_rejected(parent):
    cfg,p,_=parent
    manifest=Path(cfg['data']['root'])/'manifest.json';manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='manifest differ'):mod.load_branch_parent(cfg,p)

@pytest.mark.parametrize('kind',['mini','kitti','missing_manifest','bad_stage'])
def test_reject_incompatible_parent_even_with_correct_hash(parent,kind):
    cfg,p,ck=parent
    if kind=='mini':ck['config']['data']['version']='v1.0-mini'
    elif kind=='kitti':ck['config']['data']['dataset']='kitti360'
    elif kind=='missing_manifest':del ck['manifest_sha256']
    else:ck['stage']='unknown'
    torch.save(ck,p);cfg['experiment']['initialization_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError):mod.load_branch_parent(cfg,p)

def test_exact_loader_rejects_silent_dtype_conversion():
    model=torch.nn.Linear(2,2)
    with pytest.raises(ValueError,match='dtype'):
        mod.load_exact_model_state(model,{k:v.half() for k,v in model.state_dict().items()})

def test_exact_loader_rejects_missing_keys():
    with pytest.raises(RuntimeError):mod.load_exact_model_state(torch.nn.Linear(2,2),{})

def test_exact_loader_preserves_buffers_and_parameters():
    model=torch.nn.BatchNorm1d(3);state={k:v.clone() for k,v in model.state_dict().items()}
    state['running_mean']+=2
    mod.load_exact_model_state(model,state)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
