import argparse, copy, time
from pathlib import Path
import torch
from fsd.runtime import atomic_json
from fsd.engine import evaluate,make_loader
from fsd.model import SceneModel
from fsd.sealing import claim_final_test, identity, sha256

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--output',required=True)
    p.add_argument('--predictions')
    p.add_argument('--diagnostic',choices=['blank','shuffled','empty_history','mismatched_history','initial_only'])
    p.add_argument('--fp32',action='store_true',help='Validation backend experiment; selected modes must be saved into the checkpoint config')
    p.add_argument('--compile',action='store_true',help='Validation torch.compile experiment')
    p.add_argument('--unseal-test',action='store_true')
    p.add_argument('--selection-receipt',default='artifacts/final/selection.json')
    p.add_argument('--final-ledger',default='artifacts/final/test_ledger.json')
    p.add_argument('--resume-final',action='store_true')
    args=p.parse_args()
    if args.split=='test' and not args.unseal_test:
        p.error('Final test is sealed; explicitly freeze/select a checkpoint before --unseal-test')
    if args.split=='test' and (args.fp32 or args.compile):
        p.error('Final test uses frozen checkpoint runtime; backend overrides are validation-only')
    if args.split=='test' and args.diagnostic:
        p.error('Diagnostics are restricted to validation; final test runs the frozen ordinary configuration')
    if args.predictions and Path(args.predictions).exists():
        p.error('Prediction output exists; choose a fresh path')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=copy.deepcopy(ckpt['config'])
    if args.fp32:
        config['runtime']['mixed_precision']=False
    config['runtime']['compiled']=bool(args.compile or config['runtime'].get('compiled',False))
    runtime_backend=dict(precision='fp16' if config['runtime'].get('mixed_precision',True) else 'fp32',
                         compiled=config['runtime']['compiled'],stage=ckpt['stage'],
                         refinement_enabled=ckpt['stage']=='b' and args.diagnostic!='initial_only')
    frozen_inputs=identity(args.checkpoint,config)
    ledger=None
    if args.split=='test':
        ledger=claim_final_test(args.checkpoint,config,args.selection_receipt,args.final_ledger,args.resume_final)
    config['model']['pretrained']=False
    try:
        model=SceneModel(config).cuda()
        model.load_state_dict(ckpt['model'])
        if runtime_backend['compiled']:
            model=torch.compile(model)
        result=evaluate(model,make_loader(config,args.split),config,refine=ckpt['stage']=='b',
                        diagnostic=args.diagnostic,output_path=args.predictions)
        if sha256(args.checkpoint)!=frozen_inputs['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during evaluation; result cannot be attributed safely')
        atomic_json(args.output,dict(split=args.split,checkpoint=args.checkpoint,diagnostic=args.diagnostic,
            checkpoint_sha256=frozen_inputs['checkpoint_sha256'],dataset_manifest_sha256=frozen_inputs['manifest_sha256'],
            configuration_sha256=frozen_inputs['config_sha256'],source_sha256=frozen_inputs['source_sha256'],runtime_backend=runtime_backend,metrics=result))
        if ledger is not None:
            ledger.update(status='complete',finished=time.time(),report=args.output,predictions=args.predictions)
            atomic_json(args.final_ledger,ledger)
    except BaseException as error:
        if ledger is not None:
            ledger.update(status='failed',finished=time.time(),error=repr(error))
            atomic_json(args.final_ledger,ledger)
        raise
