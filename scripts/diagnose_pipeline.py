"""Diagnose CUDA repeatability separately from target isolation on real data."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import copy, json
import torch
from fsd.engine import make_loader, amp_context
from fsd.model import SceneModel
from fsd.losses import compute_losses
from fsd.runtime import load_config, to_device, seed_all, atomic_json

TARGETS={"boxes","labels","road","depth","depth_target","detection_valid","instance_ids"}
KEYS=("depth_logits","road_logits","center_logits","geometry_map","initial_boxes","refined_boxes")
class GuardedBatch(dict):
    reads=set()
    def __getitem__(self,key):
        self.reads.add(key)
        if key in TARGETS: raise AssertionError("Model read forbidden target: "+key)
        return super().__getitem__(key)
    def get(self,key,*args):
        self.reads.add(key)
        if key in TARGETS: raise AssertionError("Model read forbidden target: "+key)
        return super().get(key,*args)

def stats(a,b):
    result={}
    for key in KEYS:
        d=(a[key].float()-b[key].float()).abs()
        result[key]={"equal":torch.equal(a[key],b[key]),"max_abs":float(d.max()),"mean_abs":float(d.mean()),"unequal":int((d!=0).sum())}
    result["road_argmax_changed"]=int((a["road_logits"].argmax(1)!=b["road_logits"].argmax(1)).sum())
    result["proposal_cells_changed"]=int((a["proposal_cells"]!=b["proposal_cells"]).sum())
    return result

@torch.no_grad()
def run(model,batch,config):
    with amp_context(config):
        out=model(batch,refine=True)
    return {k:out[k].detach().cpu() for k in KEYS+("proposal_cells",)}

def main():
    config=load_config("configs/baseline.yaml");config["train"]["workers"]=0
    seed_all(42)
    batch=to_device(next(iter(make_loader(config,"train"))),"cuda")
    snapshot={k:v.clone() for k,v in batch.items() if isinstance(v,torch.Tensor)}
    model=SceneModel(config).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    with amp_context(config):
        out=model(batch,refine=True);loss=compute_losses(out,batch,config)["total"]
    loss.backward();opt.step();opt.zero_grad(set_to_none=True);del out,loss
    model.eval()
    stripped={k:v for k,v in batch.items() if k not in TARGETS}
    report={"frame":int(batch["frame_id"][0]),"sequence":batch["sequence"],"torch":torch.__version__,"modes":{}}
    for deterministic in (False,True):
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic=deterministic
        mode={}
        try:
            reference=run(model,batch,config)
            mode["same_batch_repeats"]=[stats(reference,run(model,batch,config)) for _ in range(3)]
            mode["target_stripped_repeats"]=[stats(reference,run(model,stripped,config)) for _ in range(3)]
            guarded=GuardedBatch(batch);mode["guarded_targets"]=stats(reference,run(model,guarded,config))
            mode["model_reads"]=sorted(guarded.reads)
        except Exception as exc:
            mode["error"]=repr(exc)
        report["modes"][str(deterministic)]=mode
        print(json.dumps({"mode":deterministic,"result":mode}),flush=True)
    # Isolate atomic lifting from every other stage without changing model code.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=False
    before=model.lift.register_forward_pre_hook(lambda m,args:torch.use_deterministic_algorithms(True))
    after=model.lift.register_forward_hook(lambda m,args,out:torch.use_deterministic_algorithms(False))
    try:
        reference=run(model,batch,config)
        report["lift_only_deterministic"]={"same":stats(reference,run(model,batch,config)),"stripped":stats(reference,run(model,stripped,config))}
    finally:
        before.remove();after.remove()
    report["mutated_input_tensors"]=[k for k,v in snapshot.items() if not torch.equal(v,batch[k])]
    atomic_json("artifacts/pipeline/determinism-diagnosis.json",report)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
