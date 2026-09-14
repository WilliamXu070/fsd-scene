"""Real-image checkpoint restore and fresh-process next-update verification.

Runs independently of engine.train. This verifies checkpoint completeness, not
held-out accuracy. Parent must schedule this short job while the GPU is idle.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, copy, gc, hashlib, json, random, subprocess, sys, warnings
from pathlib import Path
import numpy as np
import torch
from fsd.runtime import (atomic_json, history_for, load_config, restore_rng,
                         rng_state, save_checkpoint, seed_all, to_device)


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def tree_digest(value):
    digest=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous();digest.update(str((a.dtype,tuple(a.shape))).encode());digest.update(a.numpy().tobytes())
        elif isinstance(v,np.ndarray):
            digest.update(str((v.dtype,v.shape)).encode());digest.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):digest.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            digest.update(type(v).__name__.encode())
            for x in v:visit(x)
        else:digest.update(repr(v).encode())
    visit(value)
    return digest.hexdigest()


def snapshot(model,optimizer,scheduler,scaler,history,config,indices,updates):
    return cpu_tree(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),
                         history=history,config=config,indices=indices,updates=updates,epoch=0,next_step=updates))


def restore_training_state(payload,model,optimizer,scheduler,scaler,device):
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    history=to_device(payload["history"],device)
    # Constructors and loaders may consume RNG; restoration is deliberately last.
    restore_rng(payload["rng"])
    observed={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
              "scaler":scaler.state_dict(),"rng":rng_state(),"history":history}
    checks={k:tree_digest(v)==tree_digest(payload[k]) for k,v in observed.items()}
    if not all(checks.values()):raise AssertionError(f"Incomplete checkpoint restore: {checks}")
    return history,checks


def compare_trees(left,right,rtol=1e-5,atol=1e-7):
    report={"max_abs":0.,"different_tensors":0,"tensors":0,"elements":0,"sum_squared_difference":0.,"failures":[]}
    def visit(a,b,path):
        if isinstance(a,torch.Tensor):
            report["tensors"]+=1
            if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:
                report["failures"].append(path+": shape/dtype");return
            if a.numel():
                difference=a.double()-b.double()
                diff=difference.abs().max().item()
                report["elements"]+=a.numel()
                report["sum_squared_difference"]+=difference.square().sum().item()
                report["max_abs"]=max(report["max_abs"],diff)
                report["different_tensors"]+=int(not torch.equal(a,b))
                if not torch.allclose(a,b,rtol=rtol,atol=atol):report["failures"].append(path)
        elif isinstance(a,dict):
            if set(a)!=set(b):report["failures"].append(path+": keys");return
            for key in a:visit(a[key],b[key],path+"."+str(key))
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):report["failures"].append(path+": length");return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+"."+str(i))
        elif isinstance(a,np.ndarray):
            if not np.array_equal(a,b):report["failures"].append(path)
        elif a!=b:report["failures"].append(path)
    visit(left,right,"state")
    report["rmse"]=(report.pop("sum_squared_difference")/max(1,report["elements"]))**.5
    report["passed"]=not report["failures"]
    return report


def within_repeat_noise(control,observed,factor=2.,floor=1e-7):
    bounds={key:factor*control[key]+floor for key in ("max_abs","rmse")}
    return {"bounds":bounds,"observed":{k:observed[k] for k in bounds},
            "passed":all(observed[k]<=bounds[k] for k in bounds)}


def rng_probe():
    state=rng_state()
    values={"python":random.random(),"numpy":float(np.random.rand()),"torch":torch.rand(3).tolist(),
            "cuda":torch.rand(3,device="cuda").cpu().tolist() if torch.cuda.is_available() else []}
    restore_rng(state)
    return values


def construct(config):
    from fsd.model import SceneModel
    model=SceneModel(config).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["train"]["lr"],weight_decay=config["train"]["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=24)
    # This diagnostic starts conservatively to test resume rather than AMP backoff.
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    return model,optimizer,scheduler,scaler


def take_update(model,optimizer,scheduler,scaler,batch,history,config):
    from fsd.engine import amp_context
    from fsd.losses import compute_losses
    history=history_for(history,batch)
    optimizer.zero_grad(set_to_none=True)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        with amp_context(config):
            output=model(batch,history,refine=True)
            losses=compute_losses(output,batch,config,refine=True)
        if not torch.isfinite(losses["total"]):raise FloatingPointError("Non-finite diagnostic loss")
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:raise FloatingPointError("Non-finite diagnostic gradients: "+repr(bad[:8]))
        image_grad=float(model.image_encoder.backbone.conv1.weight.grad.float().abs().sum())
        if not image_grad>0:raise AssertionError("No actual image/backbone gradient")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update();scheduler.step()
    info={"loss":float(losses["total"].detach()),"gradient_norm":float(norm),"image_gradient_l1":image_grad,
          "history_used":len(history),"frame":int(batch["frame_id"][0]),"warnings":sorted(set(str(w.message) for w in notices))}
    history=(history+[output["state"]])[-config["model"].get("history_length",3):]
    return history,info


def configure_determinism():
    # grid_sample backward may lack a deterministic kernel. Warn and measure
    # that noise with an independent same-process repeat; never suppress it.
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_pair(config,indices):
    from fsd.engine import make_loader
    config=copy.deepcopy(config);config["train"]["workers"]=0
    return [to_device(b,"cuda") for b in make_loader(config,"train",indices)]


def worker(args):
    configure_determinism()
    payload=torch.load(args.worker,map_location="cpu",weights_only=False)
    config=copy.deepcopy(payload["config"]);config["model"]["pretrained"]=False
    batches=load_pair(config,payload["indices"])
    model,optimizer,scheduler,scaler=construct(config)
    history,checks=restore_training_state(payload,model,optimizer,scheduler,scaler,"cuda")
    probe=rng_probe()
    history,info=take_update(model,optimizer,scheduler,scaler,batches[1],history,config)
    next_state=snapshot(model,optimizer,scheduler,scaler,history,payload["config"],payload["indices"],2)
    save_checkpoint(Path(args.output)/"fresh_second.pt",**next_state)
    atomic_json(Path(args.output)/"fresh_info.json",dict(restore_checks=checks,rng_probe=probe,update=info,pid=os.getpid()))


def main(args):
    configure_determinism();seed_all(42)
    config=load_config(args.config);config["train"]["workers"]=0
    from fsd.data import Kitti360Dataset
    dataset=Kitti360Dataset(config["data"]["root"],"train",config)
    indices=None
    for i in range(len(dataset.samples)-1):
        a,b=dataset.samples[i:i+2]
        if a["sequence"]==b["sequence"] and 0<b["timestamp"]-a["timestamp"]<=1.01:
            indices=[i,i+1];break
    if indices is None:raise RuntimeError("Two consecutive real training timestamps required")
    batches=load_pair(config,indices)
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    model,optimizer,scheduler,scaler=construct(config)
    history,first_info=take_update(model,optimizer,scheduler,scaler,batches[0],[],config)
    base=snapshot(model,optimizer,scheduler,scaler,history,config,indices,1)
    save_checkpoint(destination/"first.pt",**base)
    expected_probe=rng_probe()
    history,second_info=take_update(model,optimizer,scheduler,scaler,batches[1],history,config)
    expected=snapshot(model,optimizer,scheduler,scaler,history,config,indices,2)
    # Independent repeat from the same saved point measures unavoidable backward noise.
    history,restore_checks=restore_training_state(base,model,optimizer,scheduler,scaler,"cuda")
    history,repeat_info=take_update(model,optimizer,scheduler,scaler,batches[1],history,config)
    repeated=snapshot(model,optimizer,scheduler,scaler,history,config,indices,2)
    noise={k:compare_trees(expected[k],repeated[k]) for k in ("model","optimizer","scheduler","scaler","rng")}
    # Check temporal discontinuity handling using the REAL saved state.
    foreign=dict(batches[1],sequence=["other_sequence"])
    future=dict(batches[1],timestamp=batches[0]["timestamp"]-.5)
    stale=dict(batches[1],timestamp=batches[0]["timestamp"]+5.)
    history_checks={"consecutive_kept":len(history_for(to_device(base["history"],"cuda"),batches[1]))==1,
                    "foreign_reset":not history_for(base["history"],foreign),
                    "reverse_reset":not history_for(base["history"],future),
                    "gap_reset":not history_for(base["history"],stale)}
    del model,optimizer,scheduler,scaler,history,batches,repeated;gc.collect();torch.cuda.empty_cache()
    command=[sys.executable,str(Path(__file__).resolve()),"--worker",str(destination/"first.pt"),"--output",str(destination)]
    with (destination/"fresh_process.log").open("w",encoding="utf8") as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    observed=torch.load(destination/"fresh_second.pt",map_location="cpu",weights_only=False)
    worker_info=json.loads((destination/"fresh_info.json").read_text())
    comparison={k:compare_trees(expected[k],observed[k]) for k in ("model","optimizer","scheduler","scaler","rng","history")}
    exact_loss=second_info["loss"]==worker_info["update"]["loss"]
    noise_bounded={k:within_repeat_noise(noise[k],comparison[k]) for k in ("model","optimizer")}
    report={"protocol":"actual_image_network_fresh_process_next_update_after_complete_restore",
            "parent_pid":os.getpid(),"worker_pid":worker_info["pid"],"indices":indices,"first_update":first_info,
            "second_update":second_info,"repeated_second_update":repeat_info,"fresh_second_update":worker_info["update"],
            "restored_exactly":worker_info["restore_checks"],"parent_restore_exactly":restore_checks,
            "rng_probe_exact":expected_probe==worker_info["rng_probe"],"next_loss_exact":exact_loss,
            "same_process_repeat_noise":noise,"fresh_process_comparison":comparison,"history_reset_checks":history_checks,
            "noise_bounded_next_update":noise_bounded,
            "numerical_tolerance":{"rtol":1e-5,"atol":1e-7,"interpretation":"strict allclose diagnostic only; unsupported FP16 grid backward judged by independent repeat max/RMS envelope"},
            "training_precision":"FP16 autocast with initial loss scale128; deterministic forward and math attention, warn-only for unsupported grid-sampling backward"}
    report["passed"]=(all(v["passed"] for v in noise_bounded.values())
                      and all(comparison[k]["passed"] for k in ("scheduler","scaler","rng","history"))
                      and all(worker_info["restore_checks"].values()) and all(restore_checks.values())
                      and all(history_checks.values()) and report["rng_probe_exact"] and exact_loss
                      and worker_info["update"]["history_used"]==1 and os.getpid()!=worker_info["pid"])
    report["status"]="restoration_verified_next_update_within_measured_backward_noise" if report["passed"] else "failed"
    report["limitations"]=["FP16 CUDA grid_sample backward is nondeterministic; bitwise training replay is not claimed.",
                           "This standalone two-update protocol verifies full state restoration, not engine interruption handling.",
                           "Repeat-noise bounds use one independent same-process repeat, not a long-run reproducibility study."]
    atomic_json(destination/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:raise AssertionError("Resume verification failed; see report.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/overfit16.yaml")
    parser.add_argument("--output",default="artifacts/resume")
    parser.add_argument("--worker")
    args=parser.parse_args()
    worker(args) if args.worker else main(args)
