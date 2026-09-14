"""Exercise actual engine.train interruption/resumption on four real updates.

This is distinct from verify_resume.py's standalone optimizer test. Runs are
isolated in fresh processes; no held-out split or epoch-end validation is read.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse,copy,json,subprocess,sys
from pathlib import Path
import torch
import yaml
from fsd.runtime import atomic_json,load_config
from verify_resume import compare_trees,tree_digest,within_repeat_noise


def worker(args):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    from fsd.model import SceneModel
    from fsd.engine import train
    trace=[]
    original=SceneModel.forward
    def traced_forward(self,batch,*a,**kw):
        if self.training:
            trace.append({"sequence":list(batch["sequence"]),"frame":int(batch["frame_id"][0]),
                          "timestamp":float(batch["timestamp"][0]),"refine":kw.get("refine",True)})
        return original(self,batch,*a,**kw)
    SceneModel.forward=traced_forward
    config=load_config(args.config)
    try:
        train(config,args.run,stage="a",resume=args.resume,max_updates=args.max_updates)
    finally:
        atomic_json(Path(args.run)/"forward_trace.json",{"pid":os.getpid(),"frames":trace})


def _launch(config_path,run,max_updates,resume=None):
    command=[sys.executable,str(Path(__file__).resolve()),"--worker","--config",str(config_path),
             "--run",str(run),"--max-updates",str(max_updates)]
    if resume:command.extend(["--resume",str(resume)])
    run.mkdir(parents=True,exist_ok=True)
    with (run/"process.log").open("w",encoding="utf8") as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    checkpoint=torch.load(run/"latest.pt",map_location="cpu",weights_only=False)
    trace=json.loads((run/"forward_trace.json").read_text())
    return checkpoint,trace


def main(args):
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(load_config(args.config))
    config["train"]["workers"]=0
    config["train"]["accumulate"]=1
    config["runtime"]["amp_init_scale"]=1024.
    from fsd.data import Kitti360Dataset
    data=Kitti360Dataset(config["data"]["root"],"train",config)
    if len(data)<=4:raise ValueError("Need >4 real frames to stay before epoch-end validation")
    expected=[{"sequence":[r["sequence"]],"frame":int(r["frame_id"]),"timestamp":r["timestamp"],"refine":False}
              for r in data.samples[:4]]
    config_path=destination/"config.yaml";config_path.write_text(yaml.safe_dump(config),encoding="utf8")
    baseline,baseline_trace=_launch(config_path,destination/"uninterrupted",4)
    repeated,repeated_trace=_launch(config_path,destination/"repeat_control",4)
    first,first_trace=_launch(config_path,destination/"interrupted",2)
    restored,restored_trace=_launch(config_path,destination/"resumed",4,destination/"interrupted/latest.pt")
    keys=("model","optimizer","scheduler","scaler","rng","history")
    comparisons={k:compare_trees(baseline[k],restored[k],rtol=0,atol=0) for k in keys}
    noise={k:compare_trees(baseline[k],repeated[k],rtol=0,atol=0) for k in keys}
    exact={k:tree_digest(baseline[k])==tree_digest(restored[k]) for k in keys}
    trace_checks={"uninterrupted_first_four":baseline_trace["frames"]==expected,
                  "control_first_four":repeated_trace["frames"]==expected,
                  "stopped_after_two":first_trace["frames"]==expected[:2],
                  "resume_only_remaining_two":restored_trace["frames"]==expected[2:],
                  "combined_no_skip_no_repeat":first_trace["frames"]+restored_trace["frames"]==expected}
    boundary_checks={"mid_epoch":first["epoch"]==0,"next_step_two":first["next_step"]==2,
                     "successful_updates_two":first["updates"]==2,
                     "scheduler_unadvanced":first["scheduler"]["last_epoch"]==0,
                     "resumed_next_step_four":restored["next_step"]==4,
                     "resumed_successful_updates_four":restored["updates"]==4,
                     "resumed_epoch_zero":restored["epoch"]==0,
                     "resumed_scheduler_unadvanced":restored["scheduler"]["last_epoch"]==0,
                     "no_amp_skips":first.get("overflow_count",0)==0 and restored.get("overflow_count",0)==0,
                     "history_empty_stage_a":len(first["history"])==0 and len(restored["history"])==0}
    pids=[x["pid"] for x in (baseline_trace,repeated_trace,first_trace,restored_trace)]
    report={"protocol":"actual_engine_stage_a_mid_epoch_stop2_resume4_vs_uninterrupted4",
            "data_manifest":config["data"].get("manifest","manifest.json"),"expected_frames":expected,
            "worker_pids":pids,"distinct_processes":len(set(pids))==4,"trace_checks":trace_checks,
            "checkpoint_boundary_checks":boundary_checks,"bitwise_state_parity":exact,
            "resumed_comparison":comparisons,"independent_repeat_control":noise,
            "precision":"FP16 autocast; deterministic CUDA algorithms strict; loss scale1024",
            "scope":"Real image4-camera model and engine checkpoint path; stageA excludes temporal-refiner backward. Separate verify_resume.py covers temporal state."}
    report["passed"]=(all(trace_checks.values()) and all(boundary_checks.values()) and all(exact.values())
                      and report["distinct_processes"] and all(v["passed"] for v in noise.values()))
    atomic_json(destination/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:raise AssertionError("Engine resume gate failed; see report.json")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/overfit16.yaml")
    p.add_argument("--output",default="artifacts/engine-resume")
    p.add_argument("--worker",action="store_true")
    p.add_argument("--run")
    p.add_argument("--resume")
    p.add_argument("--max-updates",type=int)
    args=p.parse_args()
    worker(args) if args.worker else main(args)
