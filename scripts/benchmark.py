"""Calibrated multi-camera replay timing, stage profiling and bounded parity.

Never interpret renderer FPS as fresh perception throughput. Targets are not
loaded/transferred by this benchmark. Final-stage timing excludes reset frames
from the main steady-history aggregate and reports those frames separately.
"""
from __future__ import annotations
import argparse,copy,json,math,time,hashlib,os,sys
from collections import Counter,defaultdict
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from fsd.runtime import atomic_json,to_device,history_for
from fsd.model import SceneModel
from fsd.metrics import latency_summary,pairwise_iou,wrap_angle
from fsd.inference import select_detections,SceneTracker
from fsd.sealing import identity,sha256

INFERENCE_KEYS={"images","camera_valid","camera_to_ego","intrinsics","distortion","xi","camera_model",
                "rays","ray_valid","ego_to_world","timestamp","sequence","frame_id"}
CALIBRATION_KEYS=("camera_to_ego","intrinsics","distortion","xi","camera_model","rays","ray_valid")


def _summary(values):
    return latency_summary(values) if values else None


def precision_context(fp32=False):
    return nullcontext() if fp32 else torch.autocast("cuda",dtype=torch.float16)


def inference_batch(batch):
    """Keep an explicit target-free sensor interface before any GPU transfer."""
    missing=INFERENCE_KEYS-set(batch)
    if missing:raise KeyError(f"Incomplete inference inputs: {sorted(missing)}")
    return {key:batch[key] for key in INFERENCE_KEYS}


def load_sensor_samples(config,limit=32,split="val"):
    nuscenes=config['data'].get('dataset')=='nuscenes'
    if nuscenes:
        from fsd.nuscenes_data import NuScenesDataset
        dataset=NuScenesDataset(config['data']['root'],split,config)
    else:
        from fsd.data import Kitti360Dataset
        dataset=Kitti360Dataset(config["data"]["root"],split,config)
    cameras=6 if nuscenes else 4
    if limit<4:raise ValueError("Cache at least four timestamps for temporal timing")
    samples=[];decode_ms=[];preprocess_ms=[];pose_ms=[]
    for row in dataset.samples[:limit]:
        start=time.perf_counter()
        arrays=[]
        for path in row["images"]:
            with Image.open(dataset.root/path) as image:
                arrays.append(np.asarray(image.convert("RGB")).copy())
        after_decode=time.perf_counter()
        images=torch.from_numpy(np.stack(arrays).transpose(0,3,1,2).copy()).float()/255.
        sample={"images":images[None].pin_memory(),"camera_valid":torch.ones(1,cameras,dtype=torch.bool).pin_memory(),
                "sequence":[row["sequence"]],"frame_id":torch.tensor([row["frame_id"]]),
                "timestamp":torch.tensor([row["timestamp"]],dtype=torch.float64)}
        if not nuscenes:
            for key in CALIBRATION_KEYS:
                sample[key]=torch.from_numpy(dataset.calibration[key].copy())[None].pin_memory()
        after_preprocess=time.perf_counter()
        with np.load(dataset.root/row["cache"],allow_pickle=False) as cache:
            # Access sensor calibration/pose arrays only; no ground-truth arrays.
            if nuscenes:
                for key in CALIBRATION_KEYS:
                    sample[key]=torch.from_numpy(cache[key].copy())[None].pin_memory()
            sample["ego_to_world"]=torch.from_numpy(cache["ego_to_world"].copy())[None].pin_memory()
        finish=time.perf_counter()
        if sample["images"].shape[1]!=cameras:raise ValueError(f"Baseline requires all {cameras} cameras")
        samples.append(inference_batch(sample))
        decode_ms.append((after_decode-start)*1000)
        preprocess_ms.append((after_preprocess-after_decode)*1000)
        pose_ms.append((finish-after_preprocess)*1000)
    if not samples:raise ValueError("No prepared replay samples")
    report={"samples":len(samples),"image_decode":_summary(decode_ms),"rgb_tensor_and_calibration":_summary(preprocess_ms),
            "supplied_pose_cache_read":_summary(pose_ms),"source":"processed cached images at model resolution",
            "pose_read_scope":"Per-frame calibration plus pose" if nuscenes else "Supplied pose; fixed calibration cached separately",
            "limits":"Excludes native-resolution camera acquisition, original-image resizing, and pose estimation; those are not available in this replay cache."}
    return samples,report


def _json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _cpu_sensor_digest(sample):
    """Hash actual decoded sensor inputs; never inspect targets or initialize CUDA."""
    digest=hashlib.sha256()
    for key in sorted(INFERENCE_KEYS):
        value=sample[key]
        digest.update(key.encode())
        if isinstance(value,torch.Tensor):
            if value.device.type!="cpu":raise ValueError("Workload identity must be computed before GPU transfer")
            value=value.detach().contiguous()
            metadata=[str(value.dtype),list(value.shape)]
            digest.update(json.dumps(metadata,separators=(",",":")).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:digest.update(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
    return digest.hexdigest()


def sensor_workload(samples):
    """Identity of ordered camera/calibration/pose tensors cached for replay."""
    records=[]
    for index,raw in enumerate(samples):
        raw=inference_batch(raw)
        pose=raw["ego_to_world"]
        if pose.device.type!="cpu":raise ValueError("Pose identity requires CPU inputs")
        records.append(dict(index=index,sequence=raw["sequence"][0],frame_id=int(raw["frame_id"][0]),
            timestamp=float(raw["timestamp"][0]),sensor_sha256=_cpu_sensor_digest(raw),
            ego_pose_sha256=hashlib.sha256(pose.contiguous().numpy().tobytes()).hexdigest()))
    return dict(schema_version=1,cached_scenes=len(records),records=records,ordered_sha256=_json_digest(records),
        interpretation="Actual decoded image, calibration and supplied-pose tensors; no targets. Hashing occurs outside timed intervals.")


def replay_order_summary(indices):
    """Accepted cache indices, preserving repetition/order and empty cases."""
    indices=[int(i) for i in indices]
    return dict(count=len(indices),indices=indices,order_sha256=_json_digest(indices),
                frequencies={str(k):v for k,v in sorted(Counter(indices).items())})


def history_ready(history,raw,required,max_age=2.):
    if required==0:return True,"single_frame_stage"
    if len(history)<required:return False,"insufficient_history_frames"
    now=float(raw["timestamp"][0])
    for state in history[-required:]:
        if state.get("sequence")!=raw["sequence"]:return False,"sequence_reset"
        age=now-float(state["timestamp"].detach().cpu()[0])
        if not 0<age<=max_age:return False,"history_age_invalid"
        if "valid" in state and not bool(state["valid"].any()):return False,"history_has_no_valid_object_tokens"
    return True,"full_history"


def _history_requirement(config,refine):
    return int(config["model"].get("history_length",3)) if refine else 0


def model_call(model,batch,history,refine,fp32):
    with precision_context(fp32):return model(batch,history,refine=refine)


def timed_replay(model,samples,config,refine,args):
    repeats=[];first_update=None
    required=_history_requirement(config,refine)
    for rep in range(args.repeats):
        history=[];tracker=SceneTracker();timings=defaultdict(list);reset_timings=[]
        exclusions=Counter();attempt=0;warmed=0;collected=0
        warmup_indices=[];timed_indices=[]
        max_attempts=(args.warmup+args.updates+len(samples))*12
        torch.cuda.reset_peak_memory_stats()
        while collected<args.updates:
            if attempt>=max_attempts:
                raise RuntimeError(f"Cannot collect enough full-history observations: {dict(exclusions)}. Check sequence continuity and valid memory.")
            sample_index=attempt%len(samples);raw=samples[sample_index];attempt+=1
            begin=time.perf_counter()
            transfer_start,transfer_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            transfer_start.record();batch=to_device(raw,"cuda");transfer_end.record();torch.cuda.synchronize()
            after_transfer=time.perf_counter()
            history=history_for(history,batch)
            after_history=time.perf_counter()
            model_start,model_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            model_start.record();out=model_call(model,batch,history,refine,args.fp32);model_end.record()
            torch.cuda.synchronize();after_model=time.perf_counter()
            detections=select_detections(out,threshold=args.threshold,refined=refine)
            after_nms=time.perf_counter()
            road_prob=out["road_logits"][0].float().softmax(0)
            road=road_prob.argmax(0).cpu().numpy()
            confidence=road_prob.max(0).values.cpu().numpy()
            after_road=time.perf_counter()
            timestamp=float(raw["timestamp"][0]);pose=raw["ego_to_world"][0].numpy()
            tracker.update(detections,timestamp,pose,raw["sequence"][0])
            after_tracking=time.perf_counter()
            used_history=history
            history=(history+[out["state"]])[-max(1,required):] if required else []
            end=time.perf_counter()
            # Eligibility checks happen AFTER the timed interval so GPU scalar
            # reads used solely for benchmark classification do not pollute timing.
            eligible,reason=history_ready(used_history,raw,required,config["model"].get("max_history_seconds",2.))
            row={"model_cuda_ms":model_start.elapsed_time(model_end),
                 "model_wall_ms":(after_model-after_history)*1000,
                 "transfer_cuda_ms":transfer_start.elapsed_time(transfer_end),
                 "transfer_wall_ms":(after_transfer-begin)*1000,
                 "pose_history_handling_ms":(after_history-after_transfer+end-after_tracking)*1000,
                 "detection_nms_ms":(after_nms-after_model)*1000,"road_decode_ms":(after_road-after_nms)*1000,
                 "tracking_ms":(after_tracking-after_road)*1000,"combined_wall_ms":(end-begin)*1000}
            if first_update is None:first_update=row
            if not eligible:
                exclusions[reason]+=1;reset_timings.append(row["combined_wall_ms"]);continue
            if warmed<args.warmup:
                warmed+=1;warmup_indices.append(sample_index);continue
            for key,value in row.items():timings[key].append(value)
            collected+=1;timed_indices.append(sample_index)
        repeats.append({"repeat":rep,"timings":{k:_summary(v) for k,v in timings.items()},
                        "warmup_full_history_updates":warmed,"timed_full_history_updates":collected,"all_attempts":attempt,
                        "excluded_counts":dict(exclusions),"reset_or_incomplete_history_combined":_summary(reset_timings),
                        "warmup_workload":replay_order_summary(warmup_indices),"timed_workload":replay_order_summary(timed_indices),
                        "peak_allocated_mb":torch.cuda.max_memory_allocated()/2**20,
                        "peak_reserved_mb":torch.cuda.max_memory_reserved()/2**20})
        print(json.dumps({"event":"benchmark_repeat",**repeats[-1]}),flush=True)
    return {"first_cold_update":first_update,"required_history_frames":required,"repeats":repeats}


class StageHooks:
    """CUDA events used ONLY in a separate diagnostic pass, never total timing."""
    names={"image_encoder":"image_encoder_including_fpn","image_encoder.fpn":"fpn_nested",
           "depth_head":"depth_head","context_head":"context_head","lift":"depth_lifting",
           "bev_encoder":"bev_encoder","road_head":"road_head","center_head":"center_head",
           "geometry_head":"geometry_head","refiner":"spatial_temporal_refiner"}
    def __init__(self,model):
        self.handles=[];self.events={}
        for path,label in self.names.items():
            module=model.get_submodule(path)
            def before(m,a,label=label):
                start=torch.cuda.Event(enable_timing=True);start.record();self.events[label]=[start,None]
            def after(m,a,out,label=label):
                end=torch.cuda.Event(enable_timing=True);end.record();self.events[label][1]=end
            self.handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    def reset(self):self.events={}
    def values(self):return {k:start.elapsed_time(end) for k,(start,end) in self.events.items() if end is not None}
    def close(self):
        for handle in self.handles:handle.remove()


def stage_profile(model,samples,config,refine,args):
    required=_history_requirement(config,refine);history=[];values=defaultdict(list)
    # Warm up eager modules before installing instrumenting hooks.
    for i in range(max(10,required+1)):
        raw=samples[i%len(samples)];batch=to_device(raw,"cuda");history=history_for(history,batch)
        out=model_call(model,batch,history,refine,args.fp32)
        history=(history+[out["state"]])[-max(1,required):] if required else []
    torch.cuda.synchronize();hooks=StageHooks(model);collected=0;attempts=0
    try:
        while collected<args.profile_updates:
            if attempts>(args.profile_updates+len(samples))*12:raise RuntimeError("Insufficient continuous history for profiler")
            raw=samples[attempts%len(samples)];attempts+=1;batch=to_device(raw,"cuda");history=history_for(history,batch)
            eligible,_=history_ready(history,raw,required,config["model"].get("max_history_seconds",2.))
            hooks.reset();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();out=model_call(model,batch,history,refine,args.fp32);end.record();torch.cuda.synchronize()
            if eligible:
                for key,value in hooks.values().items():values[key].append(value)
                values["instrumented_model_total"].append(start.elapsed_time(end));collected+=1
            history=(history+[out["state"]])[-max(1,required):] if required else []
    finally:hooks.close()
    aggregates={k:_summary(v) for k,v in values.items()}
    return {"updates":collected,"required_history_frames":required,"backend":"eager","timings":aggregates,
            "interpretation":"Diagnostic CUDA-event module timings. FPN is nested inside image_encoder and must not be added twice. Initial proposal decoding and bookkeeping are outside module sums. Hooked timings are not the published uninstrumented total."}


def compare_detection_sets(reference,candidate):
    stats={"reference_count":len(reference["boxes"]),"candidate_count":len(candidate["boxes"]),
           "matched":0,"unmatched_reference":0,"unmatched_candidate":0,"center_errors_m":[],
           "dimension_errors_m":[],"yaw_errors_deg":[],"confidence_errors":[]}
    for cls in (0,1):
        ri=np.flatnonzero(reference["labels"]==cls);ci=np.flatnonzero(candidate["labels"]==cls)
        if len(ri) and len(ci):
            overlap=pairwise_iou(reference["boxes"][ri],candidate["boxes"][ci],mode="bev")
            rows,cols=linear_sum_assignment(1-overlap)
            selected=[(int(ri[r]),int(ci[c])) for r,c in zip(rows,cols) if overlap[r,c]>=.5]
        else:selected=[]
        stats["matched"]+=len(selected)
        stats["unmatched_reference"]+=len(ri)-len(selected);stats["unmatched_candidate"]+=len(ci)-len(selected)
        for r,c in selected:
            rb,cb=reference["boxes"][r],candidate["boxes"][c]
            stats["center_errors_m"].append(float(np.linalg.norm(rb[:3]-cb[:3])))
            stats["dimension_errors_m"].append(float(np.mean(abs(rb[3:6]-cb[3:6]))))
            stats["yaw_errors_deg"].append(float(abs(wrap_angle(rb[6]-cb[6]))*180/np.pi))
            stats["confidence_errors"].append(float(abs(reference["scores"][r]-candidate["scores"][c])))
    return stats


def prediction_parity(candidate,samples,config,checkpoint,refine,args):
    reference=SceneModel(config).cuda().eval();reference.load_state_dict(checkpoint["model"])
    hr=[];hc=[];road_agreement=[];road_probability_errors=[];objects=Counter();errors=defaultdict(list)
    for raw in samples[:args.parity_updates]:
        batch=to_device(raw,"cuda");hr=history_for(hr,batch);hc=history_for(hc,batch)
        expected=model_call(reference,batch,hr,refine,True)
        observed=model_call(candidate,batch,hc,refine,args.fp32)
        rp=expected["road_logits"].float().softmax(1);cp=observed["road_logits"].float().softmax(1)
        road_agreement.append(float((rp.argmax(1)==cp.argmax(1)).float().mean()))
        road_probability_errors.append(float((rp-cp).abs().mean()))
        matched=compare_detection_sets(select_detections(expected,threshold=args.threshold,refined=refine),
                                       select_detections(observed,threshold=args.threshold,refined=refine))
        for k,v in matched.items():
            if isinstance(v,list):errors[k].extend(v)
            else:objects[k]+=v
        length=int(config["model"].get("history_length",3))
        hr=(hr+[expected["state"]])[-max(1,length):];hc=(hc+[observed["state"]])[-max(1,length):]
    del reference;torch.cuda.empty_cache()
    return {"frames":len(road_agreement),"reference":"same checkpoint, eager FP32, chronological independent memory",
            "candidate_precision":"fp32" if args.fp32 else "fp16","candidate_compiled":args.compile,
            "road_argmax_agreement":float(np.mean(road_agreement)),"road_probability_mean_abs_error":float(np.mean(road_probability_errors)),
            "objects":dict(objects),"matched_geometry_errors":{k:{"mean":float(np.mean(v)) if v else None,"max":max(v) if v else None} for k,v in errors.items()},
            "matching":"Class-consistent Hungarian matching with BEV IoU>=0.5; proposal array order is irrelevant.",
            "limitations":"A bounded prediction parity diagnostic, not ground-truth accuracy or an optimization acceptance gate. Uses the whole BEV for road agreement, including unknown ground. Empty outputs cannot establish object accuracy. Full held-out validation must still enforce the approved accuracy tolerances."}


def resolve_backend(config,fp32_requested=False,compile_requested=False):
    runtime=config.get("runtime",{})
    return (bool(fp32_requested or not runtime.get("mixed_precision",True)),
            bool(compile_requested or runtime.get("compiled",False)))


def configure_inference_determinism(config):
    """Match the evaluator/export default, and bind the resolved choice to config."""
    enabled=bool(config.setdefault("runtime",{}).get("deterministic_inference",True))
    config["runtime"]["deterministic_inference"]=enabled
    torch.use_deterministic_algorithms(enabled)
    return enabled


def verify_input_identity(checkpoint,config,frozen_inputs,script_path,script_sha256):
    after=identity(checkpoint,config)
    changed=[key for key,value in frozen_inputs.items() if after.get(key)!=value]
    if sha256(script_path)!=script_sha256:changed.append("benchmark_script_sha256")
    if changed:
        raise RuntimeError(f"Benchmark inputs changed during run ({changed}); timings were not published")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    p.add_argument("--warmup",type=int,default=50);p.add_argument("--updates",type=int,default=500)
    p.add_argument("--repeats",type=int,default=3);p.add_argument("--cache-frames",type=int,default=32)
    p.add_argument("--fp32",action="store_true");p.add_argument("--compile",action="store_true")
    p.add_argument("--profile",action="store_true");p.add_argument("--profile-only",action="store_true")
    p.add_argument("--profile-updates",type=int,default=30)
    p.add_argument("--parity",action="store_true");p.add_argument("--parity-updates",type=int,default=16)
    p.add_argument("--threshold",type=float,default=.1);p.add_argument("--split",choices=["train","val"],default="val")
    args=p.parse_args()
    if min(args.warmup,args.updates,args.repeats,args.profile_updates,args.parity_updates)<1:raise ValueError("All iteration counts must be positive")
    requested_arguments=vars(args).copy()
    checkpoint_before_load=sha256(args.checkpoint)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    checkpoint_configuration_sha256=hashlib.sha256(json.dumps(checkpoint["config"],sort_keys=True).encode()).hexdigest()
    config=copy.deepcopy(checkpoint["config"]);config["model"]["pretrained"]=False;config["train"]["workers"]=0
    configured_runtime=config.setdefault("runtime",{})
    args.fp32,args.compile=resolve_backend(config,args.fp32,args.compile)
    configured_runtime.update(device="cuda",mixed_precision=not args.fp32,compiled=args.compile)
    configure_inference_determinism(config)
    frozen_inputs=identity(args.checkpoint,config)
    if frozen_inputs["checkpoint_sha256"]!=checkpoint_before_load:
        raise RuntimeError("Checkpoint changed while loading; benchmark aborted")
    script_path=Path(__file__).resolve();script_sha256=sha256(script_path)
    model=SceneModel(config).cuda().eval();model.load_state_dict(checkpoint["model"])
    samples,loading=load_sensor_samples(config,args.cache_frames,args.split)
    workload=sensor_workload(samples)
    refine=checkpoint["stage"]=="b"
    required=_history_requirement(config,refine)
    if required:
        windows=0
        for i in range(required,len(samples)):
            window=samples[i-required:i+1]
            times=[float(item["timestamp"][0]) for item in window]
            contiguous=all(0<b-a<=1.01 for a,b in zip(times,times[1:]))
            same=all(item["sequence"]==window[0]["sequence"] for item in window)
            windows+=int(contiguous and same and times[-1]-times[0]<=config["model"].get("max_history_seconds",2.))
        if not windows:raise RuntimeError("Cached replay has no full-history contiguous window; increase --cache-frames or fix the prepared sequence sampling.")
    candidate=torch.compile(model) if args.compile else model
    report={"checkpoint":str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256":frozen_inputs["checkpoint_sha256"],
            "dataset_manifest":str((Path(config["data"]["root"])/config["data"].get("manifest","manifest.json")).resolve()),
            "dataset_manifest_sha256":frozen_inputs["manifest_sha256"],
            "source_sha256":frozen_inputs["source_sha256"],"configuration_sha256":frozen_inputs["config_sha256"],
            "checkpoint_configuration_sha256":checkpoint_configuration_sha256,
            "configuration":config,"benchmark_script_sha256":script_sha256,
            "invocation":{"argv":[sys.executable,*sys.argv],"cwd":os.getcwd(),
                "requested_arguments":requested_arguments,"resolved_arguments":vars(args)},
            "runtime_backend":{"precision":"fp32" if args.fp32 else "fp16","compiled":args.compile,
                "stage":checkpoint["stage"],"refinement_enabled":refine},
            "score_protocol":{"metric_confidence_cutoff":args.threshold,"nms_bev_iou":.5,"max_detections":100},
            "runtime_choices":{"device":"cuda","autocast_enabled":not args.fp32,
                "autocast_dtype":"float16" if not args.fp32 else None,"parameter_dtype":"float32",
                "compiled":args.compile,"cudnn_benchmark":torch.backends.cudnn.benchmark,
                "cudnn_deterministic":torch.backends.cudnn.deterministic,
                "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision":torch.get_float32_matmul_precision(),
                "cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version()},
            "precision":"fp32" if args.fp32 else "fp16","compiled":args.compile,
            "stage":checkpoint["stage"],"refinement_enabled":refine,"device":torch.cuda.get_device_name(),"torch":torch.__version__,
            "benchmark_purpose":"training-fit profiler pilot; not final performance" if args.split=="train" else "held-out validation replay timing",
            "batch_scenes":1,"cameras":int(samples[0]['images'].shape[1]),"image_size":config["model"]["image_size"],"split":args.split,
            "cpu_decode_preprocess":loading,"input_keys":sorted(INFERENCE_KEYS),"sensor_workload":workload,
            "scope":"Cached-image replay model, inference-only H2D transfer, supplied-pose/history handling, road decoding, score filtering/NMS and tracking. Excludes native capture/resize, pose estimation, road mesh construction, rendering/display and queues. Ego-motion alignment inside the refiner is included in model time."}
    with torch.inference_mode():
        if not args.profile_only:report["steady_state"]=timed_replay(candidate,samples,config,refine,args)
        if args.profile or args.profile_only:report["stage_profile"]=stage_profile(model,samples,config,refine,args)
        if args.parity:report["prediction_parity"]=prediction_parity(candidate,samples,config,checkpoint,refine,args)
    verify_input_identity(args.checkpoint,config,frozen_inputs,script_path,script_sha256)
    report["input_identity_verified_unchanged"]=True
    atomic_json(args.output,report)
    print(json.dumps({"output":args.output,"stage":checkpoint["stage"],"profile":bool(args.profile or args.profile_only)}),flush=True)


if __name__=="__main__":main()
