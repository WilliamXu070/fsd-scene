"""Validation-only candidate/geometry/filtering diagnosis, never GT-seeded inference."""
from __future__ import annotations
import argparse,copy,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from fsd.metrics import pairwise_iou
from fsd.inference import select_detections
from fsd.runtime import atomic_json,append_jsonl,history_for,to_device
from fsd.sealing import identity,sha256

SENSOR_KEYS={"images","camera_valid","camera_to_ego","intrinsics","distortion","xi","camera_model",
             "rays","ray_valid","ego_to_world","timestamp","sequence","frame_id"}
CLASS_NAMES={0:"car",1:"pedestrian"}


def sensor_only(raw):
    missing=SENSOR_KEYS-set(raw)
    if missing:raise KeyError(f"Missing sensor fields: {sorted(missing)}")
    return {k:raw[k] for k in SENSOR_KEYS}


def validate_split(value):
    if value!="val":raise argparse.ArgumentTypeError("Proposal diagnostics are validation-only; test and training splits are rejected")
    return value


def _array(value,dtype=float):
    if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
    return np.asarray(value,dtype=dtype)


def detections(boxes,labels,scores=None):
    boxes=_array(boxes).reshape(-1,7);labels=_array(labels,np.int64).reshape(-1)
    scores=np.ones(len(boxes)) if scores is None else _array(scores).reshape(-1)
    if not len(boxes)==len(labels)==len(scores):raise ValueError("Prediction array lengths differ")
    if ((labels<0)|(labels>1)).any():raise ValueError("Only car/pedestrian prediction labels are supported")
    return {"boxes":boxes,"labels":labels,"scores":scores}


def subset(dets,indices):return {key:value[indices] for key,value in dets.items()}


def decoded_outputs(outputs,prefix):
    boxes=outputs[prefix+"_boxes"][0].detach().float().cpu().numpy()
    probability=outputs[prefix+"_logits"][0].detach().float().sigmoid().cpu().numpy()
    return detections(boxes,probability.argmax(-1),probability.max(-1))


def extract_heatmap_peaks(outputs):
    """Target-independent copy of decode_proposals' exact peak convention.

    Keep sigmoid/max-pooling on the source device and in the source dtype so
    FP16 quantization, including equal-score plateaus, matches actual decoding.
    Actual emitted cells determine top-K membership; topk's tie order is not
    reproduced or assumed stable.
    """
    logits=outputs["center_logits"].detach()
    if logits.ndim!=4 or logits.shape[0]!=1 or logits.shape[1]!=2:
        raise ValueError("Heatmap diagnosis requires one scene with two classes")
    if not torch.isfinite(logits).all():raise ValueError("Non-finite class heatmap")
    probability=logits.sigmoid()
    local=probability==torch.nn.functional.max_pool2d(probability,3,stride=1,padding=1)
    scores=_array(probability[0]);local=_array(local[0],bool)
    labels=_array(outputs["proposal_labels"][0],np.int64)
    cells=_array(outputs["proposal_cells"][0],np.int64)
    _,h,w=scores.shape
    if len(labels)!=len(cells) or ((labels<0)|(labels>=2)).any() or ((cells<0)|(cells>=h*w)).any():
        raise ValueError("Invalid emitted proposal cells/classes")
    selected=np.zeros_like(local)
    selected[labels,cells//w,cells%w]=True
    return {"scores":scores,"local":local,"selected":selected,"global_k":len(cells),
            "local_maxima_count":{name:int(local[c].sum()) for c,name in CLASS_NAMES.items()},
            "selected_peak_slots":int(local[labels,cells//w,cells%w].sum()),
            "non_peak_fallback_slots":int((~local[labels,cells//w,cells%w]).sum()),
            "score_dtype":str(logits.dtype)}


def rank_interval(sorted_scores,score):
    """One-based best/worst ranks across every maximum tied at this score."""
    return [int(len(sorted_scores)-np.searchsorted(sorted_scores,score,side="right")+1),
            int(len(sorted_scores)-np.searchsorted(sorted_scores,score,side="left"))]


def diagnose_heatmap(peaks,gt_boxes,gt_labels,model_config,radius_cells=2,class_quota=None,score_cutoff=.01):
    """Find GT-neighborhood responses after inference; never construct proposals."""
    if radius_cells<0:raise ValueError("Neighborhood radius must be nonnegative")
    quota=max(1,peaks["global_k"]//2) if class_quota is None else int(class_quota)
    if quota<1:raise ValueError("Hypothetical per-class quota must be positive")
    scores,local,selected=peaks["scores"],peaks["local"],peaks["selected"]
    _,h,w=scores.shape
    minimum=float(model_config.get("bev_min",-40.));step=float(model_config.get("bev_step",.5))
    if not np.isfinite(step) or step<=0:raise ValueError("Invalid BEV cell size")
    boxes=_array(gt_boxes).reshape(-1,7);labels=_array(gt_labels,np.int64).reshape(-1)
    if len(boxes)!=len(labels) or not np.isfinite(boxes).all() or ((labels<0)|(labels>1)).any():
        raise ValueError("Invalid GT for downstream heatmap diagnosis")
    global_scores=np.sort(scores[local])
    class_scores={c:np.sort(scores[c][local[c]]) for c in CLASS_NAMES}
    rows=[]
    for index,(box,cls) in enumerate(zip(boxes,labels)):
        x,y=np.floor((box[:2]-minimum)/step).astype(int)
        row={"gt_index":index,"class":CLASS_NAMES[int(cls)],"gt_cell_xy":[int(x),int(y)],
             "peak_score":None,"neighborhood_max_score":None,"global_rank":None,"same_class_rank":None,
             "selected_same_class_peak_in_neighborhood":False,"fits_same_class_quota":False,
             "quota_fits_but_global_excluded":False,"poor_within_class_rank":False,
             "below_score_cutoff":False}
        if not (0<=x<w and 0<=y<h):
            row["diagnosis"]="gt_outside_bev";rows.append(row);continue
        left,right=max(0,x-radius_cells),min(w,x+radius_cells+1)
        top,bottom=max(0,y-radius_cells),min(h,y+radius_cells+1)
        window=scores[cls,top:bottom,left:right]
        row["neighborhood_max_score"]=float(window.max())
        yy,xx=np.where(local[cls,top:bottom,left:right]);yy+=top;xx+=left
        if not len(xx):
            row["diagnosis"]="no_local_peak_in_neighborhood";rows.append(row);continue
        values=scores[cls,yy,xx];best=float(values.max());best_mask=values==best
        best_y,best_x=yy[best_mask],xx[best_mask]
        best_selected=selected[cls,best_y,best_x]
        any_selected=bool(selected[cls,yy,xx].any())
        # Prefer an actually selected tied peak for presentation, then closest
        # cell centre. The complete tie interval remains the reported rank.
        distance=(minimum+(best_x+.5)*step-box[0])**2+(minimum+(best_y+.5)*step-box[1])**2
        choice=int(np.lexsort((distance,~best_selected))[0])
        global_rank=rank_interval(global_scores,best);class_rank=rank_interval(class_scores[cls],best)
        fits=class_rank[1]<=quota;poor=class_rank[0]>quota
        if not any_selected and global_rank[1]<=peaks["global_k"]:
            raise AssertionError("Actual proposal selection disagrees with reconstructed peak ranks")
        if any_selected:diagnosis="selected_in_global_proposals"
        elif fits and global_rank[0]>peaks["global_k"]:diagnosis="quota_rescues_global_budget_miss"
        elif fits:diagnosis="quota_fits_global_tie_miss"
        elif poor:diagnosis="poor_within_class_rank"
        else:diagnosis="class_quota_tie_ambiguous"
        row.update(peak_score=best,peak_cell_xy=[int(best_x[choice]),int(best_y[choice])],
                   global_rank=global_rank,same_class_rank=class_rank,diagnosis=diagnosis,
                   selected_same_class_peak_in_neighborhood=any_selected,
                   highest_score_peak_selected=bool(best_selected.any()),
                   fits_same_class_quota=fits,quota_fits_but_global_excluded=bool(fits and not any_selected),
                   poor_within_class_rank=poor,below_score_cutoff=bool(best<score_cutoff),
                   global_rank_tie=global_rank[0]!=global_rank[1],same_class_rank_tie=class_rank[0]!=class_rank[1])
        rows.append(row)
    return {"global_k":peaks["global_k"],"hypothetical_same_class_quota":quota,
            "neighborhood_radius_cells":radius_cells,"neighborhood_radius_axis_m":radius_cells*step,
            "score_dtype":peaks["score_dtype"],"local_maxima_count":peaks["local_maxima_count"],
            "selected_peak_slots":peaks["selected_peak_slots"],"non_peak_fallback_slots":peaks["non_peak_fallback_slots"],
            "ground_truth":rows}


def distribution(values):
    return {"mean":float(np.mean(values)) if values else None,
            "median":float(np.median(values)) if values else None,
            "p90":float(np.percentile(values,90)) if values else None}


class HeatmapSummary:
    def __init__(self):
        self.frames=0;self.protocol=None;self.counts=defaultdict(lambda:defaultdict(int))
        self.values=defaultdict(lambda:defaultdict(list));self.local_maxima=defaultdict(int)
        self.non_peak_fallback_slots=0
    def update(self,frame):
        self.frames+=1
        protocol={key:frame[key] for key in ("global_k","hypothetical_same_class_quota","neighborhood_radius_cells",
                                             "neighborhood_radius_axis_m","score_dtype")}
        if self.protocol is not None and self.protocol!=protocol:raise ValueError("Heatmap protocol changed between frames")
        self.protocol=protocol;self.non_peak_fallback_slots+=frame["non_peak_fallback_slots"]
        for name,value in frame["local_maxima_count"].items():self.local_maxima[name]+=value
        for row in frame["ground_truth"]:
            count=self.counts[row["class"]];values=self.values[row["class"]]
            count["gt_count"]+=1;count[row["diagnosis"]]+=1
            for key in ("selected_same_class_peak_in_neighborhood","fits_same_class_quota","quota_fits_but_global_excluded",
                        "poor_within_class_rank","below_score_cutoff","global_rank_tie","same_class_rank_tie"):
                count[key+"_count"]+=int(row.get(key,False))
            for key in ("peak_score","neighborhood_max_score"):
                if row[key] is not None:values[key].append(row[key])
            for key in ("global_rank","same_class_rank"):
                if row[key] is not None:
                    values[key+"_lower"].append(row[key][0]);values[key+"_upper"].append(row[key][1])
    def result(self):
        classes={}
        for name in CLASS_NAMES.values():
            count=self.counts[name];values=self.values[name]
            classes[name]={"counts":{"gt_count":count["gt_count"],**dict(count)},"local_maxima_per_frame":self.local_maxima[name]/max(self.frames,1),
                           "distributions":{key:distribution(values[key]) for key in
                           ("peak_score","neighborhood_max_score","global_rank_lower","global_rank_upper","same_class_rank_lower","same_class_rank_upper")}}
        return {"frames":self.frames,"protocol":self.protocol,"non_peak_fallback_slots":self.non_peak_fallback_slots,
                "classes":classes}


def traced_filter(dets,threshold=.01,max_objects=100,nms_threshold=.5):
    """Expose intermediate sets, checked against the production decoder by CLI."""
    boxes,scores,labels=dets["boxes"],dets["scores"],dets["labels"]
    valid=(np.isfinite(boxes).all(-1)&np.isfinite(scores)
           &(boxes[:,3:6]>.05).all(-1)&(boxes[:,3:6]<30).all(-1))
    valid_indices=np.flatnonzero(valid)
    geometry_valid=subset(dets,valid_indices)
    score_indices=np.flatnonzero(valid&(scores>=threshold))
    score_indices=score_indices[np.argsort(-scores[score_indices],kind="stable")]
    after_score=subset(dets,score_indices)
    keep=[]
    for index in score_indices:
        same=[j for j in keep if labels[j]==labels[index]]
        if same and pairwise_iou(boxes[index:index+1],boxes[same],mode="bev").max()>nms_threshold:continue
        keep.append(int(index))
    return {"geometry_valid":geometry_valid,"score_filtered":after_score,
            "nms_uncapped":subset(dets,keep),"final":subset(dets,keep[:max_objects])}


def matched_targets(overlap,threshold):
    """Maximum-cardinality one-to-one IoU matching, then maximum total IoU."""
    matched=np.zeros(overlap.shape[1],dtype=bool)
    if not overlap.size:return matched
    cardinality_weight=min(overlap.shape)+1
    rows,cols=linear_sum_assignment(-((overlap>=threshold)*cardinality_weight+overlap))
    good=overlap[rows,cols]>=threshold;matched[cols[good]]=True
    return matched


def stage_geometry(dets,gt_boxes,gt_labels,center_thresholds=(.5,1.,2.),iou_threshold=.5):
    gt_boxes=_array(gt_boxes).reshape(-1,7);gt_labels=_array(gt_labels,np.int64).reshape(-1)
    if len(gt_boxes)!=len(gt_labels):raise ValueError("Ground-truth box/label lengths differ")
    if not np.isfinite(gt_boxes).all() or (gt_boxes[:,3:6]<=0).any():raise ValueError("Invalid ground-truth geometry")
    if ((gt_labels<0)|(gt_labels>1)).any():raise ValueError("Invalid ground-truth class")
    result={};covered=np.zeros(len(gt_boxes),bool);unique=np.zeros(len(gt_boxes),bool)
    for cls,name in CLASS_NAMES.items():
        pi=np.flatnonzero(dets["labels"]==cls);gi=np.flatnonzero(gt_labels==cls)
        raw=dets["boxes"][pi]
        valid=np.isfinite(raw).all(-1)&(raw[:,3:6]>0).all(-1)
        predictions=raw[valid];truth=gt_boxes[gi]
        nearest=np.full(len(gi),np.inf);best_iou=np.zeros(len(gi));matched=np.zeros(len(gi),bool)
        if len(predictions) and len(truth):
            distances=np.linalg.norm(predictions[:,None,:2]-truth[None,:,:2],axis=-1)
            nearest=distances.min(0)
            overlap=pairwise_iou(predictions,truth,mode="bev")
            best_iou=overlap.max(0);matched=matched_targets(overlap,iou_threshold)
        has_box=best_iou>=iou_threshold
        covered[gi]=has_box;unique[gi]=matched
        result[name]={"candidates":len(pi),"valid_geometry_candidates":len(predictions),"invalid_geometry_candidates":int((~valid).sum()),
                      "gt_count":len(gi),"class_starvation":bool(len(gi)>0 and len(pi)==0),
                      "nearest_center_m":[float(v) if np.isfinite(v) else None for v in nearest],
                      "center_covered":{str(float(t)):int((nearest<=t).sum()) for t in center_thresholds},
                      "bev_iou_covered":int(has_box.sum()),"bev_iou_unique_matched":int(matched.sum()),
                      "near_center_but_geometry_miss":int(((nearest<=1.)&~has_box).sum()),
                      "no_center_within_1m":int((nearest>1.).sum()),
                      "coverage_without_unique_match":int((has_box&~matched).sum())}
    return result,covered,unique


def diagnose_frame(initial,refined,gt_boxes,gt_labels,threshold=.01,max_objects=100,iou_threshold=.5):
    stages={"initial":initial,"refined":refined,**traced_filter(refined,threshold,max_objects)}
    summaries={};covered={};unique={}
    labels=_array(gt_labels,np.int64)
    for name,dets in stages.items():
        summaries[name],covered[name],unique[name]=stage_geometry(dets,gt_boxes,labels,iou_threshold=iou_threshold)
    losses={}
    for name,before,after in (("refinement","initial","refined"),("geometry_filter","refined","geometry_valid"),
                              ("score_filter","geometry_valid","score_filtered"),("nms","score_filtered","nms_uncapped"),
                              ("top_k","nms_uncapped","final")):
        losses[name]={}
        for cls,clsname in CLASS_NAMES.items():
            mask=labels==cls
            losses[name][clsname]={"lost_gt_coverage":int((covered[before]&~covered[after]&mask).sum()),
                                   "gained_gt_coverage":int((~covered[before]&covered[after]&mask).sum()),
                                   "unique_match_count_change":int((unique[after]&mask).sum()-(unique[before]&mask).sum())}
    return {"stages":summaries,"transitions":losses},stages["final"]


class Summary:
    def __init__(self):
        self.frames=0;self.stage=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
        self.distances=defaultdict(lambda:defaultdict(list));self.transitions=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    def update(self,frame):
        self.frames+=1
        for stage,classes in frame["stages"].items():
            for cls,data in classes.items():
                total=self.stage[stage][cls]
                for key in ("candidates","valid_geometry_candidates","invalid_geometry_candidates","gt_count","bev_iou_covered",
                            "bev_iou_unique_matched","near_center_but_geometry_miss","no_center_within_1m","coverage_without_unique_match"):
                    total[key]+=data[key]
                total["class_starvation_frames"]+=int(data["class_starvation"])
                total["frames_with_gt"]+=int(data["gt_count"]>0)
                for radius,count in data["center_covered"].items():total["center_covered_"+radius]+=count
                self.distances[stage][cls].extend(v for v in data["nearest_center_m"] if v is not None)
        for transition,classes in frame["transitions"].items():
            for cls,values in classes.items():
                for key,value in values.items():self.transitions[transition][cls][key]+=value
    def result(self):
        stages={}
        for stage,classes in self.stage.items():
            stages[stage]={}
            for cls,counts in classes.items():
                gt=int(counts["gt_count"]);d=self.distances[stage][cls]
                stages[stage][cls]={"candidate_count_total":int(counts["candidates"]),"candidates_per_frame":counts["candidates"]/max(self.frames,1),
                                   "invalid_geometry_candidates":int(counts["invalid_geometry_candidates"]),"gt_count":gt,
                                   "class_starvation_frames":int(counts["class_starvation_frames"]),"frames_with_gt":int(counts["frames_with_gt"]),
                                   "gt_with_any_valid_same_class_candidate":len(d),
                                   "nearest_center_m":{"mean":float(np.mean(d)) if d else None,"median":float(np.median(d)) if d else None,"p90":float(np.percentile(d,90)) if d else None},
                                   "center_coverage":{"0.5m":counts["center_covered_0.5"]/gt if gt else None,"1.0m":counts["center_covered_1.0"]/gt if gt else None,"2.0m":counts["center_covered_2.0"]/gt if gt else None},
                                   "bev_iou_gt_coverage":counts["bev_iou_covered"]/gt if gt else None,
                                   "bev_iou_one_to_one_recall":counts["bev_iou_unique_matched"]/gt if gt else None,
                                   "near_center_but_geometry_miss":int(counts["near_center_but_geometry_miss"]),
                                   "no_center_within_1m":int(counts["no_center_within_1m"]),
                                   "coverage_without_unique_match":int(counts["coverage_without_unique_match"])}
        return {"frames":self.frames,"stages":stages,"transitions":{t:{c:dict(v) for c,v in classes.items()} for t,classes in self.transitions.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--split",type=validate_split,default="val",help="Only val is permitted")
    parser.add_argument("--per-frame",help="Optional diagnostic JSONL; GT is used downstream only")
    parser.add_argument("--per-gt",action="store_true",help="Include compact per-GT heatmap ranks in --per-frame JSONL")
    parser.add_argument("--peak-radius-cells",type=int,default=2,help="Square neighborhood radius around the GT BEV center cell (analysis only)")
    parser.add_argument("--class-quota",type=int,help="Hypothetical per-class peak budget, default global K divided by two (analysis only)")
    parser.add_argument("--max-frames",type=int,default=0,help="0 means all validation frames")
    parser.add_argument("--threshold",type=float,help="Defaults to checkpoint evaluation metric cutoff, otherwise 0.01")
    parser.add_argument("--iou-threshold",type=float,default=.5)
    parser.add_argument("--max-detections",type=int,default=100)
    parser.add_argument("--fp32",action="store_true");parser.add_argument("--compile",action="store_true")
    args=parser.parse_args()
    if args.max_frames<0 or args.max_detections<1:parser.error("Invalid frame/detection limit")
    if args.peak_radius_cells<0 or (args.class_quota is not None and args.class_quota<1):parser.error("Invalid diagnostic neighborhood/quota")
    if args.per_gt and not args.per_frame:parser.error("--per-gt requires --per-frame")
    if not 0<args.iou_threshold<=1:parser.error("IoU threshold must be in (0,1]")
    if args.per_frame and Path(args.per_frame).exists():parser.error("Choose a fresh per-frame output path")
    before=sha256(args.checkpoint);checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    config=copy.deepcopy(checkpoint["config"]);config["model"]["pretrained"]=False;config["train"]["workers"]=0
    runtime=config.setdefault("runtime",{});runtime["device"]="cuda"
    runtime["mixed_precision"]=bool(runtime.get("mixed_precision",True) and not args.fp32)
    runtime["compiled"]=bool(runtime.get("compiled",False) or args.compile)
    threshold=args.threshold if args.threshold is not None else config.get("evaluation",{}).get("metric_threshold",.01)
    if not 0<=threshold<=1:parser.error("Confidence threshold must be in [0,1]")
    frozen=identity(args.checkpoint,config)
    if frozen["checkpoint_sha256"]!=before:raise RuntimeError("Checkpoint changed while loading")
    from fsd.model import SceneModel
    from fsd.engine import make_loader,amp_context
    model=SceneModel(config).cuda().eval();model.load_state_dict(checkpoint["model"])
    if runtime["compiled"]:model=torch.compile(model)
    history=[];summary=Summary();heatmaps=HeatmapSummary();source_counts=np.zeros(2,dtype=np.int64);label_disagreements=0
    refine=checkpoint["stage"]=="b"
    provenance={"checkpoint_sha256":frozen["checkpoint_sha256"],"dataset_manifest_sha256":frozen["manifest_sha256"],
                "configuration_sha256":frozen["config_sha256"],"source_sha256":frozen["source_sha256"]}
    if args.per_frame:append_jsonl(args.per_frame,{"type":"provenance","complete":False,**provenance})
    with torch.inference_mode():
        for index,raw in enumerate(make_loader(config,"val")):
            if args.max_frames and index>=args.max_frames:break
            sensors=to_device(sensor_only(raw),"cuda");history=history_for(history,sensors)
            with amp_context(config):output=model(sensors,history,refine=refine)
            initial=decoded_outputs(output,"initial");refined=decoded_outputs(output,"refined")
            frame,final=diagnose_frame(initial,refined,raw["boxes"][0],raw["labels"][0],threshold,args.max_detections,args.iou_threshold)
            actual=select_detections(output,threshold=threshold,max_objects=args.max_detections,refined=True)
            if not all(np.array_equal(final[key],actual[key]) for key in ("boxes","labels","scores")):
                raise AssertionError("Diagnostic filtering diverged from production select_detections")
            if "proposal_labels" in output:
                source=_array(output["proposal_labels"][0],np.int64)
                source_counts+=np.bincount(source,minlength=2)
                label_disagreements+=int((source!=initial["labels"]).sum())
            # This analysis starts only after sensor-only inference has finished.
            peaks=extract_heatmap_peaks(output)
            heatmap_frame=diagnose_heatmap(peaks,raw["boxes"][0],raw["labels"][0],config["model"],
                                           args.peak_radius_cells,args.class_quota,threshold)
            heatmaps.update(heatmap_frame);summary.update(frame)
            if args.per_frame:
                heatmap_detail={key:value for key,value in heatmap_frame.items() if key!="ground_truth" or args.per_gt}
                append_jsonl(args.per_frame,{"type":"frame","sequence":raw["sequence"][0],"frame_id":int(raw["frame_id"][0]),
                                            **frame,"heatmap":heatmap_detail})
            length=config["model"].get("history_length",3)
            history=(history+[output["state"]])[-max(1,length):] if refine and length else []
            if index%100==0:print(json.dumps({"event":"proposal_diagnostics","frames":index+1}),flush=True)
    if identity(args.checkpoint,config)!=frozen:raise RuntimeError("Checkpoint, manifest, configuration or source changed during diagnosis")
    report={"split":"val","checkpoint":str(Path(args.checkpoint).resolve()),**provenance,"configuration":config,
            "invocation":[sys.executable,*sys.argv],"runtime_backend":{"precision":"fp16" if runtime["mixed_precision"] else "fp32","compiled":runtime["compiled"],"stage":checkpoint["stage"],"refinement_enabled":refine},
            "configured_initial_proposals":config["model"].get("num_proposals",200),"score_protocol":{"metric_confidence_cutoff":threshold,"nms_bev_iou":.5,"max_detections":args.max_detections},
            "iou_threshold":args.iou_threshold,"summary":summary.result(),
            "heatmap_selection":{"source_class_candidate_counts":dict(zip(CLASS_NAMES.values(),source_counts.tolist())),"source_class_vs_decoded_argmax_disagreements":label_disagreements,"gt_neighborhood_ranking":heatmaps.result()},
            "heatmap_definitions":{"peak_convention":"Original-dtype sigmoid; equality with 3x3 stride-1 max pooling; every plateau cell remains a peak. No diagnostic score threshold is applied before ranking.",
                                   "rank_interval":"One-based [1 + strictly-higher peaks, greater-or-equal peaks]; tie order is unspecified. Global-K membership uses actual emitted proposal cells/classes.",
                                   "neighborhood":"Square region within the stated number of cells of floor((GT XY - BEV minimum)/cell size); GT is used only after inference.",
                                   "quota_rescues_global_budget_miss":"A local GT-neighborhood peak definitely fits the hypothetical same-class quota, ranks strictly outside global K, and was not selected.",
                                   "quota_fits_global_tie_miss":"The peak definitely fits the class quota but lost an actual global top-K tie; reported separately from strict global rank starvation.",
                                   "poor_within_class_rank":"Even the best possible tied rank is beyond the hypothetical class quota. This relative-response diagnostic is distinct from the absolute score cutoff.",
                                   "hypothetical_quota":"Analysis of peak inclusion only; does not change inference and does not establish improved boxes, classification or AP."},
            "definitions":{"class_starvation":"A frame has GT for a class but zero decoded candidates for that class.","geometry_miss":"A same-class center lies within 1 m, but no same-class predicted box reaches the IoU threshold.","transition_loss":"GT best-IoU coverage disappears at this stage; one-to-one matching changes are reported separately.","center_distance":"XY ground-plane distance in metres.","unique_recall":"Maximum-cardinality class-consistent bipartite matching at the stated BEV-IoU threshold; distinct from the score-ordered AP evaluation."},
            "limitations":["No AP or false-positive precision is claimed; candidate counts include unknown regions.","GT is never passed into the network or proposal generation.","Stage A refined outputs are identical to initial outputs because its refinement is disabled.","Nearest-center geometry categories are diagnostic evidence, not proof of a unique root cause.","Heatmap quota results are counterfactual candidate-inclusion evidence, not an accuracy estimate; crowded GT neighborhoods may share peaks.","Heatmap-source class can differ from the decoded two-class argmax; the disagreement count is reported separately."],"input_identity_verified_unchanged":True}
    atomic_json(args.output,report)
    if args.per_frame:append_jsonl(args.per_frame,{"type":"completed","summary_report":str(args.output),**provenance})
    print(json.dumps({"output":args.output,"frames":summary.frames,"stage":checkpoint["stage"]}),flush=True)


if __name__=="__main__":main()
