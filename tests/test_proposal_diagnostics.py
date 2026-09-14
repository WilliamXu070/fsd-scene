"""Controlled positive/negative tests for downstream proposal diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from fsd.inference import select_detections

spec=importlib.util.spec_from_file_location("proposal_diagnostics",Path(__file__).resolve().parents[1]/"scripts/proposal_diagnostics.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def box(x=0,y=0,length=4,width=2):return [x,y,1,length,width,2,0]
def ds(boxes,labels,scores=None):return m.detections(np.array(boxes).reshape(-1,7),labels,scores)

def test_missing_class_is_separate_from_geometry_failure():
    candidates=ds([box()],[0],[.9])
    frame,_=m.diagnose_frame(candidates,candidates,[box(),box(10,length=1,width=1)],[0,1])
    car=frame["stages"]["initial"]["car"];person=frame["stages"]["initial"]["pedestrian"]
    assert car["bev_iou_unique_matched"]==1 and not car["class_starvation"]
    assert person["class_starvation"] and person["nearest_center_m"]==[None]
    assert person["near_center_but_geometry_miss"]==0
    tiny=ds([box(length=.2,width=.2)],[0],[.9])
    frame,_=m.diagnose_frame(tiny,tiny,[box()],[0])
    assert frame["stages"]["initial"]["car"]["near_center_but_geometry_miss"]==1
    assert not frame["stages"]["initial"]["car"]["class_starvation"]

def test_score_filter_is_identified_without_changing_geometry():
    candidates=ds([box()],[0],[.001])
    frame,final=m.diagnose_frame(candidates,candidates,[box()],[0],threshold=.1)
    assert frame["stages"]["refined"]["car"]["bev_iou_unique_matched"]==1
    assert frame["transitions"]["score_filter"]["car"]["lost_gt_coverage"]==1
    assert not len(final["boxes"])

def test_nms_and_top_cap_losses_are_separate():
    candidates=ds([box(0),box(1)],[0,0],[.9,.8])
    frame,_=m.diagnose_frame(candidates,candidates,[box(0),box(2)],[0,0],threshold=.1)
    assert frame["transitions"]["nms"]["car"]["lost_gt_coverage"]==1
    assert frame["transitions"]["nms"]["car"]["unique_match_count_change"]==-1
    distant=ds([box(0),box(10,length=1,width=1)],[0,1],[.9,.8])
    frame,final=m.diagnose_frame(distant,distant,distant["boxes"],[0,1],threshold=.1,max_objects=1)
    assert frame["transitions"]["nms"]["pedestrian"]["lost_gt_coverage"]==0
    assert frame["transitions"]["top_k"]["pedestrian"]["lost_gt_coverage"]==1
    assert final["labels"].tolist()==[0]

def test_refinement_gain_and_one_to_one_cardinality():
    initial=ds([box(5)],[0],[.9]);refined=ds([box()],[0],[.9])
    frame,_=m.diagnose_frame(initial,refined,[box()],[0])
    assert frame["transitions"]["refinement"]["car"]["gained_gt_coverage"]==1
    assert m.matched_targets(np.array([[.99,.6],[.59,.49]]),.5).sum()==2

def test_trace_filter_matches_actual_decoder_even_with_invalid_size():
    boxes=torch.tensor([[box(0),box(1),box(10,length=1,width=1),box(20,length=31)]],dtype=torch.float32)
    logits=torch.tensor([[[3.,-4.],[2.,-5.],[-3.,2.],[4.,-4.]]])
    output={"refined_boxes":boxes,"refined_logits":logits}
    decoded=m.decoded_outputs(output,"refined")
    trace=m.traced_filter(decoded,threshold=.1,max_objects=2)["final"]
    actual=select_detections(output,threshold=.1,max_objects=2)
    for key in actual:assert np.array_equal(trace[key],actual[key])

def test_summary_empty_classes_stays_json_safe_and_sensor_boundary_excludes_gt():
    empty=ds([],[])
    frame,_=m.diagnose_frame(empty,empty,[],[])
    summary=m.Summary();summary.update(frame)
    import json
    json.dumps(summary.result(),allow_nan=False)
    raw={key:object() for key in m.SENSOR_KEYS}
    raw.update(boxes="GT",labels="GT",depth_target="GT",road="GT",instance_ids="GT")
    assert set(m.sensor_only(raw))==m.SENSOR_KEYS
    with pytest.raises(Exception):m.validate_split("test")
    with pytest.raises(Exception):m.validate_split("train")
    assert m.validate_split("val")=="val"


def heatmap_output(points,k=4,height=9,width=9,dtype=torch.float32):
    from fsd.model import decode_proposals
    logits=torch.full((1,2,height,width),-20.,dtype=dtype)
    for cls,y,x,value in points:logits[0,cls,y,x]=value
    proposals=decode_proposals(logits,torch.zeros((1,8,height,width),dtype=dtype),
                               torch.zeros((1,4,height,width),dtype=dtype),
                               {"num_proposals":k,"bev_min":0.,"bev_step":1.})
    return {"center_logits":logits,"proposal_labels":proposals["labels"],"proposal_cells":proposals["cells"]}


def test_heatmap_global_budget_starvation_is_distinct_from_weak_within_class_rank():
    cars=[(0,1,1,8.),(0,1,4,7.),(0,4,1,6.),(0,4,4,5.)]
    output=heatmap_output(cars+[(1,7,7,0.)])
    peaks=m.extract_heatmap_peaks(output)
    result=m.diagnose_heatmap(peaks,[box(7.5,7.5)],[1],{"bev_min":0.,"bev_step":1.},radius_cells=0)
    row=result["ground_truth"][0]
    assert row["diagnosis"]=="quota_rescues_global_budget_miss"
    assert row["global_rank"]==[5,5] and row["same_class_rank"]==[1,1]
    assert row["quota_fits_but_global_excluded"] and not row["poor_within_class_rank"]
    assert output["proposal_labels"].tolist()==[[0,0,0,0]]
    weak=heatmap_output(cars+[(1,1,1,4.),(1,1,4,3.),(1,4,1,2.),(1,7,7,1.)])
    result=m.diagnose_heatmap(m.extract_heatmap_peaks(weak),[box(7.5,7.5)],[1],
                              {"bev_min":0.,"bev_step":1.},radius_cells=0)
    row=result["ground_truth"][0]
    assert row["diagnosis"]=="poor_within_class_rank"
    assert row["global_rank"]==[8,8] and row["same_class_rank"]==[4,4]
    assert row["poor_within_class_rank"] and not row["quota_fits_but_global_excluded"]


def test_heatmap_selected_peak_and_absolute_weak_score_are_disclosed_independently():
    output=heatmap_output([(0,1,1,8.),(0,1,4,7.),(0,4,1,6.),(1,7,7,-10.)])
    result=m.diagnose_heatmap(m.extract_heatmap_peaks(output),[box(7.5,7.5)],[1],
                              {"bev_min":0.,"bev_step":1.},radius_cells=0)
    row=result["ground_truth"][0]
    assert row["diagnosis"]=="selected_in_global_proposals"
    assert row["selected_same_class_peak_in_neighborhood"] and row["below_score_cutoff"]
    assert row["peak_score"]<.01 and row["same_class_rank"]==[1,1]


def test_heatmap_preserves_actual_fp16_equal_peak_plateaus_and_rank_intervals():
    output=heatmap_output([(cls,y,x,0.) for cls in range(2) for y in range(3) for x in range(3)],
                           k=4,height=3,width=3,dtype=torch.float16)
    peaks=m.extract_heatmap_peaks(output)
    assert peaks["local_maxima_count"]=={"car":9,"pedestrian":9}
    assert peaks["selected_peak_slots"]==4 and peaks["score_dtype"]=="torch.float16"
    cls,y,x=np.argwhere(~peaks["selected"])[0]
    args=(peaks,[box(x+.5,y+.5)],[int(cls)],{"bev_min":0.,"bev_step":1.})
    row=m.diagnose_heatmap(*args,radius_cells=0,class_quota=2)["ground_truth"][0]
    assert row["global_rank"]==[1,18] and row["same_class_rank"]==[1,9]
    assert row["diagnosis"]=="class_quota_tie_ambiguous"
    row=m.diagnose_heatmap(*args,radius_cells=0,class_quota=9)["ground_truth"][0]
    assert row["diagnosis"]=="quota_fits_global_tie_miss"
    assert row["quota_fits_but_global_excluded"] and row["global_rank_tie"]


def test_heatmap_no_local_peak_outside_bev_and_optional_neighborhood_are_separate():
    output=heatmap_output([(1,2,3,8.)])
    peaks=m.extract_heatmap_peaks(output)
    rows=m.diagnose_heatmap(peaks,[box(2.5,2.5),box(-1.,2.5)],[1,1],
                            {"bev_min":0.,"bev_step":1.},radius_cells=0)["ground_truth"]
    assert rows[0]["diagnosis"]=="no_local_peak_in_neighborhood"
    assert rows[0]["neighborhood_max_score"] is not None and rows[0]["peak_score"] is None
    assert rows[1]["diagnosis"]=="gt_outside_bev"
    row=m.diagnose_heatmap(peaks,[box(2.5,2.5)],[1],{"bev_min":0.,"bev_step":1.},
                            radius_cells=1)["ground_truth"][0]
    assert row["diagnosis"]=="selected_in_global_proposals" and row["peak_cell_xy"]==[3,2]


def test_heatmap_summary_is_json_safe_and_peak_extraction_has_no_gt_dependency():
    import json
    output=heatmap_output([(0,1,1,8.),(0,1,4,7.),(0,4,1,6.),(0,4,4,5.),(1,7,7,0.)])
    class Guard(dict):
        def __getitem__(self,key):
            assert key in {"center_logits","proposal_labels","proposal_cells"}
            return super().__getitem__(key)
    peaks=m.extract_heatmap_peaks(Guard(output))
    snapshot={key:value.clone() for key,value in output.items()}
    summary=m.HeatmapSummary()
    summary.update(m.diagnose_heatmap(peaks,[],[],{"bev_min":0.,"bev_step":1.},radius_cells=0))
    summary.update(m.diagnose_heatmap(peaks,[box(7.5,7.5)],[1],{"bev_min":0.,"bev_step":1.},radius_cells=0))
    result=summary.result();json.dumps(result,allow_nan=False)
    assert result["classes"]["pedestrian"]["counts"]["quota_rescues_global_budget_miss"]==1
    assert result["classes"]["pedestrian"]["distributions"]["global_rank_lower"]["median"]==5
    assert all(torch.equal(snapshot[key],output[key]) for key in output)
    broken=m.extract_heatmap_peaks(output);broken["selected"][:]=False
    with pytest.raises(AssertionError):
        m.diagnose_heatmap(broken,[box(1.5,1.5)],[0],{"bev_min":0.,"bev_step":1.},radius_cells=0)


def test_fp16_sigmoid_rounding_is_not_silently_promoted_for_peak_ranking():
    output=heatmap_output([(0,1,1,10.),(1,7,7,11.)],k=1,dtype=torch.float16)
    peaks=m.extract_heatmap_peaks(output)
    # Both logits saturate to the same half-precision score, while converting
    # the logits to float32 before sigmoid would produce two distinct scores.
    assert peaks["scores"][0,1,1]==peaks["scores"][1,7,7]==1.
    assert output["center_logits"].float().sigmoid()[0,0,1,1] < output["center_logits"].float().sigmoid()[0,1,7,7]
    rows=m.diagnose_heatmap(peaks,[box(1.5,1.5),box(7.5,7.5)],[0,1],
                            {"bev_min":0.,"bev_step":1.},radius_cells=0,class_quota=1)["ground_truth"]
    assert all(row["global_rank"]==[1,2] and row["same_class_rank"]==[1,1] for row in rows)
    assert sorted(row["diagnosis"] for row in rows)==["quota_fits_global_tie_miss","selected_in_global_proposals"]
