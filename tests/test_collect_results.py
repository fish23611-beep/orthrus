from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import pytest
SRC_ROOT=Path(__file__).resolve().parents[1]/"src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0,str(SRC_ROOT))
from experiments import collect_results,run_matrix

def config(tmp,name="mstc_full.yml"):
 p=tmp/name;p.write_text("experiment_identity: {semantics_version: temporal_v2}\nmodel: {variant: mstc}\n",encoding="utf-8");return p.resolve()
def make_run(root,cfg,seed,status="completed",metrics=None,runtime=True,dataset="THEIA_E3"):
 scoped=run_matrix.run_artifact_root(root,cfg); marker=run_matrix.run_status_path(root,dataset,cfg,seed); payload=run_matrix._status_payload(dataset,cfg,seed,status,scoped_root=scoped)
 run_matrix._atomic_json(marker,payload); run_dir=scoped/dataset/"runs"/"mstc"/f"seed_{seed}"; run_dir.mkdir(parents=True,exist_ok=True)
 if status=="completed":
  (run_dir/"environment.json").write_text(json.dumps({"git_commit":"abc","model":"mstc"}),encoding="utf-8")
  (run_dir/"config_resolved.yml").write_text("model: {variant: mstc}\ndataset_view: {mode: host_only}\ndetection: {gnn_training: {encoder: {backbone: graphsage}}}\n",encoding="utf-8")
  if runtime: (run_dir/"runtime.json").write_text(json.dumps({"training":{"train_seconds_per_epoch":[1,3],"total_train_seconds":4,"events_per_second":10,"peak_gpu_memory_mb":2,"peak_cpu_memory_mb":5},"testing":{"test_seconds":2,"events_per_second":8,"peak_gpu_memory_mb":3,"peak_cpu_memory_mb":6},"model":{"parameter_count":100,"trainable_parameter_count":90}}),encoding="utf-8")
  node=run_dir/"node_scores";node.mkdir();(node/"metrics.json").write_text(json.dumps(metrics or {"tp":1,"fp":2,"tn":3,"fn":4,"mcc":.8,"f1":.7,"fp_per_million":5,"attack_detection_rate":.9}),encoding="utf-8")
 else:
  failure=marker.with_name("failure.json");failure.write_text(json.dumps({"exception_type":"RuntimeError","exception_message":"boom","git_commit":"abc","traceback":"trace"}),encoding="utf-8"); payload["failure_path"]=str(failure);run_matrix._atomic_json(marker,payload)
 return run_dir,marker

def rows(root):
 path=collect_results.main(["--artifact-root",str(root)]);return list(csv.DictReader(path.open(encoding="utf-8")))
def test_two_completed_seeds_are_collected(tmp_path):
 c=config(tmp_path);make_run(tmp_path,c,0);make_run(tmp_path,c,1)
 result=rows(tmp_path)
 assert len(result)==2 and [r["seed"] for r in result]==["0","1"] and all(r["status"]=="completed" for r in result)
def test_failed_run_keeps_nan_metrics_and_failure_reason(tmp_path):
 c=config(tmp_path);make_run(tmp_path,c,0);make_run(tmp_path,c,1,"failed")
 result=rows(tmp_path); failed=result[1]
 assert failed["failure_message"]=="boom" and all(math.isnan(float(failed[k])) for k in ("MCC","F1","FP","FP_per_million"))
def test_missing_runtime_keeps_metrics_and_reports_error(tmp_path):
 c=config(tmp_path);make_run(tmp_path,c,0,runtime=False)
 row=rows(tmp_path)[0]
 assert row["MCC"]=="0.8" and math.isnan(float(row["total_train_seconds"])) and "runtime.json missing" in row["collection_error"]
def test_corrupt_metrics_does_not_abort_other_run(tmp_path):
 c=config(tmp_path); bad,_=make_run(tmp_path,c,0);make_run(tmp_path,c,1);(bad/"node_scores"/"metrics.json").write_text("{bad",encoding="utf-8")
 result=rows(tmp_path)
 assert result[0]["status"]=="incomplete" and "invalid JSON" in result[0]["collection_error"] and result[1]["status"]=="completed"
def test_repeated_collection_is_stable_and_nan_is_preserved(tmp_path):
 c=config(tmp_path);make_run(tmp_path,c,1,metrics={"mcc":float("nan"),"auroc":float("nan")});make_run(tmp_path,c,0)
 first=(tmp_path/"results"/"all_runs.csv");rows(tmp_path);a=first.read_text(encoding="utf-8");rows(tmp_path);b=first.read_text(encoding="utf-8")
 assert a==b; result=list(csv.DictReader(first.open()))
 assert [r["seed"] for r in result]==["0","1"] and math.isnan(float(result[1]["MCC"]))
def test_completed_without_metrics_becomes_incomplete(tmp_path):
 c=config(tmp_path);run_dir,_=make_run(tmp_path,c,0);(run_dir/"node_scores"/"metrics.json").unlink()
 row=rows(tmp_path)[0]
 assert row["status"]=="incomplete" and "metrics.json missing" in row["collection_error"]
