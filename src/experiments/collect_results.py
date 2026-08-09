"""Collect C8-C matrix run metadata into a stable all_runs.csv."""
from __future__ import annotations
import argparse, csv, json, math, os, sys, tempfile
from pathlib import Path
from typing import Any, Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from experiments.run_matrix import _config_id

FIELDS = [
 "dataset","config","config_path","config_id","seed","status","artifact_dir","git_commit","model_variant","backbone","dataset_view","experiment","failure_type","failure_message","failure_path","collection_error",
 "TP","FP","TN","FN","Precision","Recall","F1","MCC","AUPRC","AUROC","FPR","FP_per_million","Attack_Detection_Rate",
 "parameter_count","trainable_parameter_count","train_seconds_per_epoch_mean","train_seconds_per_epoch_std","num_trained_epochs","train_seconds_per_epoch_json","total_train_seconds","test_seconds","train_events_per_second","test_events_per_second","train_peak_gpu_memory_mb","test_peak_gpu_memory_mb","train_peak_cpu_memory_mb","test_peak_cpu_memory_mb","peak_gpu_memory_mb","peak_cpu_memory_mb",
]
METRIC_MAP = {"tp":"TP","fp":"FP","tn":"TN","fn":"FN","precision":"Precision","recall":"Recall","f1":"F1","mcc":"MCC","auprc":"AUPRC","auroc":"AUROC","fpr":"FPR","fppermillion":"FP_per_million","attackdetectionrate":"Attack_Detection_Rate"}
def norm(s: object) -> str: return "".join(c for c in str(s).lower() if c.isalnum())
def nan() -> float: return float("nan")
def empty_row() -> dict: return {key: nan() for key in FIELDS}
def read_json(path: Path) -> tuple[dict|None,str|None]:
 try:
  with path.open(encoding="utf-8") as f: value=json.load(f)
 except FileNotFoundError: return None, "missing"
 except (OSError,json.JSONDecodeError) as exc: return None, f"invalid JSON ({path.name}): {exc}"
 return (value,None) if isinstance(value,dict) else (None,f"invalid JSON object ({path.name})")
def read_yaml(path: Path) -> tuple[dict|None,str|None]:
 try:
  import yaml
  with path.open(encoding="utf-8") as f: value=yaml.safe_load(f)
 except FileNotFoundError: return None,"missing"
 except Exception as exc: return None,f"invalid YAML ({path.name}): {exc}"
 return (value,None) if isinstance(value,dict) else (None,f"invalid YAML object ({path.name})")
def get(d: dict, *keys: str):
 value: Any=d
 for key in keys:
  if not isinstance(value,dict): return None
  value=value.get(key)
 return value
def num(value: Any) -> float:
 if value is None: return nan()
 try: return float(value)
 except (TypeError,ValueError): return nan()
def find_run_dir(scoped: Path, dataset: str, seed: int) -> Path|None:
 candidates=sorted((scoped/dataset/"runs").glob(f"*/seed_{seed}"))
 return candidates[0] if len(candidates)==1 else None
def marker_paths(root: Path): return sorted((root/"results"/"run_status").glob("**/run_status.json"))
def collect(artifact_root: Path) -> list[dict]:
 rows=[]; seen={}
 for marker in marker_paths(artifact_root):
  status, error=read_json(marker)
  if not status:
   continue
  dataset,status_name,config_path,seed=status.get("dataset"),status.get("status"),status.get("config"),status.get("seed")
  if not isinstance(dataset,str) or not isinstance(config_path,str) or not isinstance(seed,int):
   continue
  config=Path(config_path).expanduser().resolve(); identity=(dataset,str(config),seed)
  if identity in seen:
   previous=seen[identity]; previous["status"]="incomplete"; previous["collection_error"]=(previous.get("collection_error","") + "; duplicate_identity").strip("; ")
   continue
  row=empty_row(); row.update({"dataset":dataset,"config":config.stem,"config_path":str(config),"config_id":_config_id(config),"seed":seed,"status":status_name,"artifact_dir":"","git_commit":"","model_variant":"","backbone":"","dataset_view":"","experiment":config.stem,"failure_type":"","failure_message":"","failure_path":"","collection_error":""})
  scoped=Path(status.get("artifact_root", "")) if isinstance(status.get("artifact_root"),str) else None
  run_dir=find_run_dir(scoped,dataset,seed) if scoped else None
  if run_dir: row["artifact_dir"]=str(run_dir)
  failure_path=status.get("failure_path") or str(marker.with_name("failure.json"))
  if status_name=="failed":
   failure,ferr=read_json(Path(failure_path)); row["failure_path"]=str(failure_path)
   if failure: row["failure_type"]=failure.get("exception_type",""); row["failure_message"]=failure.get("exception_message",""); row["git_commit"]=failure.get("git_commit","") or ""
   elif ferr!="missing": row["collection_error"]=ferr
  if status_name=="completed":
   if not run_dir:
    row["status"]="incomplete"; row["collection_error"]="completed marker has no unique run directory"
   else:
    env,eerr=read_json(run_dir/"environment.json"); runtime,rerr=read_json(run_dir/"runtime.json"); cfg,cerr=read_yaml(run_dir/"config_resolved.yml"); metrics,merr=read_json(run_dir/"node_scores"/"metrics.json")
    errors=[e for e in (eerr if eerr not in (None,"missing") else None,cerr if cerr not in (None,"missing") else None) if e]
    if env: row["git_commit"]=env.get("git_commit","") or ""; row["model_variant"]=env.get("model","") or ""; row["experiment"]=env.get("experiment_name") or row["experiment"]
    if cfg:
     row["model_variant"]=get(cfg,"model","variant") or row["model_variant"]; row["backbone"]=get(cfg,"detection","gnn_training","encoder","backbone") or ""; row["dataset_view"]=get(cfg,"dataset_view","mode") or ""
    if runtime:
     train=get(runtime,"training") or {}; test=get(runtime,"testing") or {}; model=get(runtime,"model") or {}; epochs=train.get("train_seconds_per_epoch")
     row.update({"parameter_count":num(model.get("parameter_count")),"trainable_parameter_count":num(model.get("trainable_parameter_count")),"total_train_seconds":num(train.get("total_train_seconds")),"test_seconds":num(test.get("test_seconds")),"train_events_per_second":num(train.get("events_per_second")),"test_events_per_second":num(test.get("events_per_second")),"train_peak_gpu_memory_mb":num(train.get("peak_gpu_memory_mb")),"test_peak_gpu_memory_mb":num(test.get("peak_gpu_memory_mb")),"train_peak_cpu_memory_mb":num(train.get("peak_cpu_memory_mb")),"test_peak_cpu_memory_mb":num(test.get("peak_cpu_memory_mb"))})
     if isinstance(epochs,list):
      values=[num(x) for x in epochs]; finite=[x for x in values if math.isfinite(x)]; row["num_trained_epochs"]=len(values); row["train_seconds_per_epoch_json"]=json.dumps(epochs); row["train_seconds_per_epoch_mean"]=sum(finite)/len(finite) if finite else nan(); row["train_seconds_per_epoch_std"]=math.sqrt(sum((x-row["train_seconds_per_epoch_mean"])**2 for x in finite)/(len(finite)-1)) if len(finite)>1 else nan()
    elif rerr=="missing": errors.append("runtime.json missing")
    elif rerr: errors.append(rerr)
    row["peak_gpu_memory_mb"]=max((x for x in (row["train_peak_gpu_memory_mb"],row["test_peak_gpu_memory_mb"]) if math.isfinite(x)),default=nan()); row["peak_cpu_memory_mb"]=max((x for x in (row["train_peak_cpu_memory_mb"],row["test_peak_cpu_memory_mb"]) if math.isfinite(x)),default=nan())
    if metrics:
     for key,value in metrics.items():
      target=METRIC_MAP.get(norm(key))
      if target: row[target]=num(value)
    else:
     row["status"]="incomplete"; errors.append("metrics.json missing" if merr=="missing" else merr or "metrics.json invalid")
    row["collection_error"]="; ".join(errors)
  seen[identity]=row; rows.append(row)
 return sorted(rows,key=lambda r:(r["dataset"],r["config"],int(r["seed"])))
def write_csv(path: Path, rows: list[dict]) -> None:
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="",dir=path.parent,delete=False) as f:
  writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows); temp=Path(f.name)
 os.replace(temp,path)
def main(argv: Sequence[str]|None=None) -> Path:
 parser=argparse.ArgumentParser(description="Collect C8 matrix results into all_runs.csv."); parser.add_argument("--artifact-root",required=True); parser.add_argument("--output",default=None); args=parser.parse_args(argv)
 root=Path(args.artifact_root).expanduser().resolve(); output=Path(args.output).expanduser().resolve() if args.output else root/"results"/"all_runs.csv"; write_csv(output,collect(root)); return output
if __name__=="__main__": print(main())
