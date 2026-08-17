"""Aggregate all_runs.csv into stable paper-oriented CSV tables."""
from __future__ import annotations
import argparse,csv,math,os,statistics,sys,tempfile
from pathlib import Path
from typing import Sequence
SRC_ROOT=Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path: sys.path.insert(0,str(SRC_ROOT))
METRIC_DIRECTION={"Precision":"higher","Recall":"higher","F1":"higher","MCC":"higher","AUPRC":"higher","AUROC":"higher","Attack_Detection_Rate":"higher","train_events_per_second":"higher","test_events_per_second":"higher","TP":"higher","TN":"higher","FP":"lower","FN":"lower","FPR":"lower","FP_per_million":"lower","parameter_count":"lower","trainable_parameter_count":"lower","train_seconds_per_epoch_mean":"lower","total_train_seconds":"lower","test_seconds":"lower","train_peak_gpu_memory_mb":"lower","test_peak_gpu_memory_mb":"lower","train_peak_cpu_memory_mb":"lower","test_peak_cpu_memory_mb":"lower","peak_gpu_memory_mb":"lower","peak_cpu_memory_mb":"lower"}
# C8-E authoritative matrix: main, A0--A6 ablations, calibration, and efficiency.
MAIN={"backbone_mlp":"Semantic MLP","backbone_graphsage_baseline":"GraphSAGE","baseline":"ORTHRUS-ano / orthrus_baseline","mstc_full":"MSTC-PIDS Full"}
ABLATION={"baseline":"A0 ORTHRUS","ablation_no_multiscale":"A1 w/o Multi-scale","ablation_no_gate":"A2 w/o Gate","ablation_no_time":"A3 w/o Time","ablation_no_calibration":"A4 w/o Calibration","ablation_no_topk":"A5 w/o Top-k","mstc_full":"A6 Full"}
CALIBRATION={"calibration_max":"calibration_max","calibration_quantile":"calibration_quantile","calibration_kmeans":"calibration_kmeans","calibration_global_p":"calibration_global_p","calibration_relation":"calibration_relation","calibration_hierarchical":"calibration_hierarchical"}
EFFICIENCY={"baseline":"ORTHRUS-ano / orthrus_baseline","efficiency_multiscale":"ORTHRUS + Multi-scale","efficiency_multiscale_time":"ORTHRUS + Multi-scale + Time-task","mstc_full":"MSTC-PIDS Full"}
BASE_FIELDS=["dataset","experiment","successful_seeds","failed_seeds","successful_seed_count","failed_seed_count"]
OUT_FIELDS=BASE_FIELDS+[f"{m}_{s}" for m in METRIC_DIRECTION for s in ("mean","std","median","best","valid_n")]
def num(value):
 try: return float(value)
 except (ValueError,TypeError): return float("nan")
def read_csv(path):
 with path.open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))

# --------------------------------------------------------------------------- #
# E fix: deterministic per-seed attempt resolution
# --------------------------------------------------------------------------- #
def _resolve_seed_attempts(rows):
    """
    Resolve ambiguous multiple-attempt seeds to a single authoritative record.

    Rules (deterministic, paper-grade):
      1. One row for a seed: use it.
      2. Multiple rows, exactly one completed: use the completed row.
      3. Multiple rows, none completed: count seed as failed once.
      4. Multiple rows, multiple completed: FAIL FAST — raise ValueError.

    all_runs.csv is preserved with ALL rows for audit; only the authoritative
    subset contributes to paper table counts and metric statistics.
    """
    from collections import defaultdict
    by_key = defaultdict(list)
    for row in rows:
        key = (row.get("dataset",""), row.get("config",""), int(row.get("seed", 0)))
        by_key[key].append(row)

    resolved = []
    ambiguities = []
    for key, group in by_key.items():
        completed = [r for r in group if r.get("status") == "completed"]
        if len(group) == 1:
            resolved.append(group[0])
        elif len(completed) == 1:
            resolved.append(completed[0])
        elif len(completed) == 0:
            resolved.append(group[0])
        else:
            ambiguities.append(key)
    if ambiguities:
        raise ValueError(
            f"Ambiguous completed attempts — multiple completed rows for the same "
            f"(dataset, config, seed) identity: {ambiguities!r}. "
            f"Manual resolution required before exporting paper tables."
        )
    return resolved

def aggregate(rows, mapping):
 groups={}
 authoritative = _resolve_seed_attempts(rows)
 for row in authoritative:
  name=mapping.get(row.get("config",""))
  if name is not None: groups.setdefault((row.get("dataset",""),name),[]).append(row)
 output=[]
 for (dataset,name),group in sorted(groups.items()):
  success=sorted(int(r["seed"]) for r in group if r.get("status")=="completed"); failed=sorted(int(r["seed"]) for r in group if r.get("status")!="completed"); out={key:"" for key in OUT_FIELDS}; out.update({"dataset":dataset,"experiment":name,"successful_seeds":",".join(map(str,success)),"failed_seeds":",".join(map(str,failed)),"successful_seed_count":len(success),"failed_seed_count":len(failed)})
  completed=[r for r in group if r.get("status")=="completed"]
  for metric,direction in METRIC_DIRECTION.items():
   vals=[num(r.get(metric)) for r in completed]; vals=[v for v in vals if math.isfinite(v)]; out[f"{metric}_valid_n"]=len(vals)
   if vals:
    out[f"{metric}_mean"]=sum(vals)/len(vals); out[f"{metric}_median"]=statistics.median(vals); out[f"{metric}_best"]=max(vals) if direction=="higher" else min(vals); out[f"{metric}_std"]=statistics.stdev(vals) if len(vals)>1 else float("nan")
   else: out.update({f"{metric}_mean":float("nan"),f"{metric}_std":float("nan"),f"{metric}_median":float("nan"),f"{metric}_best":float("nan")})
  output.append(out)
 return output
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="",dir=path.parent,delete=False) as f:
  w=csv.DictWriter(f,fieldnames=OUT_FIELDS); w.writeheader(); w.writerows(rows); temp=Path(f.name)
 os.replace(temp,path)
def main(argv:Sequence[str]|None=None):
 p=argparse.ArgumentParser(description="Export paper tables from all_runs.csv."); p.add_argument("--artifact-root",required=True); p.add_argument("--input",default=None); a=p.parse_args(argv); root=Path(a.artifact_root).expanduser().resolve(); rows=read_csv(Path(a.input).expanduser().resolve() if a.input else root/"results"/"all_runs.csv"); outputs={"main_results.csv":MAIN,"ablation_results.csv":ABLATION,"calibration_results.csv":CALIBRATION,"efficiency_results.csv":EFFICIENCY}
 for name,mapping in outputs.items(): write(root/"results"/name,aggregate(rows,mapping))
 return [root/"results"/name for name in outputs]
if __name__=="__main__":
 for path in main(): print(path)
