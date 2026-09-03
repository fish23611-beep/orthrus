from __future__ import annotations
import csv,math,sys
from pathlib import Path
SRC_ROOT=Path(__file__).resolve().parents[1]/"src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0,str(SRC_ROOT))
from experiments import collect_results,export_tables

def write_runs(root,rows): collect_results.write_csv(root/"results"/"all_runs.csv",rows)
def row(config,seed,status="completed",dataset="THEIA_E3",**values):
 r=collect_results.empty_row();r.update({"dataset":dataset,"config":config,"config_path":f"/x/{config}.yml","config_id":config,"seed":seed,"status":status,"experiment":config,"artifact_dir":"","git_commit":"","model_variant":"","backbone":"","dataset_view":"","failure_type":"","failure_message":"","failure_path":"","collection_error":""});r.update(values);return r
def read(path): return list(csv.DictReader(path.open(encoding="utf-8")))
def test_mean_sample_std_median_and_best(tmp_path):
 write_runs(tmp_path,[row("baseline",0,MCC=.6,FP=20),row("baseline",1,MCC=.8,FP=5),row("baseline",2,MCC=1.0,FP=10)])
 export_tables.main(["--artifact-root",str(tmp_path)]);r=read(tmp_path/"results"/"main_results.csv")[0]
 assert round(float(r["MCC_mean"]), 6)==.8 and round(float(r["MCC_median"]), 6)==.8 and float(r["MCC_best"])==1.0 and round(float(r["MCC_std"]),6)==.2 and float(r["FP_best"])==5
def test_single_seed_nan_std_and_failed_exclusion(tmp_path):
 write_runs(tmp_path,[row("baseline",0,MCC=.8),row("baseline",1,"failed",MCC=0),row("baseline",2,MCC=.6)])
 export_tables.main(["--artifact-root",str(tmp_path)]);r=read(tmp_path/"results"/"main_results.csv")[0]
 assert float(r["MCC_mean"])==.7 and r["successful_seeds"]=="0,2" and r["failed_seeds"]=="1"
 write_runs(tmp_path,[row("baseline",0,MCC=.8)]);export_tables.main(["--artifact-root",str(tmp_path)]);r=read(tmp_path/"results"/"main_results.csv")[0]
 assert math.isnan(float(r["MCC_std"])) and float(r["MCC_best"])==.8
def test_metric_nan_excluded_but_seed_remains_successful(tmp_path):
 write_runs(tmp_path,[row("baseline",0,AUROC=float("nan")),row("baseline",1,AUROC=.9)])
 export_tables.main(["--artifact-root",str(tmp_path)]);r=read(tmp_path/"results"/"main_results.csv")[0]
 assert float(r["AUROC_mean"])==.9 and r["AUROC_valid_n"]=="1" and r["successful_seeds"]=="0,1"
def test_explicit_classification_and_dataset_isolation_are_stable(tmp_path):
 write_runs(tmp_path,[row("magic",0),row("backbone_mlp",0),row("backbone_graphsage_baseline",0),row("backbone_graphsage",0),row("calibration_max",0),row("calibration_global_p",0),row("ablation_no_calibration",0),row("efficiency_multiscale",0),row("mstc_full",0),row("unknown",0),row("baseline",0,dataset="THEIA_E3"),row("baseline",0,dataset="THEIA_E5")])
 export_tables.main(["--artifact-root",str(tmp_path)]); main=read(tmp_path/"results"/"main_results.csv");abl=read(tmp_path/"results"/"ablation_results.csv");backbone=read(tmp_path/"results"/"backbone_generalization_results.csv")
 assert {r["experiment"] for r in main}>={"MAGIC","GraphSAGE","MSTC-PIDS Full","ORTHRUS-ano"};assert "Semantic MLP" not in {r["experiment"] for r in main};assert {r["experiment"] for r in abl}>={"Full","w/o Calibration","Global Calibration"};assert {r["experiment"] for r in backbone}=={"GraphTransformer + MSTC","GraphSAGE + MSTC"};assert not (tmp_path/"results"/"score_calibration_results.csv").exists();assert not (tmp_path/"results"/"efficiency_results.csv").exists();assert len([r for r in main if r["experiment"]=="ORTHRUS-ano"] )==2
 first=(tmp_path/"results"/"main_results.csv").read_text();export_tables.main(["--artifact-root",str(tmp_path)]);assert first==(tmp_path/"results"/"main_results.csv").read_text()

def test_archived_optional_exports_require_explicit_flag(tmp_path):
 write_runs(tmp_path,[row("backbone_mlp",0),row("ablation_no_gate",0),row("calibration_max",0),row("efficiency_multiscale",0),row("multiscale_recent20",0),row("time_type_only",0)])
 outputs=export_tables.main(["--artifact-root",str(tmp_path),"--include-archived-optional"])
 assert {path.name for path in outputs}>={"archived_optional_results.csv","score_calibration_results.csv","node_decision_results.csv","efficiency_results.csv","multiscale_diagnostic_results.csv","time_diagnostic_results.csv"}
 assert {r["experiment"] for r in read(tmp_path/"results"/"archived_optional_results.csv")}=={"Semantic MLP","w/o Gate"}
 assert [r["experiment"] for r in read(tmp_path/"results"/"node_decision_results.csv")]==["Max Validation"]
