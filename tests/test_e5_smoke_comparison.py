"""Exercise saved-output comparison categories using explicitly synthetic CSVs."""
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_e5_smoke import compare_csv, compare_metrics


def write(tmp_path, name, score, *, event=1, prediction=0):
    path=tmp_path/name
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["event_index","time","srcnode","dstnode","score_raw","src_time_prediction"])
        writer.writeheader()
        writer.writerow({"event_index":event,"time":1557806400216652068,"srcnode":38,"dstnode":93994,
                         "score_raw":score,"src_time_prediction":prediction})
    return path


def test_exact_saved_values_and_large_timestamp_identity(tmp_path):
    a=write(tmp_path,"a.csv",1.25)
    b=write(tmp_path,"b.csv",1.25)
    result=compare_csv(a,b)
    assert result["classification"]=="BITWISE_EQUAL"
    assert result["identity_order_equal"]


@pytest.mark.parametrize("right,category", [(1.2500001,"NUMERICALLY_CLOSE_NOT_BITWISE"),(2.0,"DIFFERENT")])
def test_close_and_different_are_distinct(tmp_path,right,category):
    assert compare_csv(write(tmp_path,"a.csv",1.25),write(tmp_path,"b.csv",right))["classification"]==category


def test_different_event_identity_is_structural_mismatch(tmp_path):
    result=compare_csv(write(tmp_path,"a.csv",1.25),write(tmp_path,"b.csv",1.25,event=2))
    assert result["classification"]=="STRUCTURAL_MISMATCH"
    assert not result["identity_order_equal"]


def test_prediction_change_is_not_hidden_by_float_tolerance(tmp_path):
    result=compare_csv(write(tmp_path,"a.csv",1.25),write(tmp_path,"b.csv",1.25,prediction=1))
    assert result["classification"]=="DIFFERENT"
    assert result["fields"]["src_time_prediction"]["changed"]==1


def test_signed_zero_is_not_bitwise_equal(tmp_path):
    result=compare_csv(write(tmp_path,"a.csv",0.0),write(tmp_path,"b.csv",-0.0))
    assert not result["all_float_bitwise_equal"]
    assert result["all_float_close"]


def test_nonfinite_event_score_is_rejected(tmp_path):
    with pytest.raises(ValueError,match="Nonfinite"):
        compare_csv(write(tmp_path,"a.csv",float("nan")),write(tmp_path,"b.csv",float("nan")))


def test_matching_undefined_metric_is_explicit():
    result=compare_metrics({"f1":float("nan"),"fp":1},{"f1":float("nan"),"fp":1})
    assert result["f1"]=={"equal":True,"matching_undefined_nan":True}
    assert result["fp"]["equal"]


def test_one_sided_undefined_metric_is_a_mismatch():
    assert not compare_metrics({"f1":float("nan")},{"f1":0.0})["f1"]["equal"]
