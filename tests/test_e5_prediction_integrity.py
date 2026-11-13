"""E5 完整性门控的 CPU 测试，不读取任何正式实验文件。"""
import csv
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("e5_integrity", ROOT / "scripts/audit_e5_prediction_integrity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(index):
    r = [str(0)] * len(audit.RAW_HEADER)
    r[0], r[1] = str(index), str(1000 + index)
    r[2], r[3], r[4], r[5], r[8] = "1", "2", "subject:test", "file:test", "EVENT_OPEN"
    for name in ["loss", "loss_type", "loss_time_src", "loss_time_dst", "loss_time", "score_raw"]:
        r[audit.RAW_HEADER.index(name)] = "1.0"
    return r


def save(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fixture_files(tmp_path):
    raws = [tmp_path / "raw_1.csv", tmp_path / "raw_2.csv"]
    records = [row(i) for i in range(5)]
    save(raws[0], audit.RAW_HEADER, records[:2])
    save(raws[1], audit.RAW_HEADER, records[2:])
    cal = tmp_path / "cal.csv"
    save(cal, audit.CAL_HEADER, [r + ["1.0", "0.0", "triplet"] for r in records])
    specs = [dict(num_events=2, global_offset=0, relative_path="test/a"),
             dict(num_events=3, global_offset=2, relative_path="test/b")]
    return raws, cal, specs, records


@pytest.mark.parametrize("chunk_size", [1, 2, 20])
def test_complete_raw_and_calibration_across_chunks(tmp_path, chunk_size):
    raws, cal, specs, _ = fixture_files(tmp_path)
    result = audit.audit_split(raws, specs, cal, chunk_size)
    assert result["raw_complete"] and result["calibrated_complete"]
    assert result["calibrated_complete_records"] == 5
    assert result["original_payload_mismatches"] == 0
    assert all(w["coverage_matches_manifest"] for w in result["windows"])


def test_truncated_calibration_never_passes_even_if_existing(tmp_path):
    raws, cal, specs, records = fixture_files(tmp_path)
    save(cal, audit.CAL_HEADER, [r + ["1.0", "0.0", "triplet"] for r in records[:3]])
    with cal.open("a") as f:
        f.write(",".join(records[3][:12]))
    result = audit.audit_split(raws, specs, cal, 2)
    assert result["raw_complete"]
    assert not result["calibrated_complete"]
    assert result["calibrated_complete_records"] == 3
    assert result["calibrated_malformed_records"] == 1
    assert result["missing_complete_calibrated_records"] == 2
    assert not result["calibrated_file"]["ends_newline"]
    assert result["malformed_examples"][0]["fields"] == 12


def test_well_formed_short_prefix_is_still_incomplete(tmp_path):
    raws, cal, specs, records = fixture_files(tmp_path)
    save(cal, audit.CAL_HEADER, [r + ["1.0", "0.0", "triplet"] for r in records[:4]])
    result = audit.audit_split(raws, specs, cal, 3)
    assert result["calibrated_file"]["ends_newline"]
    assert result["calibrated_malformed_records"] == 0
    assert not result["calibrated_complete"]
    assert result["missing_complete_calibrated_records"] == 1


def test_same_count_wrong_original_score_is_rejected(tmp_path):
    raws, cal, specs, records = fixture_files(tmp_path)
    records[2][audit.RAW_HEADER.index("score_raw")] = "9.0"
    save(cal, audit.CAL_HEADER, [r + ["1.0", "0.0", "triplet"] for r in records])
    result = audit.audit_split(raws, specs, cal, 3)
    assert result["original_payload_mismatches"] == 1
    assert not result["calibrated_complete"]


def test_incomplete_raw_window_rejected_by_independent_manifest(tmp_path):
    raws, cal, specs, _ = fixture_files(tmp_path)
    specs[-1]["num_events"] = 4
    result = audit.audit_split(raws, specs, cal, 2)
    assert not result["raw_complete"]
    assert not result["calibrated_complete"]


def test_nonfinite_calibration_and_extra_record_rejected(tmp_path):
    raws, cal, specs, records = fixture_files(tmp_path)
    save(cal, audit.CAL_HEADER, [r + ["nan", "0.0", "triplet"] for r in records] + [records[-1] + ["1.0", "0.0", "triplet"]])
    result = audit.audit_split(raws, specs, cal, 2)
    assert result["calibration_value_errors"] == 5
    assert result["calibrated_extra_records"] == 1
    assert not result["calibrated_complete"]


def test_schema_mismatch_and_frozen_two_schemes(tmp_path):
    raws, cal, specs, _ = fixture_files(tmp_path)
    save(cal, audit.CAL_HEADER[:-1], [])
    with pytest.raises(ValueError, match="列结构不匹配"):
        audit.audit_split(raws, specs, cal)
    assert [s["k"] for s in audit.FROZEN["schemes"]] == [5, 1]
    assert audit.FROZEN["quantile"] == .999
