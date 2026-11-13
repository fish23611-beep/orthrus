"""只读审计 E5 原始/校准预测；不读取攻击标签，不计算检测指标或补齐文件。

复用 E3 的流式哈希和独占 JSON 输出。完整性失败也写审计结论，进程返回 2。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e3_postprocessing import HashReader, write_json, sha, fingerprint
import numpy as np

RAW_HEADER = ["event_index", "time", "srcnode", "dstnode", "srcmsg", "dstmsg", "src_type", "dst_type",
              "edge_type", "edge_type_index", "loss", "loss_type", "loss_time_src", "loss_time_dst",
              "loss_time", "score_raw", "src_time_target", "dst_time_target", "src_time_prediction", "dst_time_prediction"]
CAL_HEADER = RAW_HEADER + ["empirical_p", "score_calibrated", "calibration_level"]
FROZEN = {"schemes": [{"method": "topk_mean", "k": 5}, {"method": "topk_mean", "k": 1}],
          "include_dst": True, "score_field": "score_calibrated", "quantile": 0.999,
          "threshold_source": "仅使用各方案的 E5 验证节点分数，禁止使用 E3 阈值或 E5 测试标签选参",
          "prediction_rule": "score > threshold", "auprc_definition": "average_precision_score",
          "metrics": ["tp", "fp", "fn", "tn", "precision", "recall", "fpr", "mcc", "auprc", "auroc"],
          "on_integrity_failure": "停止于完整性审计，不计算任何检测指标、攻击重合率或误报诊断"}


class CSVStream:
    def __init__(self, path, expected_header):
        self.path = Path(path)
        self.before = fingerprint(path)
        self.raw = HashReader(path)
        self.text = io.TextIOWrapper(io.BufferedReader(self.raw), encoding="utf-8", newline="")
        self.reader = csv.reader(self.text, strict=True)
        self.header = next(self.reader)
        if self.header != expected_header:
            self.text.close()
            raise ValueError(f"列结构不匹配: {path}: {self.header}")
        self.rows = 0
        with self.path.open("rb") as f:
            if self.path.stat().st_size:
                f.seek(-1, 2)
                self.ends_newline = f.read(1) == b"\n"
            else:
                self.ends_newline = False

    def take(self, count):
        rows = list(itertools.islice(self.reader, count))
        self.rows += len(rows)
        return rows

    def finish(self):
        evidence = dict(self.before, sha256=self.raw.digest.hexdigest(), rows=self.rows,
                        ends_newline=self.ends_newline, fields=len(self.header))
        evidence["unchanged"] = self.before == fingerprint(self.path)
        self.text.close()
        if not evidence["unchanged"]:
            raise ValueError(f"审计期间输入发生变化: {self.path}")
        return evidence


def audit_split(raw_paths, specs, calibrated_path, chunk_size=20000):
    if len(raw_paths) != len(specs):
        raise ValueError("原始窗口文件数与 manifest 不一致")
    result = {"expected_windows": len(specs), "expected_events": sum(s["num_events"] for s in specs),
              "raw_rows": 0, "calibrated_records": 0, "calibrated_complete_records": 0,
              "calibrated_malformed_records": 0, "original_payload_mismatches": 0,
              "calibration_value_errors": 0, "malformed_examples": [], "payload_examples": [],
              "raw_order_violations": 0, "raw_id_gaps": 0, "raw_time_reversals": 0,
              "raw_numeric_errors": 0, "calibration_log_residual_max": 0., "windows": []}
    cal = CSVStream(calibrated_path, CAL_HEADER)
    previous_id = previous_time = None
    first_id = first_time = last_id = last_time = None
    for file_index, (path, spec) in enumerate(zip(raw_paths, specs)):
        raw = CSVStream(path, RAW_HEADER)
        f_first_id = f_last_id = f_first_time = f_last_time = None
        f_count = 0
        while True:
            rows = raw.take(chunk_size)
            if not rows:
                break
            if any(len(r) != len(RAW_HEADER) for r in rows):
                raise ValueError(f"原始 CSV 存在列数不完整的记录: {path}")
            arr = np.asarray(rows, dtype=object)
            ints = arr[:, [0, 1, 2, 3, 6, 7, 9, 16, 17, 18, 19]].astype(np.int64)
            values = arr[:, [10, 11, 12, 13, 14, 15]].astype(np.float64)
            result["raw_numeric_errors"] += int((~np.isfinite(values)).sum() + (ints < 0).sum())
            ids, times = ints[:, 0], ints[:, 1]
            result["raw_order_violations"] += int((np.diff(ids) <= 0).sum()) + int(previous_id is not None and ids[0] <= previous_id)
            result["raw_id_gaps"] += int((np.diff(ids) != 1).sum()) + int(previous_id is not None and ids[0] != previous_id + 1)
            result["raw_time_reversals"] += int((np.diff(times) < 0).sum()) + int(previous_time is not None and times[0] < previous_time)
            previous_id, previous_time = int(ids[-1]), int(times[-1])
            if first_id is None:
                first_id, first_time = int(ids[0]), int(times[0])
            last_id, last_time = int(ids[-1]), int(times[-1])
            if f_first_id is None:
                f_first_id, f_first_time = int(ids[0]), int(times[0])
            f_last_id, f_last_time = int(ids[-1]), int(times[-1])
            f_count += len(rows)
            cal_rows = cal.take(len(rows))
            result["calibrated_records"] += len(cal_rows)
            complete = []
            for offset, (raw_row, cal_row) in enumerate(zip(rows, cal_rows)):
                if len(cal_row) != len(CAL_HEADER):
                    result["calibrated_malformed_records"] += 1
                    if len(result["malformed_examples"]) < 3:
                        result["malformed_examples"].append({"record_ordinal": result["raw_rows"] + offset + 1,
                            "fields": len(cal_row), "expected_fields": len(CAL_HEADER), "event_index": cal_row[0] if cal_row else None,
                            "time": cal_row[1] if len(cal_row) > 1 else None, "last_field": cal_row[-1] if cal_row else None})
                    continue
                result["calibrated_complete_records"] += 1
                complete.append(cal_row)
                if raw_row != cal_row[:len(RAW_HEADER)]:
                    result["original_payload_mismatches"] += 1
                    if len(result["payload_examples"]) < 3:
                        result["payload_examples"].append({"raw_event_index": raw_row[0], "cal_event_index": cal_row[0],
                            "different_fields": [name for name, a, b in zip(RAW_HEADER, raw_row, cal_row) if a != b]})
            if complete:
                calibrated = np.asarray(complete, dtype=object)
                p_and_score = calibrated[:, [20, 21]].astype(float)
                invalid = (~np.isfinite(p_and_score)).any(axis=1) | (p_and_score[:, 0] <= 0) | (p_and_score[:, 0] > 1) | (p_and_score[:, 1] < 0)
                invalid |= ~np.isin(calibrated[:, 22], ["triplet", "type_pair", "global"])
                result["calibration_value_errors"] += int(invalid.sum())
                valid = ~invalid
                if valid.any():
                    residual = np.abs(-np.log(np.maximum(p_and_score[valid, 0], 1e-12)) - p_and_score[valid, 1])
                    result["calibration_log_residual_max"] = max(result["calibration_log_residual_max"], float(residual.max()))
            result["raw_rows"] += len(rows)
        file_evidence = raw.finish()
        file_evidence.update(expected_events=spec["num_events"], expected_global_offset=spec["global_offset"],
                            expected_source_path=spec["relative_path"], first_id=f_first_id, last_id=f_last_id,
                            first_time=f_first_time, last_time=f_last_time)
        file_evidence["coverage_matches_manifest"] = (f_count == spec["num_events"] and f_first_id == spec["global_offset"] and f_last_id == spec["global_offset"] + spec["num_events"] - 1)
        result["windows"].append(file_evidence)
        if (file_index + 1) % 10 == 0:
            print(f"审计窗口 {file_index+1}/{len(specs)}，原始事件 {result['raw_rows']:,}，完整校准记录 {result['calibrated_complete_records']:,}", flush=True)
    # 检查原始全集以外是否还有额外校准记录；逐块计数，不保留全量数据。
    extra = 0
    while True:
        rows = cal.take(chunk_size)
        if not rows:
            break
        extra += len(rows)
    result["calibrated_extra_records"] = extra
    result["calibrated_file"] = cal.finish()
    result.update(first_raw_id=first_id, last_raw_id=last_id, first_raw_time=first_time, last_raw_time=last_time)
    result["raw_complete"] = (result["raw_rows"] == result["expected_events"] and
        all(w["coverage_matches_manifest"] and w["ends_newline"] for w in result["windows"]) and
        result["raw_order_violations"] == result["raw_id_gaps"] == result["raw_time_reversals"] == result["raw_numeric_errors"] == 0)
    result["missing_complete_calibrated_records"] = max(0, result["raw_rows"] - result["calibrated_complete_records"])
    result["calibrated_complete"] = (result["raw_complete"] and result["calibrated_complete_records"] == result["raw_rows"] and
        result["calibrated_malformed_records"] == result["original_payload_mismatches"] == result["calibration_value_errors"] == extra == 0 and
        result["calibration_log_residual_max"] < 1e-10 and result["calibrated_file"]["ends_newline"])
    return result


def main(args):
    started = time.perf_counter()
    run = args.run.resolve(strict=True)
    if run.name != "seed_0" or run.parents[2].name != "THEIA_E5":
        raise ValueError("只允许指定的 THEIA_E5 seed_0")
    if os.stat(args.output.parent).st_dev == os.stat(run).st_dev or shutil.disk_usage(args.output.parent).free < 1024**3:
        raise ValueError("输出位置须与数据盘不同文件系统且至少余1 GiB")
    selection = json.loads((run / "checkpoint_selection.json").read_text())
    runtime = json.loads((run / "runtime.json").read_text())
    manifest = json.loads(args.window_manifest.read_text())
    if selection["selected_epoch"] != "model_epoch_6" or selection["model_selection_method"] != "min_val_objective":
        raise ValueError("checkpoint 选择身份不匹配")
    if manifest["dataset"] != "THEIA_E5" or not manifest["completed"]:
        raise ValueError("窗口 manifest 身份或完成标志不匹配")
    args.output.mkdir(exist_ok=False)
    checkpoint = run / "checkpoints/model_epoch_6/checkpoint.pt"
    tracked = [run / "checkpoint_selection.json", run / "config_resolved.yml", run / "runtime.json", args.window_manifest, checkpoint]
    before = [fingerprint(p) for p in tracked]
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "frozen_comparison": FROZEN,
                "stage": "仅数据完整性审计", "test_labels_read": False,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "audit_script_sha256": sha(__file__),
                "reused_e3_script_sha256": sha(Path(__file__).with_name("analyze_e3_postprocessing.py")),
                "sources": [dict(fingerprint(p), sha256=sha(p)) for p in tracked]}
    write_json(args.output / "protocol.json", protocol)
    results = {}
    for split, cal_name in (("val", "validation"), ("test", "test")):
        specs = sorted([w for w in manifest["window_specs"] if w["split_name"] == split], key=lambda w: w["split_window_index"])
        paths = sorted((run / "edge_scores" / split / "model_epoch_6").glob("*.csv"))
        print(f"开始审计 {split}: {len(specs)} 窗口，预期 {sum(w['num_events'] for w in specs):,} 事件", flush=True)
        results[split] = audit_split(paths, specs, run / f"evaluation_results/calibration/model_epoch_6/{cal_name}_calibrated.csv", args.chunk_size)
        write_json(args.output / f"{split}_integrity.json", results[split])
    after = [fingerprint(p) for p in tracked]
    if before != after:
        raise ValueError("审计期间来源身份发生变化")
    passed = all(r["calibrated_complete"] for r in results.values())
    result = {"integrity_passed": passed, "comparison_executed": False, "test_labels_read": False,
              "conclusion": "仅完成完整性审计；本脚本不计算检测指标" if passed else "预测不完整，停止于数据完整性审计，未计算任何检测性能",
              "checkpoint_selection": selection, "original_runtime_status": runtime["status"],
              "original_runtime_error": runtime.get("error_message"),
              "expected_total_runtime_processed_events": runtime["testing"]["processed_event_count"],
              "original_node_scores_files": sorted(p.name for p in (run / "node_scores").iterdir()),
              "calibration_summary_exists": (run / "evaluation_results/calibration/model_epoch_6/calibration_summary.json").is_file(),
              "inputs_unchanged": before == after,
              "split_summary": {k: {key: value for key, value in r.items() if key not in ["windows", "calibrated_file"]} for k, r in results.items()},
              "resource": {"elapsed_seconds": time.perf_counter() - started,
                           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "root_free_bytes": shutil.disk_usage(args.output).free,
                           "data_disk_free_bytes": shutil.disk_usage(run).free}}
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=20000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("chunk-size 必须为正数")
    sys.exit(main(args))
