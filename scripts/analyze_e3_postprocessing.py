"""CPU-only, single-test-scan exploratory E3 postprocessing. Never writes inputs.

Run with PYTHONDONTWRITEBYTECODE=1. Fit/freeze uses validation only; there is no
winner selection using test metrics. Output directory must be new and on root FS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from mstc.metrics import compute_classification_metrics
from mstc.thresholding import compute_node_threshold

KS = (1, 3, 5, 10)
QS = (0.999, 0.9995, 0.9998, 0.9999)
VARIANTS = ("saved_calibrated", "raw_joint", "raw_type", "cal_type", "cal_time01", "cal_time06")
COLUMNS = ["event_index", "time", "srcnode", "dstnode", "src_type", "dst_type",
           "edge_type_index", "loss_type", "loss_time", "loss_time_src", "loss_time_dst",
           "score_raw", "empirical_p", "score_calibrated", "calibration_level"]
PRIMARY = {"variant": "saved_calibrated", "direction": "both", "k": 5, "quantile": 0.9995}


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            x = x.item()
        return None if isinstance(x, float) and not np.isfinite(x) else x
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(clean(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class HashReader(io.RawIOBase):
    """Hash the same bytes pandas consumes, without a second full CSV scan."""
    def __init__(self, path):
        self.file = Path(path).open("rb")
        self.digest = hashlib.sha256()
    def readable(self):
        return True
    def readinto(self, b):
        n = self.file.readinto(b)
        self.digest.update(memoryview(b)[:n])
        return n
    def close(self):
        self.file.close()
        super().close()


def read_chunks(path, chunk_size, evidence):
    raw = HashReader(path)
    with io.BufferedReader(raw) as handle:
        for chunk in pd.read_csv(handle, usecols=COLUMNS, chunksize=chunk_size,
                                 float_precision="round_trip"):
            if chunk.isna().any().any():
                raise ValueError(f"Missing fields in {path}")
            numeric = chunk.drop(columns="calibration_level")
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(f"Non-finite fields in {path}")
            for field in ("event_index", "time", "srcnode", "dstnode", "src_type", "dst_type", "edge_type_index"):
                if chunk[field].dtype.kind not in "iu" or (chunk[field] < 0).any():
                    raise ValueError(f"Invalid integer field {field}")
            yield chunk
        evidence.update(path=str(Path(path).resolve()), bytes=Path(path).stat().st_size,
                        sha256=raw.digest.hexdigest())


class References:
    """Vector query of the production empirical CDF and exact LOO convention.

    New score definitions always get newly fitted validation references.
    Tested against HierarchicalRelationCalibrator, including fallback and ties.
    """
    def __init__(self, frame, values, min_triplet=100, min_pair=200):
        self.min_triplet, self.min_pair = min_triplet, min_pair
        self.global_values = np.sort(values)
        self.triplets = {key: np.sort(values[idx]) for key, idx in
                         frame.groupby(["src_type", "edge_type_index", "dst_type"]).indices.items()}
        self.pairs = {key: np.sort(values[idx]) for key, idx in
                      frame.groupby(["src_type", "dst_type"]).indices.items()}

    def query(self, frame, values, loo=False):
        result = np.empty(len(frame))
        levels = np.empty(len(frame), dtype=object)
        adjustment = int(loo)
        for key, idx in frame.groupby(["src_type", "edge_type_index", "dst_type"]).indices.items():
            ref = self.triplets.get(key)
            level = "triplet"
            if ref is None or len(ref) - adjustment < self.min_triplet:
                ref = self.pairs.get((key[0], key[2]))
                level = "type_pair"
                if ref is None or len(ref) - adjustment < self.min_pair:
                    ref, level = self.global_values, "global"
            count_ge = len(ref) - np.searchsorted(ref, values[idx], side="left")
            p = (1.0 + count_ge - adjustment) / (len(ref) + 1.0 - adjustment)
            result[idx] = -np.log(np.maximum(p, 1e-12))
            levels[idx] = level
        return result, levels


class TopK:
    """Bounded O(nodes * K) memory; retain multiplicity, including self loops."""
    def __init__(self, node_ids, maximum=10):
        self.ids = np.asarray(node_ids, dtype=np.int64)
        if len(self.ids) == 0 or np.any(np.diff(self.ids) <= 0):
            raise ValueError("Node IDs must be sorted, unique and non-empty")
        self.maximum = maximum
        self.values = np.full((len(node_ids), maximum), -np.inf)
        self.counts = np.zeros(len(node_ids), dtype=np.int64)

    def update(self, ids, scores):
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite event score")
        pos = np.searchsorted(self.ids, ids)
        if np.any(pos >= len(self.ids)) or not np.array_equal(self.ids[pos], ids):
            raise ValueError("Event node outside frozen node universe")
        order = np.lexsort((-scores, pos))
        nodes, ordered = pos[order], scores[order]
        unique, start, counts = np.unique(nodes, return_index=True, return_counts=True)
        ranks = np.arange(len(nodes)) - np.repeat(start, counts)
        keep = ranks < self.maximum
        incoming = np.full((len(unique), self.maximum), -np.inf)
        groups = np.repeat(np.arange(len(unique)), counts)[keep]
        incoming[groups, ranks[keep]] = ordered[keep]
        merged = np.concatenate((self.values[unique], incoming), axis=1)
        self.values[unique] = np.partition(merged, -self.maximum, axis=1)[:, -self.maximum:]
        self.counts[unique] += counts

    def scores(self, k, other=None):
        values = self.values if other is None else np.concatenate((self.values, other.values), axis=1)
        highest = np.sort(values, axis=1)[:, ::-1][:, :k]
        counts = np.minimum(self.counts if other is None else self.counts + other.counts, k)
        sums = np.where(np.isfinite(highest), highest, 0).sum(axis=1)
        return np.divide(sums, counts, out=np.full(len(counts), -np.inf), where=counts > 0)


class EventAudit:
    def __init__(self):
        self.seen = np.zeros(0, dtype=bool)
        self.rows = self.duplicates = self.id_reversals = self.time_reversals = self.self_loops = 0
        self.previous_id = self.previous_time = None
        self.min_id = self.max_id = self.min_time = self.max_time = None
        self.max_score_residual = self.max_time_residual = self.max_cal_residual = 0.0
        self.levels = Counter()
        self.relations = Counter()
        self.moments = Counter()
        self.tail_moments = Counter()

    def add(self, frame, cal_residual):
        ids, times = frame.event_index.to_numpy(), frame.time.to_numpy()
        maximum = int(ids.max())
        if maximum > 100_000_000:
            raise ValueError("Event-index bitset limit exceeded; refusing excessive allocation")
        if maximum >= len(self.seen):
            enlarged = np.zeros(max(maximum + 1, 2 * len(self.seen)), dtype=bool)
            enlarged[:len(self.seen)] = self.seen
            self.seen = enlarged
        unique = np.unique(ids)
        self.duplicates += len(ids) - len(unique) + int(self.seen[unique].sum())
        self.seen[unique] = True
        self.id_reversals += int((np.diff(ids) < 0).sum()) + int(self.previous_id is not None and ids[0] < self.previous_id)
        self.time_reversals += int((np.diff(times) < 0).sum()) + int(self.previous_time is not None and times[0] < self.previous_time)
        self.previous_id, self.previous_time = int(ids[-1]), int(times[-1])
        self.min_id = int(ids.min()) if self.min_id is None else min(self.min_id, int(ids.min()))
        self.max_id = maximum if self.max_id is None else max(self.max_id, maximum)
        self.min_time = int(times.min()) if self.min_time is None else min(self.min_time, int(times.min()))
        self.max_time = int(times.max()) if self.max_time is None else max(self.max_time, int(times.max()))
        self.rows += len(ids)
        self.self_loops += int((frame.srcnode == frame.dstnode).sum())
        self.max_score_residual = max(self.max_score_residual, float(np.abs(frame.score_raw - frame.loss_type - 0.3 * frame.loss_time).max()))
        self.max_time_residual = max(self.max_time_residual, float(np.abs(frame.loss_time - (frame.loss_time_src + frame.loss_time_dst) / 2).max()))
        self.max_cal_residual = max(self.max_cal_residual, cal_residual)
        self.levels.update(frame.calibration_level)
        self.relations.update(frame.edge_type_index)
        for name, mask in (("moments", np.ones(len(frame), dtype=bool)), ("tail_moments", frame.score_calibrated.to_numpy() > 5)):
            target = getattr(self, name)
            target["events"] += int(mask.sum())
            for field in ("loss_type", "loss_time", "score_raw"):
                target[field + "_sum"] += float(frame[field].to_numpy()[mask].sum())

    def summary(self):
        return {key: value for key, value in vars(self).items() if key not in {"seen", "previous_id", "previous_time"}}


def score_variants(frame, refs, loo=False):
    raw, kind, temporal = (frame[c].to_numpy() for c in ("score_raw", "loss_type", "loss_time"))
    matched, levels = refs["joint"].query(frame, raw, loo)
    residual = float(np.abs(matched - frame.score_calibrated.to_numpy()).max())
    if residual > 1e-10 or not np.array_equal(levels, frame.calibration_level.to_numpy()):
        raise ValueError("Saved calibrated scores do not match validation-only references")
    return {
        "saved_calibrated": frame.score_calibrated.to_numpy(), "raw_joint": raw, "raw_type": kind,
        "cal_type": refs["type"].query(frame, kind, loo)[0],
        "cal_time01": refs["time01"].query(frame, kind + 0.1 * temporal, loo)[0],
        "cal_time06": refs["time06"].query(frame, kind + 0.6 * temporal, loo)[0],
    }, residual


def accumulators(ids):
    return {name: [TopK(ids), TopK(ids)] for name in VARIANTS}


def update_all(frame, banks, values):
    for name, scores in values.items():
        banks[name][0].update(frame.srcnode.to_numpy(), scores)
        banks[name][1].update(frame.dstnode.to_numpy(), scores)


def plans(banks):
    for name, (src, dst) in banks.items():
        for direction in (("both", "src", "dst") if name == "saved_calibrated" else ("both",)):
            for k in KS:
                scores = src.scores(k, dst) if direction == "both" else (src if direction == "src" else dst).scores(k)
                yield name, direction, k, scores


def fingerprint(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main(args):
    started = time.perf_counter()
    run = args.run.resolve(strict=True)
    # Explicit E3 identity gate; never guess checkpoints from other runs.
    config = run / "config_resolved.yml"
    official = json.loads((run / "node_scores/metrics.json").read_text())
    if run.name != "seed_0" or run.parents[2].name != "THEIA_E3" or official["selected_epoch"] != 4:
        raise ValueError("Expected THEIA_E3 seed_0 selected epoch 4")
    cal = run / "evaluation_results/calibration/model_epoch_4"
    summary = json.loads((cal / "calibration_summary.json").read_text())
    if summary["calibration_parameters"] != {"epsilon": 1e-12, "min_triplet_samples": 100, "min_type_pair_samples": 200}:
        raise ValueError("Unsupported calibration parameters")
    if os.stat(args.output.parent).st_dev == os.stat(run).st_dev:
        raise ValueError("Output must be on a separate filesystem from the data disk")
    disk_before = shutil.disk_usage(args.output.parent)
    if disk_before.free < 1024**3:
        raise ValueError("Less than 1 GiB root-disk safety margin")
    args.output.mkdir(exist_ok=False)
    source_paths = [config, run / "node_scores/metrics.json", run / "node_scores/node_predictions.csv",
                    run / "node_scores/event_predictions.csv", cal / "validation_calibrated.csv",
                    cal / "calibration_summary.json", run / "checkpoints/model_epoch_4/checkpoint.pt",
                    args.metadata / "ground_truth_nodes.pkl", args.metadata / "uuid_to_node_id.pkl",
                    args.metadata / "dataset_manifest.json"]
    before = [fingerprint(p) for p in source_paths]
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "script_sha256": sha(__file__), "ks": KS, "quantiles": QS, "variants": VARIANTS,
                "primary_predeclared": PRIMARY, "selection": "No test-based winner selection; primary is a conservative threshold probe, not a claim of optimality",
                "validation_labels": "No event attack labels; graph_9 predates listed attacks. Do not optimize supervised metrics.",
                "direction_missing_node_policy": "Keep entire baseline universe, absent-direction score=0 for nonnegative losses",
                "randomness": "none", "chunk_size": args.chunk_size, "sources": before}
    write_json(args.output / "protocol.json", protocol)
    print("protocol frozen; loading validation", flush=True)
    vevidence = {}
    validation = pd.concat(read_chunks(cal / "validation_calibrated.csv", args.chunk_size, vevidence), ignore_index=True)
    kind, temporal = validation.loss_type.to_numpy(), validation.loss_time.to_numpy()
    refs = {"joint": References(validation, validation.score_raw.to_numpy()),
            "type": References(validation, kind), "time01": References(validation, kind + 0.1 * temporal),
            "time06": References(validation, kind + 0.6 * temporal)}
    val_ids = np.union1d(validation.srcnode, validation.dstnode)
    vbanks = accumulators(val_ids)
    vaudit = EventAudit()
    values, residual = score_variants(validation, refs, loo=True)
    update_all(validation, vbanks, values)
    vaudit.add(validation, residual)
    thresholds, val_distributions = {}, {}
    for name, direction, k, scores in plans(vbanks):
        finite = scores[np.isfinite(scores)]  # Fit on observed nodes in that direction only.
        key = f"{name}/{direction}/k{k}"
        thresholds[key] = {str(q): compute_node_threshold(finite, quantile=q) for q in QS}
        val_distributions[key] = {"nodes": len(finite), "quantiles": np.quantile(finite, [.5, .9, .99, .999, 1]).tolist(),
                                  "exceedance": {str(q): float((finite > thresholds[key][str(q)]).mean()) for q in QS}}
    write_json(args.output / "frozen_validation_thresholds.json", {"thresholds": thresholds, "distributions": val_distributions,
              "validation_source": vevidence, "test_labels_read": False})
    print("validation thresholds frozen; loading canonical node universe and labels", flush=True)
    nodes_path = run / "node_scores/node_predictions.csv"
    nodes = pd.read_csv(nodes_path, float_precision="round_trip").sort_values("node_id").reset_index(drop=True)
    if nodes.isna().any().any() or nodes.node_id.duplicated().any() or not np.isfinite(nodes.to_numpy()).all():
        raise ValueError("Invalid canonical node table")
    if not nodes.y_true.isin([0, 1]).all() or not nodes.y_hat.isin([0, 1]).all():
        raise ValueError("Nonbinary labels")
    ids = nodes.node_id.to_numpy()
    # These trusted local project caches are read directly; no cache manager/DB call.
    with (args.metadata / "ground_truth_nodes.pkl").open("rb") as f:
        gt = set(map(int, pickle.load(f)))
    with (args.metadata / "uuid_to_node_id.pkl").open("rb") as f:
        uuid_map = pickle.load(f)
    mapped_gt = set()
    for name in ("node_Browser_Extension_Drakon_Dropper.csv", "node_Firefox_Backdoor_Drakon_In_Memory.csv"):
        import csv
        with (args.ground_truth / name).open(newline="") as f:
            for row in csv.reader(f):
                mapped_gt.add(int(uuid_map[row[0]]))
    if mapped_gt != gt or not np.array_equal(np.isin(ids, list(gt)).astype(int), nodes.y_true.to_numpy()):
        raise ValueError("Ground Truth UUID mapping does not match canonical node labels")
    del uuid_map
    truth = nodes.y_true.to_numpy()
    baseline = compute_classification_metrics(truth, nodes.y_hat, nodes.score)
    metric_deltas = {key: float(baseline[key]) - official[key] for key in ("tp", "fp", "fn", "tn", "precision", "recall", "fpr", "mcc", "auprc", "auroc")}
    if any(abs(d) > 1e-12 for d in metric_deltas.values()):
        raise ValueError(f"Canonical baseline mismatch: {metric_deltas}")
    tbanks = accumulators(ids)
    taudit, tevidence = EventAudit(), {}
    diagnostic = Counter()
    gt_array = np.array(sorted(gt))
    for frame in read_chunks(run / "node_scores/event_predictions.csv", args.chunk_size, tevidence):
        values, residual = score_variants(frame, refs)
        update_all(frame, tbanks, values)
        taudit.add(frame, residual)
        incident = np.isin(frame.srcnode, gt_array) | np.isin(frame.dstnode, gt_array)
        for label, mask in (("gt_incident", incident), ("other", ~incident)):
            diagnostic[label + "_events"] += int(mask.sum())
            diagnostic[label + "_type_sum"] += float(frame.loss_type.to_numpy()[mask].sum())
            diagnostic[label + "_time_sum"] += float(frame.loss_time.to_numpy()[mask].sum())
        if taudit.rows % (args.chunk_size * 5) == 0:
            print(f"test events={taudit.rows:,}; seconds={time.perf_counter()-started:.1f}", flush=True)
    if taudit.rows != summary["event_counts"]["num_test_events"] or vaudit.rows != summary["event_counts"]["num_validation_events"]:
        raise ValueError("Event count mismatch")
    if taudit.duplicates or vaudit.duplicates or np.any(taudit.seen[:min(len(taudit.seen), len(vaudit.seen))] & vaudit.seen[:min(len(taudit.seen), len(vaudit.seen))]):
        raise ValueError("Duplicate/overlapping validation-test event identities")
    if vaudit.max_time >= taudit.min_time:
        raise ValueError("Validation/test time ordering overlap")
    total_counts = tbanks["saved_calibrated"][0].counts + tbanks["saved_calibrated"][1].counts
    if np.any(total_counts == 0):
        raise ValueError("Baseline nodes missing from test events")
    reconstructed = tbanks["saved_calibrated"][0].scores(5, tbanks["saved_calibrated"][1])
    node_max_error = float(np.abs(reconstructed - nodes.score.to_numpy()).max())
    base_threshold = thresholds["saved_calibrated/both/k5"]["0.999"]
    pred_mismatch = int(((reconstructed > base_threshold).astype(int) != nodes.y_hat.to_numpy()).sum())
    if node_max_error > 1e-12 or pred_mismatch:
        raise ValueError(f"Reaggregated baseline mismatch: error={node_max_error}, predictions={pred_mismatch}")
    import csv
    audit_hashes = {}
    for line in (run / "final_audit/final_outputs.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        audit_hashes[Path(path).name] = digest
    hashes = {"metrics.json": sha(run / "node_scores/metrics.json"), "node_predictions.csv": sha(nodes_path), "event_predictions.csv": tevidence["sha256"]}
    if hashes != audit_hashes:
        raise ValueError("Official final-audit hashes do not match")
    rows, distributions = [], {}
    for name, direction, k, scores in plans(tbanks):
        missing = ~np.isfinite(scores)
        # No node drops for direction probes; nonnegative-score neutral value.
        scores[missing] = 0.0
        key = f"{name}/{direction}/k{k}"
        rank = compute_classification_metrics(truth, np.zeros(len(truth), dtype=int), scores)
        distributions[key] = {"absent_direction_nodes": int(missing.sum()),
                              "all_quantiles": np.quantile(scores, [.5, .9, .99, .999, 1]).tolist(),
                              "benign_quantiles": np.quantile(scores[truth == 0], [.5, .9, .99, .999, 1]).tolist(),
                              "attack_quantiles": np.quantile(scores[truth == 1], [0, .5, .9, 1]).tolist()}
        for q in QS:
            threshold = thresholds[key][str(q)]
            metrics = compute_classification_metrics(truth, scores > threshold)
            metrics.update(auprc=rank["auprc"], auroc=rank["auroc"])
            row = dict(variant=name, direction=direction, k=k, quantile=q, threshold=threshold, **metrics)
            row["predeclared_primary"] = all(row[field] == value for field, value in PRIMARY.items())
            rows.append(row)
    with (args.output / "candidate_metrics.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base_positive = nodes.y_hat.to_numpy() == 1
    count_strata = []
    for low, high in ((1, 5), (6, 10), (11, 100), (101, 1000), (1001, 10**12)):
        mask = (total_counts >= low) & (total_counts <= high) & (truth == 0)
        count_strata.append(dict(min_events=low, max_events=high, benign_nodes=int(mask.sum()), fp=int((mask & base_positive).sum())))
    directions = {}
    for label, mask in (("fp", (truth == 0) & base_positive), ("tp", (truth == 1) & base_positive)):
        src_scores, dst_scores = tbanks["saved_calibrated"][0].scores(5), tbanks["saved_calibrated"][1].scores(5)
        directions[label] = {"src_above_baseline_threshold": int((mask & (src_scores > base_threshold)).sum()),
                             "dst_above_baseline_threshold": int((mask & (dst_scores > base_threshold)).sum())}
    after = [fingerprint(p) for p in source_paths]
    if before != after:
        raise ValueError("Input metadata changed during analysis")
    primary = next(row for row in rows if row["predeclared_primary"])
    result = {"baseline_reproduced": True, "baseline": baseline, "official_metric_deltas": metric_deltas,
              "primary_predeclared_result": primary, "candidate_count": len(rows),
              "mapping": {"test_nodes": len(ids), "test_positive": int(truth.sum()), "test_negative": int((truth == 0).sum()),
                          "gt_cache_nodes": len(gt), "gt_uuid_mapping_exact": True,
                          "validation_nodes": len(val_ids), "validation_gt_identity_overlap": int(np.isin(val_ids, gt_array).sum()),
                          "validation_attack_labels_available": False, "test_event_node_universe_exact": True},
              "validation_audit": vaudit.summary(), "test_audit": taudit.summary(),
              "baseline_threshold": base_threshold, "node_score_max_abs_error": node_max_error,
              "baseline_prediction_mismatches": pred_mismatch, "hashes_match_final_audit": True,
              "event_diagnostics": diagnostic, "baseline_benign_frequency_strata": count_strata,
              "baseline_direction_diagnostics": directions, "test_distributions": distributions,
              "sources": {"validation": vevidence, "test": tevidence, "official_hashes": hashes},
              "inputs_unchanged": before == after,
              "protocol_limit": "E3 previously observed; all results exploratory. No candidate selected by test metrics. No independent confirmation.",
              "resource": {"elapsed_seconds": time.perf_counter() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "root_free_before": disk_before.free, "root_free_after": shutil.disk_usage(args.output).free}}
    write_json(args.output / "summary.json", result)
    print(json.dumps({"baseline": baseline, "primary": primary, "resources": result["resource"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=Path(__file__).resolve().parents[1] / "Ground_Truth/darpa/darpa/E3-THEIA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parsed = parser.parse_args()
    if parsed.chunk_size <= 0:
        parser.error("chunk-size must be positive")
    main(parsed)
