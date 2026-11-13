"""E5 seed0：只用原始保存分数恢复两个冻结后处理方案，不写全量预测。

CPU；不导入模型/数据库。预测数组只读并落盘摘要哈希后，才读取攻击标签。
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
from types import MappingProxyType
import shutil

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e3_postprocessing import References, TopK, HashReader, write_json, sha, fingerprint
from audit_e5_prediction_integrity import CSVStream, RAW_HEADER, CAL_HEADER
import numpy as np
import pandas as pd
import yaml
from mstc.calibration import HierarchicalRelationCalibrator
from mstc.metrics import compute_classification_metrics
from mstc.thresholding import compute_node_threshold

KS = (5, 1)
Q = 0.999
P_ATOL, SCORE_ATOL = 1e-15, 1e-12
LEVELS = np.array(["triplet", "type_pair", "global"], dtype=object)
FIELDS = ["event_index", "time", "srcnode", "dstnode", "src_type", "dst_type", "edge_type_index", "score_raw"]
RAW_INDEX = [RAW_HEADER.index(f) for f in FIELDS]


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def numeric_frame(rows):
    if any(len(r) != 20 for r in rows):
        raise ValueError("原始CSV记录列数不为20")
    arr = np.asarray(rows, dtype=object)
    integers = arr[:, [i for i in range(20) if i not in [4, 5, 8, 10, 11, 12, 13, 14, 15]]].astype(np.int64)
    numbers = arr[:, [10, 11, 12, 13, 14, 15]].astype(float)
    if (integers < 0).any() or not np.isfinite(numbers).all():
        raise ValueError("原始事件含非法ID或非有限数值")
    result = {f: arr[:, i].astype(float if f == "score_raw" else np.int64) for f, i in zip(FIELDS, RAW_INDEX)}
    return pd.DataFrame(result)


def query_full(refs, frame, loo=False):
    """直接返回正式公式的p、score和层级，保留ties与LOO边界回退。"""
    values = frame.score_raw.to_numpy()
    p = np.empty(len(frame)); levels = np.empty(len(frame), dtype=np.uint8)
    adjustment = int(loo)
    for key, idx in frame.groupby(["src_type", "edge_type_index", "dst_type"]).indices.items():
        ref, level = refs.triplets.get(key), 0
        if ref is None or len(ref) - adjustment < refs.min_triplet:
            ref, level = refs.pairs.get((key[0], key[2])), 1
            if ref is None or len(ref) - adjustment < refs.min_pair:
                ref, level = refs.global_values, 2
        ge = len(ref) - np.searchsorted(ref, values[idx], side="left")
        p[idx] = (1. + ge - adjustment) / (len(ref) + 1. - adjustment)
        levels[idx] = level
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("校准概率非法")
    return p, -np.log(np.maximum(p, 1e-12)), levels


class Equivalence:
    def __init__(self):
        self.rows = self.p_mismatches = self.score_mismatches = self.level_mismatches = 0
        self.max_p_error = self.max_score_error = 0.
        self.bitwise_p_different = self.bitwise_score_different = 0
    def add(self, p, score, level, saved_p, saved_score, saved_level):
        if not np.isfinite(saved_p).all() or not np.isfinite(saved_score).all():
            raise ValueError("保存的校准结果含非有限数值")
        dp, ds = np.abs(p - saved_p), np.abs(score - saved_score)
        self.rows += len(p)
        self.p_mismatches += int((dp > P_ATOL).sum())
        self.score_mismatches += int((ds > SCORE_ATOL).sum())
        self.level_mismatches += int((level != saved_level).sum())
        self.bitwise_p_different += int((p.view(np.uint64) != saved_p.view(np.uint64)).sum())
        self.bitwise_score_different += int((score.view(np.uint64) != saved_score.view(np.uint64)).sum())
        self.max_p_error = max(self.max_p_error, float(dp.max(initial=0)))
        self.max_score_error = max(self.max_score_error, float(ds.max(initial=0)))
    def finish(self):
        passed = self.p_mismatches == self.score_mismatches == self.level_mismatches == 0
        return dict(vars(self), passed=passed, p_atol=P_ATOL, score_atol=SCORE_ATOL, rtol=0,
                    field_match_rate=(1 - (self.p_mismatches + self.score_mismatches + self.level_mismatches) / (3 * self.rows)) if self.rows else None)


def parse_saved(rows):
    if any(len(r) != 23 for r in rows):
        raise ValueError("完整校准记录必须有23列")
    arr = np.asarray(rows, dtype=object)
    p, score = arr[:, 20].astype(float), arr[:, 21].astype(float)
    if not np.isfinite(p).all() or not np.isfinite(score).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("已保存尾概率或分数非法")
    codes = pd.Categorical(arr[:, 22], categories=list(LEVELS)).codes
    if (codes < 0).any():
        raise ValueError("已保存校准层级非法")
    return p, score, codes.astype(np.uint8)


class RawSource:
    def __init__(self, audit, chunk_size):
        self.audit, self.chunk_size, self.evidence = audit, chunk_size, []
    def chunks(self):
        previous_id = previous_time = None
        total = 0
        for index, expected in enumerate(self.audit["windows"], 1):
            stream = CSVStream(expected["path"], RAW_HEADER)
            first_id = last_id = None
            while True:
                rows = stream.take(self.chunk_size)
                if not rows:
                    break
                frame = numeric_frame(rows)
                ids, times = frame.event_index.to_numpy(), frame.time.to_numpy()
                if np.any(np.diff(ids) != 1) or (previous_id is not None and ids[0] != previous_id + 1):
                    raise ValueError("原始事件ID不连续、重复或顺序错误")
                if np.any(np.diff(times) < 0) or (previous_time is not None and times[0] < previous_time):
                    raise ValueError("原始事件时间倒序")
                if first_id is None:
                    first_id = int(ids[0])
                last_id, previous_id, previous_time = int(ids[-1]), int(ids[-1]), int(times[-1])
                total += len(rows)
                yield rows, frame
            evidence = stream.finish()
            if evidence["sha256"] != expected["sha256"] or evidence["rows"] != expected["expected_events"] or not evidence["ends_newline"]:
                raise ValueError("原始文件哈希、数量或格式与审计不一致")
            if first_id != expected["first_id"] or last_id != expected["last_id"] or first_id != expected["expected_global_offset"]:
                raise ValueError("原始窗口ID边界不匹配")
            self.evidence.append(evidence)
            if index % 20 == 0:
                print(f"原始窗口 {index}/{len(self.audit['windows'])}，事件 {total:,}", flush=True)
        if total != self.audit["expected_events"]:
            raise ValueError("原始全集事件数量错误")


def load_pickle(path):
    raw = HashReader(path)
    before = fingerprint(path)
    with io.BufferedReader(raw) as f:
        value = pickle.load(f)
        if f.read(1):
            raise ValueError("pickle对象结束后仍有额外字节")
        evidence = dict(before, sha256=raw.digest.hexdigest(), eof_complete=True)
    if before != fingerprint(path):
        raise ValueError("pickle输入在读取期间改变")
    return value, evidence


def validate_calibrator(cal, refs, frame):
    if not isinstance(cal, HierarchicalRelationCalibrator) or not cal._fitted:
        raise ValueError("原校准器类或拟合状态非法")
    if (cal.method, cal.min_triplet_samples, cal.min_type_pair_samples, cal.epsilon) != ("hierarchical_relation", 100, 200, 1e-12):
        raise ValueError("原校准器参数与冻结正式算法不一致")
    if not np.array_equal(cal.global_scores, refs.global_values):
        raise ValueError("原校准器global参考与原始验证数据重拟合结果不同")
    for original, fresh in [(cal.triplet_scores, refs.triplets), (cal.type_pair_scores, refs.pairs)]:
        if original.keys() != fresh.keys() or any(not np.array_equal(original[k], fresh[k]) for k in fresh):
            raise ValueError("原校准器分层参考与验证重拟合结果不同")
    if len(cal._fit_events) != len(frame):
        raise ValueError("原校准器LOO拟合事件数不同")
    for start in range(0, len(frame), 100000):
        events = cal._fit_events[start:start + 100000]
        part = frame.iloc[start:start + 100000]
        if not np.array_equal(np.fromiter((e.score for e in events), float), part.score_raw.to_numpy()):
            raise ValueError("原校准器拟合事件分数或顺序不匹配")
        for axis, field in enumerate(["src_type", "edge_type_index", "dst_type"]):
            if not np.array_equal(np.fromiter((e.triplet_key[axis] for e in events), np.int64), part[field].to_numpy()):
                raise ValueError("原校准器拟合事件关系或顺序不匹配")
        if any(e.type_pair_key != (e.triplet_key[0], e.triplet_key[2]) for e in events):
            raise ValueError("原校准器事件类型对与三元组不匹配")
    return {"passed": True, "parameters_exact": True, "all_reference_arrays_exact": True,
            "fit_event_sequence_exact": True, "fit_events": len(frame)}


def make_scores(bank):
    mask = bank.counts > 0
    ids = bank.ids[mask].copy()
    # 正式协议只包含观察到的节点，无事件节点不补零、不计入全集。
    scores = {k: bank.scores(k)[mask].copy() for k in KS}
    if any(not np.isfinite(a).all() for a in scores.values()):
        raise ValueError("观察到的节点产生非有限分数")
    return ids, scores, bank.counts[mask].copy()


def freeze_predictions(path, ids, scores, thresholds):
    predictions = {k: (scores[k] > thresholds[k]).astype(np.int8) for k in KS}
    for array in [ids, *scores.values(), *predictions.values()]:
        array.flags.writeable = False
    payload = {"stage": "全部节点预测已冻结，尚未读取测试攻击标签", "test_labels_read": False,
               "node_count": len(ids), "node_ids_sha256": digest_arrays(ids),
               "schemes": {str(k): {"k": k, "q": Q, "threshold": thresholds[k],
                    "predicted_positive": int(predictions[k].sum()), "scores_sha256": digest_arrays(scores[k]),
                    "predictions_sha256": digest_arrays(predictions[k])} for k in KS}}
    write_json(path, payload)
    return MappingProxyType(predictions)


def require_seal(path, ids, scores, predictions):
    payload = json.loads(Path(path).read_text())
    if payload["test_labels_read"] or payload["node_ids_sha256"] != digest_arrays(ids):
        raise ValueError("预测冻结摘要非法")
    for k in KS:
        if scores[k].flags.writeable or predictions[k].flags.writeable or payload['schemes'][str(k)]['predictions_sha256'] != digest_arrays(predictions[k]) or payload['schemes'][str(k)]['scores_sha256'] != digest_arrays(scores[k]):
            raise ValueError("预测未冻结或冻结后改变")


def map_ground_truth(gt_rows, uuid_map, inverse_map, cached_gt, observed_ids):
    # 映射按UUID重建，绝不采用GT CSV中历史整数ID列。
    all_ids = np.fromiter((int(v) for v in uuid_map.values()), np.int64)
    if len(np.unique(all_ids)) != len(all_ids) or len(inverse_map) != len(uuid_map):
        raise ValueError("UUID映射不是双射")
    normalized_inverse = {int(k): v for k, v in inverse_map.items()}
    if len(normalized_inverse) != len(inverse_map) or any(normalized_inverse.get(int(v)) != u for u, v in uuid_map.items()):
        raise ValueError("UUID正反映射不一致")
    if not np.isin(observed_ids, all_ids).all():
        raise ValueError("测试观察节点没有完整UUID映射")
    mapped, uuid_by_gt = set(), {}
    for row in gt_rows:
        if not row or row[0] not in uuid_map:
            raise ValueError("GT UUID缺失映射")
        nid = int(uuid_map[row[0]])
        if normalized_inverse.get(nid) != row[0]:
            raise ValueError("GT UUID反向映射错误")
        mapped.add(nid); uuid_by_gt[nid] = row[0]
    if mapped != set(map(int, cached_gt)):
        raise ValueError("GT UUID重映射与正式缓存不一致")
    truth = np.isin(observed_ids, list(mapped)).astype(np.int8)
    return truth, mapped, uuid_by_gt


def load_labels_after_seal(seal_path, ids, scores, predictions, metadata, gt_path):
    require_seal(seal_path, ids, scores, predictions)
    # 攻击标签首次读取点；调用前必须有两方案完整预测数组及摘要哈希。
    import csv
    with gt_path.open(newline="") as f:
        gt_rows = list(csv.reader(f))
    cached, cache_evidence = load_pickle(metadata / "ground_truth_nodes.pkl")
    uuid_map, uuid_evidence = load_pickle(metadata / "uuid_to_node_id.pkl")
    inverse, inverse_evidence = load_pickle(metadata / "node_id_to_uuid.pkl")
    truth, mapped, uuid_by_gt = map_ground_truth(gt_rows, uuid_map, inverse, cached, ids)
    attack_map, attack_evidence = load_pickle(metadata / "attack_to_nodes.pkl")
    if set().union(*(set(map(int, v)) for v in attack_map.values())) != mapped:
        raise ValueError("攻击场景缓存与GT不一致")
    evidence = {"passed": True, "test_nodes": len(ids), "test_positive": int(truth.sum()),
                "test_negative": int((truth == 0).sum()), "gt_nodes": len(mapped),
                "gt_nodes_outside_test": sorted(mapped - set(map(int, ids))),
                "uuid_bijection_all_nodes_passed": True, "gt_uuid_cache_exact": True,
                "sources": [cache_evidence, uuid_evidence, inverse_evidence, attack_evidence,
                            dict(fingerprint(gt_path), sha256=sha(gt_path))]}
    return truth, mapped, uuid_by_gt, evidence


def overlap(ids, truth, predictions, label):
    a = ids[(truth == label) & (predictions[5] == 1)]
    b = ids[(truth == label) & (predictions[1] == 1)]
    groups = {"both": np.intersect1d(a, b), "baseline_only": np.setdiff1d(a, b), "k1_only": np.setdiff1d(b, a)}
    return {key: {"count": len(value), "ids_sha256": digest_arrays(value),
                  "node_ids": value.tolist() if label == 1 or len(value) <= 1000 else value[:30].tolist(),
                  "ids_list_complete": bool(label == 1 or len(value) <= 1000)} for key, value in groups.items()}


def distributions(scores):
    return {"nodes": len(scores), "quantiles": dict(zip(["min", "p50", "p90", "p99", "p999", "max"], np.quantile(scores, [0, .5, .9, .99, .999, 1]).tolist()))} if len(scores) else {"nodes": 0, "quantiles": None}


def main(args):
    started = time.perf_counter(); out, run = args.output, args.run.resolve(strict=True)
    if run.name != "seed_0" or run.parents[2].name != "THEIA_E5":
        raise ValueError("只允许THEIA_E5 seed0")
    if shutil.disk_usage(out.parent).free < 1024**3 or os.stat(out.parent).st_dev == os.stat(run).st_dev:
        raise ValueError("系统盘安全余量不足或输出指向数据盘")
    audits = {s: json.loads((args.audit / f'{s}_integrity.json').read_text()) for s in ['val', 'test']}
    old_protocol = json.loads((args.audit / 'protocol.json').read_text())
    config = yaml.safe_load((run / 'config_resolved.yml').read_text())
    if config['calibration'] != dict(method='hierarchical_relation', min_triplet_samples=100, min_type_pair_samples=200, epsilon=1e-12):
        raise ValueError("正式校准参数不一致")
    if config['node_aggregation'] != dict(method='topk_mean', topk=5, include_dst=True, score_field='score_calibrated') or config['node_threshold'] != dict(method='validation_quantile', quantile=Q):
        raise ValueError("正式原方案参数不一致")
    selection = json.loads((run / 'checkpoint_selection.json').read_text())
    if selection['selected_epoch'] != 'model_epoch_6' or min(selection['per_epoch_validation_objective'], key=selection['per_epoch_validation_objective'].get) != 'model_epoch_6':
        raise ValueError("checkpoint选择不匹配")
    for expected in old_protocol['sources']:
        if sha(expected['path']) != expected['sha256']:
            raise ValueError("原运行身份或窗口manifest与审计不同")
    metadata_manifest = json.loads((args.metadata / 'dataset_manifest.json').read_text())
    if metadata_manifest['dataset'] != 'THEIA_E5' or any(metadata_manifest[k] != config['dataset'][k] for k in ['train_files', 'val_files', 'test_files']):
        raise ValueError("元数据集身份不一致")
    manifest_path = next(Path(p['path']) for p in old_protocol['sources'] if Path(p['path']).name.startswith('manifest__'))
    manifest = json.loads(manifest_path.read_text())
    capacity = int(manifest['max_node']) + 1
    input_paths = [Path(w['path']) for a in audits.values() for w in a['windows']] + [Path(a['calibrated_file']['path']) for a in audits.values()]
    input_paths += [run/'evaluation_results/calibration/model_epoch_6/calibrator.pkl', *[Path(p['path']) for p in old_protocol['sources']]]
    input_paths += [args.metadata / name for name in ['ground_truth_nodes.pkl', 'uuid_to_node_id.pkl', 'node_id_to_uuid.pkl', 'attack_to_nodes.pkl', 'dataset_manifest.json']] + [args.ground_truth]
    before = [fingerprint(p) for p in input_paths]
    out.mkdir(exist_ok=False)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "ks": KS, "q": Q, "include_dst": True,
                "score": "原joint score，原hierarchical_relation校准", "prediction_rule": "score > threshold",
                "node_universe": "测试事件的源/目标节点并集；无事件节点不纳入评价，与正式流水线一致",
                "metrics": "复用mstc.metrics，AP为average_precision_score，排名指标用全部节点连续分数",
                "comparison_atol": {"empirical_p": P_ATOL, "score_calibrated": SCORE_ATOL, "rtol": 0},
                "selection": "只比较预先冻结的两个方案，无搜索或测试选参", "test_labels_read": False,
                "script_sha256": sha(__file__), "git_head": subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                "reused_sources": {str(p): sha(p) for p in [Path(__file__).with_name('analyze_e3_postprocessing.py'), Path(__file__).with_name('audit_e5_prediction_integrity.py'), Path(__file__).resolve().parents[1]/'src/mstc/metrics.py']},
                "inputs": before, "chunk_size": args.chunk_size}
    write_json(out/'protocol.json', protocol)
    print('参数和指标定义已冻结；读取原校准器及完整验证分数', flush=True)
    cal_path = run/'evaluation_results/calibration/model_epoch_6/calibrator.pkl'
    original = None
    try:
        original, pickle_evidence = load_pickle(cal_path)
    except Exception as exc:
        pickle_evidence = {'load_error': repr(exc), 'fallback': '从完整验证原始分数按原算法重新拟合，并要求重现保存结果'}
    validation_source = RawSource(audits['val'], args.chunk_size)
    saved_val = CSVStream(audits['val']['calibrated_file']['path'], CAL_HEADER)
    parts = []; original_payload_rows = 0
    for rows, frame in validation_source.chunks():
        saved_rows = saved_val.take(len(rows))
        if len(saved_rows) != len(rows) or any(a != b[:20] for a,b in zip(rows, saved_rows)):
            raise ValueError('验证原始20字段或数量与保存校准表不一致')
        p, score, level = parse_saved(saved_rows)
        frame['saved_p'], frame['saved_score'], frame['saved_level'] = p, score, level
        parts.append(frame); original_payload_rows += len(rows)
    if saved_val.take(1):
        raise ValueError('验证校准表存在多余记录')
    val_file_evidence = saved_val.finish()
    if val_file_evidence['sha256'] != audits['val']['calibrated_file']['sha256']:
        raise ValueError('验证校准表哈希改变')
    val = pd.concat(parts, ignore_index=True); del parts, frame, rows, saved_rows; gc.collect()
    refs = References(val, val.score_raw.to_numpy())
    if original is not None:
        try:
            pickle_evidence['validation_identity'] = validate_calibrator(original, refs, val)
        except Exception as exc:
            pickle_evidence['validation_identity'] = {'passed': False, 'error': repr(exc), 'fallback': '使用验证原始数据重新拟合的相同算法'}
        del original; gc.collect()
    val_ids = np.union1d(val.srcnode.to_numpy(), val.dstnode.to_numpy())
    val_bank = TopK(val_ids, maximum=5); val_equivalence = Equivalence()
    for start in range(0, len(val), args.chunk_size):
        part = val.iloc[start:start+args.chunk_size]
        p, score, level = query_full(refs, part, loo=True)
        val_equivalence.add(p, score, level, part.saved_p.to_numpy(), part.saved_score.to_numpy(), part.saved_level.to_numpy())
        val_bank.update(part.srcnode.to_numpy(), score); val_bank.update(part.dstnode.to_numpy(), score)
    val_check = val_equivalence.finish()
    write_json(out/'validation_reconstruction.json', {'calibrator_pickle': pickle_evidence, 'equivalence': val_check,
        'original_payload_rows_exact': original_payload_rows, 'source': val_file_evidence})
    if not val_check['passed'] or val_check['rows'] != audits['val']['expected_events']:
        raise ValueError('验证校准重建不等价，禁止继续')
    _, val_scores, val_counts = make_scores(val_bank)
    thresholds = {k: compute_node_threshold(val_scores[k], quantile=Q) for k in KS}
    write_json(out/'frozen_thresholds.json', {'thresholds': thresholds, 'q':Q, 'validation_nodes':len(val_ids), 'test_labels_read':False})
    print(f'验证重现通过：{val_check}; 阈值已冻结 {thresholds}', flush=True)
    del val, val_bank, part; gc.collect()
    test_bank = TopK(np.arange(capacity, dtype=np.int64), maximum=5)
    test_source = RawSource(audits['test'], args.chunk_size)
    saved_test = CSVStream(audits['test']['calibrated_file']['path'], CAL_HEADER)
    test_equivalence = Equivalence(); processed = malformed = 0; level_counts = Counter()
    expected_bad = audits['test']['malformed_examples'][0]
    for rows, frame in test_source.chunks():
        if frame.srcnode.max() >= capacity or frame.dstnode.max() >= capacity:
            raise ValueError('节点ID超出原数据manifest范围')
        p, score, level = query_full(refs, frame)
        saved_rows = saved_test.take(len(rows)); complete_rows = []; complete_idx = []
        for offset, saved_row in enumerate(saved_rows):
            if len(saved_row) != 23:
                malformed += 1
                if malformed != 1 or processed + offset + 1 != expected_bad['record_ordinal'] or len(saved_row) != expected_bad['fields'] or saved_row[0] != expected_bad['event_index']:
                    raise ValueError('发现未在审计中确认的校准记录损坏')
                if saved_row[:-1] != rows[offset][:len(saved_row)-1] or not rows[offset][len(saved_row)-1].startswith(saved_row[-1]):
                    raise ValueError('已知截断记录并非对应原始事件的前缀')
                continue
            if rows[offset] != saved_row[:20]:
                raise ValueError('测试校准表原始20字段或行顺序不匹配')
            complete_rows.append(saved_row); complete_idx.append(offset)
        if complete_rows:
            sp, ss, sl = parse_saved(complete_rows); idx = np.asarray(complete_idx, dtype=np.int64)
            test_equivalence.add(p[idx], score[idx], level[idx], sp, ss, sl)
        test_bank.update(frame.srcnode.to_numpy(), score); test_bank.update(frame.dstnode.to_numpy(), score)
        processed += len(rows); level_counts.update(map(int, level))
    if saved_test.take(1):
        raise ValueError('原始全集以外存在校准记录')
    test_file_evidence = saved_test.finish(); test_check = test_equivalence.finish()
    if test_file_evidence['sha256'] != audits['test']['calibrated_file']['sha256'] or not test_check['passed'] or test_check['rows'] != audits['test']['calibrated_complete_records'] or malformed != 1:
        raise ValueError('测试前缀重建不等价或保存记录覆盖改变')
    write_json(out/'test_reconstruction.json', {'equivalence': test_check, 'raw_events_reconstructed': processed,
        'newly_reconstructed_complete_records': processed-test_check['rows'], 'known_malformed_records':malformed,
        'levels':{LEVELS[int(k)]:v for k,v in level_counts.items()}, 'source':test_file_evidence,
        'raw_original_payload_exact_prefix_records':test_check['rows']})
    if before != [fingerprint(p) for p in input_paths]:
        raise ValueError('来源文件在恢复期间改变')
    ids, scores, counts = make_scores(test_bank)
    del test_bank, frame, rows, complete_rows; gc.collect()
    seal_path = out/'frozen_predictions.json'
    predictions = freeze_predictions(seal_path, ids, scores, thresholds)
    print('全部节点预测已冻结；现在首次读取攻击标签并验收UUID映射', flush=True)
    truth, gt, gt_uuid, mapping = load_labels_after_seal(seal_path, ids, scores, predictions, args.metadata, args.ground_truth)
    metrics, drift, frequency = {}, {}, {}
    for k in KS:
        metrics[k] = dict(compute_classification_metrics(truth, predictions[k], scores[k]),
                          threshold=thresholds[k], predicted_positive=int(predictions[k].sum()))
        val_tail = float((val_scores[k] > thresholds[k]).mean())
        drift[k] = {'validation_assumed_normal': distributions(val_scores[k]),
                    'validation_excluding_future_gt_identities': distributions(val_scores[k][~np.isin(val_ids,list(gt))]),
                    'test_benign':distributions(scores[k][truth==0]), 'validation_exceedance':val_tail,
                    'test_benign_exceedance': metrics[k]['fpr'], 'test_fpr_over_val_exceedance':metrics[k]['fpr']/val_tail if val_tail else None}
        frequency[k] = []
        for low,high in [(1,5),(6,10),(11,100),(101,1000),(1001,10**12)]:
            benign = (counts>=low)&(counts<=high)&(truth==0)
            fp = int((benign&(predictions[k]==1)).sum()); n = int(benign.sum())
            frequency[k].append({'min_events':low,'max_events':high,'benign_nodes':n,'fp':fp,'fpr':fp/n if n else None,
                                 'share_of_all_fp':fp/metrics[k]['fp'] if metrics[k]['fp'] else None})
    attack_overlap, fp_overlap = overlap(ids,truth,predictions,1), overlap(ids,truth,predictions,0)
    for group in attack_overlap.values():
        group['uuids'] = [gt_uuid[int(n)] for n in group['node_ids']]
    require_seal(seal_path, ids, scores, predictions)
    if before != [fingerprint(p) for p in input_paths]:
        raise ValueError('最终来源文件身份改变')
    flags = {'E5_RAW_SCORES_COMPLETE':'YES','E5_CALIBRATION_RECONSTRUCTED':'YES','E5_CALIBRATION_EQUIVALENCE':'PASS',
             'E5_NODE_MAPPING':'PASS','E5_BASELINE_COMPUTED':'YES','E5_FROZEN_K1_COMPUTED':'YES',
             'TEST_LEAKAGE_CHECK':'PASS','DISK_SAFETY':'PASS','RETRAIN_REQUIRED':'NO'}
    summary = {'flags':flags,'metrics':metrics,'node_mapping':mapping,'validation_equivalence':val_check,'test_prefix_equivalence':test_check,
               'calibrator_pickle':pickle_evidence,'attack_overlap':attack_overlap,'false_positive_overlap':fp_overlap,
               'frequency_strata':frequency,'score_distribution_drift':drift,'test_event_count':processed,'validation_event_count':audits['val']['expected_events'],
               'excluded_no_test_event_nodes_in_capacity':capacity-len(ids),'test_nodes_sha256':digest_arrays(ids),
               'test_labels_only_after_predictions_frozen':True,'inputs_unchanged':True,
               'protocol_limit':'E3为事后探索；E5仅按已冻结方案作单seed跨数据集比较，不能替代多seed稳定性和独立重复确认',
               'resource':{'elapsed_seconds':time.perf_counter()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           'root_free_bytes':shutil.disk_usage(out).free,'data_disk_free_bytes':shutil.disk_usage(run).free}}
    write_json(out/'summary.json',summary)
    print(json.dumps({'flags':flags,'metrics':metrics,'node_mapping':{k:v for k,v in mapping.items() if k!='sources'},'resource':summary['resource']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,default=Path(__file__).resolve().parents[1]/'Ground_Truth/darpa/darpa/E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chunk-size',type=int,default=20000)
    args=parser.parse_args()
    if args.chunk_size<=0:parser.error('chunk-size必须为正数')
    try:
        main(args)
    except Exception as exc:
        if args.output.is_dir() and not (args.output/'summary.json').exists() and not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json',{'error':repr(exc),'metrics_valid':False,'note':'校验或运行失败，禁止报告未经验证的指标'})
        raise
