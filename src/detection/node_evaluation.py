from collections import defaultdict
import json

import torch
import numpy as np
import wandb

from provnet_utils import *
from config import *
from .evaluation_utils import *
from mstc.metrics import (
    compute_attack_detection_rate, compute_classification_metrics, compute_fp_per_million,
)
from labelling import get_GP_of_each_attack


def get_node_predictions(val_tw_path, test_tw_path, cfg):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Loading data from {test_tw_path}...")
    
    thr = get_threshold(val_tw_path, cfg.detection.evaluation.node_evaluation.threshold_method)
    log(f"Threshold: {thr:.3f}")

    node_to_losses = defaultdict(list)
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)
    
    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient='records')
        for line in df:
            srcnode = line['srcnode']
            dstnode = line['dstnode']
            loss = line['loss']
            
            # Scores
            node_to_losses[srcnode].append(loss)
            if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                node_to_losses[dstnode].append(loss)
                
            # If max-val thr is used, we want to keep track when the node with max loss happens
            if loss > node_to_max_loss[srcnode]:
                node_to_max_loss[srcnode] = loss
                node_to_max_loss_tw[srcnode] = tw
            if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
                if loss > node_to_max_loss[dstnode]:
                    node_to_max_loss[dstnode] = loss
                    node_to_max_loss_tw[dstnode] = tw
                    
    use_kmeans = cfg.detection.evaluation.node_evaluation.use_kmeans
    results = defaultdict(dict)
    for node_id, losses in node_to_losses.items():
        pred_score = reduce_losses_to_score(losses, cfg.detection.evaluation.node_evaluation.threshold_method)

        results[node_id]["score"] = pred_score
        results[node_id]["tw_with_max_loss"] = node_to_max_loss_tw.get(node_id, -1)
        results[node_id]["y_true"] = int(node_id in ground_truth_nids)
        
        if use_kmeans: # in this mode, we add the label after
            results[node_id]["y_hat"] = 0
        else:
            results[node_id]["y_hat"] = int(pred_score > thr)
        
    if use_kmeans:
        results = compute_kmeans_labels(results, topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K)
        
    return results

def analyze_false_positives(y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes):
    log(f"Analysis of false positives:")
    fp_indices = [i for i, (true, pred) in enumerate(zip(y_truth, y_preds)) if pred and not true]
    malicious_tws = set(tw_to_malicious_nodes.keys())
    num_fps_in_malicious_tw = 0
    
    for i in fp_indices:
        is_in_malicious_tw = max_val_loss_tw[i] in malicious_tws
        num_fps_in_malicious_tw += int(is_in_malicious_tw)

        log(f"FP node {nodes[i]} -> max loss: {pred_scores[i]:.3f} | max TW: {max_val_loss_tw[i]} "
            f"| is malicious TW: " + (" ✅" if is_in_malicious_tw else " ❌"))
    
    fp_in_malicious_tw_ratio = num_fps_in_malicious_tw / len(fp_indices) if len(fp_indices) > 0 else float("nan")
    log(f"Percentage of FPs present in malicious TWs: {fp_in_malicious_tw_ratio:.3f}")
    return fp_in_malicious_tw_ratio

def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    results = get_node_predictions(val_tw_path, test_tw_path, cfg)
    node_to_path = get_node_to_path_and_type(cfg)

    out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    dor_img_file = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")

    # ------------------------------------------------------------------ #
    # Compute val_mean_edge_loss BEFORE returning stats so it can be used
    # by evaluation.py to select the best epoch (min val loss).
    # ------------------------------------------------------------------ #
    _val_loss_list = []
    for _f in sorted(os.listdir(val_tw_path)):
        _df = pd.read_csv(os.path.join(val_tw_path, _f))
        _val_loss_list.extend(_df["loss"].tolist())
    val_mean_edge_loss = float(np.mean(_val_loss_list)) if _val_loss_list else float("nan")

    log("Analysis of malicious nodes:")
    nodes, y_truth, y_preds, pred_scores, max_val_loss_tw = [], [], [], [], []
    for nid, result in results.items():
        nodes.append(nid)
        score, y_hat, y_true, max_tw = result["score"], result["y_hat"], result["y_true"], result["tw_with_max_loss"]
        y_truth.append(y_true)
        y_preds.append(y_hat)
        pred_scores.append(score)
        max_val_loss_tw.append(max_tw)

        if y_true == 1:
            log(f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:" + (" ✅ " if y_true == y_hat else " ❌ ") + (node_to_path[nid]['path']))

    # Plots the PR curve and scores for mean node loss
    print(f"Saving figures to {out_dir}...")
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_dor_recall_curve(pred_scores, y_truth, dor_img_file)
    plot_simple_scores(pred_scores, y_truth, simple_scores_img_file)
    plot_scores_with_paths(pred_scores, y_truth, nodes, max_val_loss_tw, tw_to_malicious_nodes, scores_img_file, cfg)
    # C8 standard metrics use explicit NaN for undefined edge cases.  Keep the
    # legacy evaluator invocation for its existing logging/auxiliary statistics.
    stats = classifier_evaluation(y_truth, y_preds, pred_scores)
    c8_metrics = compute_classification_metrics(y_truth, y_preds, pred_scores)
    stats.update(c8_metrics)
    stats["fp_per_million"] = compute_fp_per_million(
        int(c8_metrics["fp"]), int(c8_metrics["tn"]) + int(c8_metrics["fp"]),
    )
    attack_to_nodes = get_GP_of_each_attack(cfg)
    stats["attack_detection_rate"] = compute_attack_detection_rate(
        attack_to_nodes, (node for node, prediction in zip(nodes, y_preds) if prediction),
    )

    fp_in_malicious_tw_ratio = analyze_false_positives(y_truth, y_preds, pred_scores, max_val_loss_tw, nodes, tw_to_malicious_nodes)
    stats["fp_in_malicious_tw_ratio"] = fp_in_malicious_tw_ratio
    stats["val_mean_edge_loss"] = val_mean_edge_loss  # used by evaluation.py for best-epoch selection

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")

    runtime_dir = getattr(cfg, "_run_dir", None)
    if not isinstance(runtime_dir, (str, os.PathLike)) or not os.fspath(runtime_dir):
        runtime_dir = os.path.dirname(out_dir)
    try:
        with open(os.path.join(runtime_dir, "runtime.json"), encoding="utf-8") as handle:
            runtime = json.load(handle)
    except (OSError, json.JSONDecodeError):
        runtime = {}
    testing_runtime = runtime.get("testing", {}) if isinstance(runtime, dict) else {}
    model_runtime = runtime.get("model", {}) if isinstance(runtime, dict) else {}
    stats["parameter_count"] = model_runtime.get("parameter_count")
    stats["events_per_second"] = testing_runtime.get("events_per_second")
    stats["peak_gpu_memory_mb"] = testing_runtime.get("peak_gpu_memory_mb")
    stats["peak_cpu_memory_mb"] = testing_runtime.get("peak_cpu_memory_mb")

    torch.save(results, results_file)
    torch.save(stats, stats_file)
    for metrics_path in (
        os.path.join(out_dir, f"metrics_{model_epoch_dir}.json"),
        os.path.join(out_dir, "metrics.json"),
    ):
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, allow_nan=True)

    return stats
