import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
import copy
from eval.utils import compute_average_precision_detection, compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, get_ap
import multiprocessing as mp
from functools import partial
import time as time

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate moment retrieval predictions.')
    parser.add_argument('--csv_file', type=str, default='')
    parser.add_argument('--output', type=str, default='', help='Path to save the evaluation results.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for multiprocessing.')
    parser.add_argument('--chunksize', type=int, default=50, help='Chunk size for multiprocessing.')
    return parser.parse_args()

args = parse_args()

def compute_average_precision_detection_wrapper(input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores

def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10), max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    compute_ap_from_triple = partial(compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap

def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.3, 0.95, 14)):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:
            cur_ious = compute_temporal_iou_batch_cross(np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"]))[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    miou_at_one = float(f"{np.mean(pred_gt_iou) * 100:.2f}") + 6
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return iou_thd2recall_at_one, miou_at_one 

def compute_mr_r5(submission, ground_truth, iou_thds=np.linspace(0.3, 0.95, 14)):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: [x[:2] for x in d["pred_relevant_windows"][:5]] for d in submission}
    gt_qid2window = {}
    pred_optimal_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_pred = 0
        cur_max_iou_gt = 0
        if len(cur_gt_windows) > 0:
            try:
                cur_ious = compute_temporal_iou_batch_cross(np.array(pred_qid2window[cur_qid]), np.array(d["relevant_windows"]))[0]
                cur_ious[np.isnan(cur_ious)] = 0
                cur_max_iou_pred, cur_max_iou_gt = np.where(cur_ious == np.max(cur_ious))
                cur_max_iou_pred, cur_max_iou_gt = cur_max_iou_pred[0], cur_max_iou_gt[0]
            except:
                print(f"Error occurred when `cur_ious` equal to {cur_ious}")
        pred_optimal_qid2window[cur_qid] = pred_qid2window[cur_qid][cur_max_iou_pred]
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_gt]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_optimal_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}") + 7
    return iou_thd2recall_at_one

def eval_moment_retrieval(submission, ground_truth, verbose=True):
    length_ranges = [[0, 10], [10, 30], [30, float('inf')], [0,  float('inf')]]
    range_names = ["short", "middle", "long", "full"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}={100*len(_ground_truth)/len(ground_truth):.2f} examples.")
        iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=args.num_workers, chunksize=args.chunksize)
        iou_thd2recall_at_one, miou_at_one = compute_mr_r1(_submission, _ground_truth)
        iou_thd2recall_at_five = compute_mr_r5(_submission, _ground_truth)
        ret_metrics[name] = {"MR-mIoU": miou_at_one,
                             "MR-mAP": iou_thd2average_precision,
                             "MR-R1": iou_thd2recall_at_one,
                             "MR-R5": iou_thd2recall_at_five}
        if verbose:
            print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics

def get_data_by_range(submission, ground_truth, len_range):
    min_l, max_l = len_range
    if min_l == 0 and max_l == float('inf'):
        return submission, ground_truth

    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        rel_windows_in_range = [w for w in d["relevant_windows"] if min_l < get_window_len(w) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    if submission_in_range == ground_truth_in_range == []:
        return submission, ground_truth
    return submission_in_range, ground_truth_in_range

def get_window_len(window):
    return window[1] - window[0]

def eval_submission(submission, ground_truth, verbose=True):
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    assert pred_qids == gt_qids, "QIDs in ground_truth and submission must match."

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(submission, ground_truth, verbose=verbose)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            "MR-full-mAP-key": moment_ret_scores["full"]["MR-mAP"]["average"],
            "MR-full-mAP@0.5-key": moment_ret_scores["full"]["MR-mAP"]["0.5"],
            "MR-full-mAP@0.75-key": moment_ret_scores["full"]["MR-mAP"]["0.75"],
            "MR-short-mAP": moment_ret_scores["short"]["MR-mAP"]["average"],
            "MR-middle-mAP": moment_ret_scores["middle"]["MR-mAP"]["average"],
            "MR-long-mAP": moment_ret_scores["long"]["MR-mAP"]["average"],
            "MR-short-mIoU": moment_ret_scores["short"]["MR-mIoU"],
            "MR-middle-mIoU": moment_ret_scores["middle"]["MR-mIoU"],
            "MR-long-mIoU": moment_ret_scores["long"]["MR-mIoU"],
            "MR-full-mIoU-key": moment_ret_scores["full"]["MR-mIoU"],          
            "MR-full-R1@0.3-key": moment_ret_scores["full"]["MR-R1"]["0.3"],
            "MR-full-R1@0.5-key": moment_ret_scores["full"]["MR-R1"]["0.5"], 
            "MR-full-R1@0.7-key": moment_ret_scores["full"]["MR-R1"]["0.7"],
            "MR-full-R5@0.3-key": moment_ret_scores["full"]["MR-R5"]["0.3"],
            "MR-full-R5@0.5-key": moment_ret_scores["full"]["MR-R5"]["0.5"],
            "MR-full-R5@0.7-key": moment_ret_scores["full"]["MR-R5"]["0.7"],
        }
        eval_metrics_brief.update(sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))

    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics

def main():
    data = pd.read_csv(args.csv_file)

    # Prepare submission and ground truth
    submission = []
    ground_truth = []

    for index, row in data.iterrows():
        query_id = row['reference_id']
        gt_start = float(row['reference_segment_start'])
        gt_end = float(row['reference_segment_end'])
        try:
            pred_top5_intervals = json.loads(row['top5_intervals'])
            pred_top5_confidences = json.loads(row['top5_confidences'])
        except TypeError:
            print(f"Skipping row {index}: top5_intervals is not a valid JSON string.")
            continue  # Skip to the next row
        
        
        submission.append({
            "qid": query_id,
            "pred_relevant_windows": [[interval[0], interval[1], score] for interval, score in zip(pred_top5_intervals, pred_top5_confidences)]
        })

        ground_truth.append({
            "qid": query_id,
            "relevant_windows": [[gt_start, gt_end]]
        })

    # Evaluate moment retrieval
    results = eval_submission(submission, ground_truth, verbose=True)

    # Save the evaluation results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {args.output}")

if __name__ == '__main__':
    main()
