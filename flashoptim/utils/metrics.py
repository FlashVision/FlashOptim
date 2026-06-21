"""Evaluation metrics for model optimization."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_map(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute mean Average Precision (mAP) for object detection.

    Uses IoU-based matching between predicted and ground truth bounding boxes
    with the 11-point interpolation method for AP calculation.

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'.
            boxes are in [x1, y1, x2, y2] format.
        ground_truths: List of ground truth dicts with 'boxes', 'labels'.
        iou_threshold: IoU threshold for matching (default 0.5 for mAP@50).

    Returns:
        mAP score between 0.0 and 1.0.
    """
    if not predictions or not ground_truths:
        return 0.0

    all_labels: set = set()
    for gt in ground_truths:
        labels = gt.get("labels", [])
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        all_labels.update(labels)

    if not all_labels:
        return 0.0

    ap_per_class = []

    for cls in all_labels:
        all_detections = []
        total_gt = 0

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            pred_labels = pred.get("labels", [])
            if isinstance(pred_labels, torch.Tensor):
                pred_labels = pred_labels.tolist()
            pred_boxes = pred.get("boxes", [])
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.tolist()
            pred_scores = pred.get("scores", [])
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.tolist()

            gt_labels = gt.get("labels", [])
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.tolist()
            gt_boxes = gt.get("boxes", [])
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.tolist()

            gt_indices_for_cls = [i for i, lbl in enumerate(gt_labels) if lbl == cls]
            total_gt += len(gt_indices_for_cls)

            for i, label in enumerate(pred_labels):
                if label == cls:
                    score = pred_scores[i] if i < len(pred_scores) else 1.0
                    box = pred_boxes[i] if i < len(pred_boxes) else [0, 0, 0, 0]
                    all_detections.append({
                        "score": score,
                        "box": box,
                        "img_idx": img_idx,
                    })

        if total_gt == 0:
            continue

        all_detections.sort(key=lambda x: x["score"], reverse=True)

        gt_matched: Dict[int, set] = {}
        tp_list = []
        fp_list = []

        for det in all_detections:
            img_idx = det["img_idx"]
            det_box = det["box"]

            gt = ground_truths[img_idx]
            gt_labels = gt.get("labels", [])
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.tolist()
            gt_boxes = gt.get("boxes", [])
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.tolist()

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, (gl, gb) in enumerate(zip(gt_labels, gt_boxes)):
                if gl != cls:
                    continue
                iou = _compute_iou(det_box, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                if img_idx not in gt_matched:
                    gt_matched[img_idx] = set()
                if best_gt_idx not in gt_matched[img_idx]:
                    gt_matched[img_idx].add(best_gt_idx)
                    tp_list.append(1)
                    fp_list.append(0)
                else:
                    tp_list.append(0)
                    fp_list.append(1)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for tp, fp in zip(tp_list, fp_list):
            tp_cumsum += tp
            fp_cumsum += fp
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt
            precisions.append(precision)
            recalls.append(recall)

        ap = _compute_ap_11point(precisions, recalls)
        ap_per_class.append(ap)

    return sum(ap_per_class) / len(ap_per_class) if ap_per_class else 0.0


def _compute_ap_11point(precisions: List[float], recalls: List[float]) -> float:
    """Compute AP using the 11-point interpolation method."""
    if not precisions:
        return 0.0

    ap = 0.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        prec_at_recall = 0.0
        for p, r in zip(precisions, recalls):
            if r >= t:
                prec_at_recall = max(prec_at_recall, p)
        ap += prec_at_recall

    return ap / 11.0


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple = (1,),
) -> List[float]:
    """Compute top-k accuracy.

    Args:
        predictions: Model output logits of shape (N, C).
        targets: Ground truth labels of shape (N,).
        topk: Tuple of k values for top-k accuracy.

    Returns:
        List of top-k accuracy values as percentages.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred_indices = predictions.topk(maxk, dim=1, largest=True, sorted=True)
        pred_indices = pred_indices.t()
        correct = pred_indices.eq(targets.unsqueeze(0).expand_as(pred_indices))

        result = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            result.append((correct_k / batch_size * 100.0).item())
        return result


def compute_compression_ratio(
    original_model: nn.Module,
    optimized_model: nn.Module,
) -> Dict[str, float]:
    """Compute compression metrics between original and optimized models.

    Args:
        original_model: Original (unoptimized) model.
        optimized_model: Optimized model.

    Returns:
        Dictionary with 'param_ratio', 'size_ratio', 'compression_factor'.
    """
    orig_params = sum(p.numel() for p in original_model.parameters())
    opt_params = sum(p.numel() for p in optimized_model.parameters())

    orig_size = sum(
        p.nelement() * p.element_size() for p in original_model.parameters()
    )
    opt_size = sum(
        p.nelement() * p.element_size() for p in optimized_model.parameters()
    )

    return {
        "param_ratio": opt_params / orig_params if orig_params > 0 else 1.0,
        "size_ratio": opt_size / orig_size if orig_size > 0 else 1.0,
        "compression_factor": orig_size / opt_size if opt_size > 0 else 1.0,
        "original_params": orig_params,
        "optimized_params": opt_params,
        "original_size_mb": round(orig_size / (1024 * 1024), 2),
        "optimized_size_mb": round(opt_size / (1024 * 1024), 2),
    }
