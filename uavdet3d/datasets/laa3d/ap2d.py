import numpy as np
from typing import List, Tuple, Dict, Any


def calculate_iou_2d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate 2D IoU between two bounding boxes.

    Args:
        box1, box2: Bounding boxes in format (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Check if there's an intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    # Calculate intersection area
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union = area1 + area2 - intersection

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union


def match_predictions_to_gt(pred_boxes: np.ndarray,
                            pred_scores: np.ndarray,
                            gt_boxes: np.ndarray,
                            iou_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to ground truth boxes using IoU threshold.

    Args:
        pred_boxes: Predicted boxes (N, 4) in format (x1, y1, x2, y2)
        pred_scores: Prediction confidence scores (N,)
        gt_boxes: Ground truth boxes (M, 4) in format (x1, y1, x2, y2)
        iou_threshold: IoU threshold for positive matches

    Returns:
        Tuple of (true_positives, false_positives) boolean arrays
    """
    num_preds = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_preds == 0:
        return np.array([]), np.array([])

    if num_gt == 0:
        return np.zeros(num_preds, dtype=bool), np.ones(num_preds, dtype=bool)

    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    sorted_pred_boxes = pred_boxes[sorted_indices]

    # Calculate IoU matrix
    iou_matrix = np.zeros((num_preds, num_gt))
    for i in range(num_preds):
        for j in range(num_gt):
            iou_matrix[i, j] = calculate_iou_2d(sorted_pred_boxes[i], gt_boxes[j])

    # Greedy matching: assign each prediction to best available GT
    gt_matched = np.zeros(num_gt, dtype=bool)
    true_positives = np.zeros(num_preds, dtype=bool)
    false_positives = np.zeros(num_preds, dtype=bool)

    for i in range(num_preds):
        # Find best IoU with unmatched GT boxes
        best_iou = 0
        best_gt_idx = -1

        for j in range(num_gt):
            if not gt_matched[j] and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_gt_idx = j

        # Check if match meets threshold
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives[i] = True
            gt_matched[best_gt_idx] = True
        else:
            false_positives[i] = True

    # Restore original order
    original_order = np.argsort(sorted_indices)
    true_positives = true_positives[original_order]
    false_positives = false_positives[original_order]

    return true_positives, false_positives


def calculate_precision_recall_curve(true_positives: np.ndarray,
                                     false_positives: np.ndarray,
                                     num_ground_truths: int,
                                     pred_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate precision-recall curve from TP/FP arrays.

    Args:
        true_positives: Boolean array indicating true positives
        false_positives: Boolean array indicating false positives
        num_ground_truths: Total number of ground truth boxes
        pred_scores: Prediction confidence scores

    Returns:
        Tuple of (precision, recall) arrays
    """
    if len(true_positives) == 0:
        return np.array([1.0]), np.array([0.0])

    # Sort by confidence score (descending)
    sorted_indices = np.argsort(pred_scores)[::-1]
    tp_sorted = true_positives[sorted_indices]
    fp_sorted = false_positives[sorted_indices]

    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(fp_sorted)

    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_ground_truths if num_ground_truths > 0 else tp_cumsum * 0

    # Add starting point (precision=1, recall=0)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    return precision, recall


def interpolate_precision_recall(precision: np.ndarray,
                                 recall: np.ndarray,
                                 recall_points: np.ndarray) -> np.ndarray:
    """
    Interpolate precision at specified recall points using 11-point interpolation.

    Args:
        precision: Precision values
        recall: Recall values
        recall_points: Recall points to interpolate at

    Returns:
        Interpolated precision values
    """
    interpolated_precision = np.zeros(len(recall_points))

    for i, r in enumerate(recall_points):
        # Find precisions for recalls >= r
        mask = recall >= r
        if np.any(mask):
            interpolated_precision[i] = np.max(precision[mask])
        else:
            interpolated_precision[i] = 0.0

    return interpolated_precision


def calculate_ap_single_threshold(pred_boxes: np.ndarray,
                                  pred_scores: np.ndarray,
                                  pred_frame_ids: np.ndarray,
                                  gt_boxes: np.ndarray,
                                  gt_frame_ids: np.ndarray,
                                  iou_threshold: float,
                                  recall_number: int = 41) -> float:
    """
    Calculate AP for a single IoU threshold.

    Args:
        pred_boxes: Predicted boxes (N, 4)
        pred_scores: Prediction confidence scores (N,)
        pred_frame_ids: Frame IDs for predictions (N,)
        gt_boxes: Ground truth boxes (M, 4)
        gt_frame_ids: Frame IDs for ground truth (M,)
        iou_threshold: IoU threshold for matching
        recall_number: Number of recall points for interpolation

    Returns:
        Average Precision value
    """
    # Get unique frame IDs
    unique_frames = np.unique(np.concatenate([pred_frame_ids, gt_frame_ids]))

    all_tp = []
    all_fp = []
    all_scores = []
    total_gt = 0

    # Process each frame separately
    for frame_id in unique_frames:
        # Get predictions and GT for this frame
        pred_mask = pred_frame_ids == frame_id
        gt_mask = gt_frame_ids == frame_id

        frame_pred_boxes = pred_boxes[pred_mask]
        frame_pred_scores = pred_scores[pred_mask]
        frame_gt_boxes = gt_boxes[gt_mask]

        total_gt += len(frame_gt_boxes)

        if len(frame_pred_boxes) == 0:
            continue

        # Match predictions to GT for this frame
        tp, fp = match_predictions_to_gt(frame_pred_boxes,
                                         frame_pred_scores,
                                         frame_gt_boxes,
                                         iou_threshold)

        all_tp.extend(tp)
        all_fp.extend(fp)
        all_scores.extend(frame_pred_scores)

    if len(all_tp) == 0:
        return 0.0, []

    all_tp = np.array(all_tp)
    all_fp = np.array(all_fp)
    all_scores = np.array(all_scores)

    # Calculate precision-recall curve
    precision, recall = calculate_precision_recall_curve(all_tp, all_fp, total_gt, all_scores)

    # Interpolate at standard recall points
    recall_points = np.linspace(0, 1, recall_number)
    interpolated_precision = interpolate_precision_recall(precision, recall, recall_points)

    # Calculate AP as mean of interpolated precisions
    ap = np.mean(interpolated_precision)

    return ap, interpolated_precision


def calculate_object_detection_2dap(pred_boxes: List[List[float]],
                                    pred_scores: List[float],
                                    pred_frame_ids: List[int],
                                    gt_boxes: List[List[float]],
                                    gt_frame_ids: List[int],
                                    matching_thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                    recall_number: int = 41) -> Dict[str, Any]:
    """
    Calculate object detection Average Precision (AP) with IoU-based matching.

    Args:
        pred_boxes: List of predicted bounding boxes, each in format [x1, y1, x2, y2]
        pred_scores: List of prediction confidence scores
        pred_frame_ids: List of frame IDs for predictions
        gt_boxes: List of ground truth bounding boxes, each in format [x1, y1, x2, y2]
        gt_frame_ids: List of frame IDs for ground truth
        matching_thresholds: List of IoU thresholds for matching
        recall_number: Number of recall points for interpolation

    Returns:
        Dictionary containing AP results
    """
    # Convert inputs to numpy arrays
    pred_boxes = np.array(pred_boxes)
    pred_scores = np.array(pred_scores)
    pred_frame_ids = np.array(pred_frame_ids)
    gt_boxes = np.array(gt_boxes)
    gt_frame_ids = np.array(gt_frame_ids)

    # Validate inputs
    assert len(pred_boxes) == len(pred_scores) == len(pred_frame_ids), \
        "Prediction arrays must have same length"
    assert len(gt_boxes) == len(gt_frame_ids), \
        "Ground truth arrays must have same length"
    assert pred_boxes.shape[1] == 4, "Bounding boxes must have 4 coordinates"
    assert gt_boxes.shape[1] == 4, "Bounding boxes must have 4 coordinates"

    # Calculate AP for each IoU threshold
    ap_results = {}
    ap_values = []

    detailed_ap_values = []

    for threshold in matching_thresholds:
        ap, all_values = calculate_ap_single_threshold(pred_boxes, pred_scores, pred_frame_ids,
                                                       gt_boxes, gt_frame_ids, threshold, recall_number)

        ap_results[f'AP@{threshold:.1f}'] = ap
        ap_values.append(ap)
        detailed_ap_values.append(all_values)

    # Calculate mean AP across all thresholds
    mean_ap = np.mean(ap_values)

    return mean_ap, ap_values, detailed_ap_values
