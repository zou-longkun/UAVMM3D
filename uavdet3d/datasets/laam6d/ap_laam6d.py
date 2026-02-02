import numpy as np
from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict

from numpy import ndarray


def calculate_center_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between centers of two 3D boxes.

    Args:
        box1, box2: (9, 3) format with 9 points, each with x,y,z coordinates
        The first point (index 0) is used as the center point.

    Returns:
        Euclidean distance between centers
    """
    center1 = box1[0]
    center2 = box2[0]
    return np.linalg.norm(center1 - center2)


def match_boxes_by_distance(pred_boxes: np.ndarray,
                            gt_boxes: np.ndarray,
                            threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match predicted boxes to ground truth boxes based on center distance.

    Args:
        pred_boxes: (N, 9, 3) array of predicted boxes with 9 points, each with x,y,z coordinates
        gt_boxes: (M, 9, 3) array of ground truth boxes with 9 points, each with x,y,z coordinates
        threshold: Maximum distance for matching

    Returns:
        matches: (K, 2) array of matched indices [pred_idx, gt_idx]
        unmatched_preds: array of unmatched prediction indices
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(pred_boxes))

    # Calculate distance matrix
    distances = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            distances[i, j] = calculate_center_distance(pred_box, gt_box)

    # Hungarian matching (greedy approximation for simplicity)
    matches = []
    used_gt = set()
    used_pred = set()

    # Sort by distance for greedy matching
    pred_indices, gt_indices = np.where(distances <= threshold)
    if len(pred_indices) > 0:
        valid_distances = distances[pred_indices, gt_indices]
        sorted_indices = np.argsort(valid_distances)

        for idx in sorted_indices:
            pred_idx = pred_indices[idx]
            gt_idx = gt_indices[idx]

            if pred_idx not in used_pred and gt_idx not in used_gt:
                matches.append([pred_idx, gt_idx])
                used_pred.add(pred_idx)
                used_gt.add(gt_idx)

    matches = np.array(matches) if matches else np.empty((0, 2), dtype=int)
    unmatched_preds = np.array([i for i in range(len(pred_boxes)) if i not in used_pred])

    return matches, unmatched_preds


def calculate_ap_for_recall_points(precisions: np.ndarray,
                                   recalls: np.ndarray,
                                   recall_points: np.ndarray) -> Union[
    tuple[int, list[Any]], tuple[ndarray, list[Union[ndarray, int, float, complex]]]]:
    """
    Calculate AP using interpolated precision at specific recall points.

    Args:
        precisions: Array of precision values
        recalls: Array of recall values
        recall_points: Recall points to interpolate at

    Returns:
        Average Precision
    """
    if len(precisions) == 0:
        return 1, []

    # Ensure arrays are sorted by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # Interpolate precision at recall points
    interpolated_precisions = []
    for recall_point in recall_points:
        # Find precisions at recalls >= recall_point
        valid_indices = recalls >= recall_point
        if np.any(valid_indices):
            max_precision = np.max(precisions[valid_indices])
            interpolated_precisions.append(max_precision)
        else:
            interpolated_precisions.append(0.0)

    return np.mean(interpolated_precisions), interpolated_precisions


def calculate_object_detection_3dap(pred_boxes: np.ndarray,
                                    pred_scores: np.ndarray,
                                    pred_frame_id: np.ndarray,
                                    gt_boxes: np.ndarray,
                                    gt_frame_id: np.ndarray,
                                    matching_thresholds: List[float] = [1, 2, 4],
                                    recall_number: int = 40,
                                    detection_distances: List[float] = [100]) -> tuple[
    ndarray, list[Union[float, Any]], list[Any]]:
    """
    Calculate Average Precision for 3D object detection with center distance matching.

    Args:
        pred_boxes: (N, 9, 3) predicted boxes
        pred_scores: (N,) prediction confidence scores
        pred_frame_id: (N,) frame IDs for predictions
        gt_boxes: (M, 9, 3) ground truth boxes
        gt_frame_id: (M,) frame IDs for ground truth
        matching_thresholds: Distance thresholds for matching
        recall_number: Number of recall points for AP calculation
        detection_distances: Maximum detection distances for different difficulty levels

    Returns:
        List of AP values for each detection distance level
    """

    # Convert to numpy arrays if needed
    pred_boxes = np.array(pred_boxes)
    pred_scores = np.array(pred_scores)
    pred_frame_id = np.array(pred_frame_id)
    gt_boxes = np.array(gt_boxes)
    gt_frame_id = np.array(gt_frame_id)

    # Create recall points
    recall_points = np.linspace(0, 1, recall_number)

    # Group data by frame
    frame_ids = np.unique(np.concatenate([pred_frame_id, gt_frame_id]))

    # Organize predictions and ground truth by frame
    pred_by_frame = defaultdict(list)
    gt_by_frame = defaultdict(list)

    for i, frame_id in enumerate(pred_frame_id):
        pred_by_frame[frame_id].append(i)

    for i, frame_id in enumerate(gt_frame_id):
        gt_by_frame[frame_id].append(i)

    # Calculate AP for each detection distance level
    distance_aps = []

    all_values = []

    for max_distance in detection_distances:

        # Calculate AP for each matching threshold
        threshold_aps = []

        precisions_values = []

        for match_threshold in matching_thresholds:
            # Collect all predictions and their match status across frames
            all_scores = []
            all_matches = []
            total_gt_count = 0

            for frame_id in frame_ids:
                pred_indices = pred_by_frame.get(frame_id, [])
                gt_indices = gt_by_frame.get(frame_id, [])

                if not pred_indices:
                    total_gt_count += len(gt_indices)
                    continue

                if not gt_indices:
                    # No ground truth in this frame, all predictions are false positives
                    frame_scores = pred_scores[pred_indices]
                    all_scores.extend(frame_scores)
                    all_matches.extend([False] * len(frame_scores))
                    continue

                # Get boxes for this frame
                frame_pred_boxes = pred_boxes[pred_indices]
                frame_gt_boxes = gt_boxes[gt_indices]
                frame_scores = pred_scores[pred_indices]
                pred_distances = np.linalg.norm(frame_pred_boxes[:,  0], axis=1)
                mask_pred = pred_distances <= max_distance
                frame_pred_boxes = frame_pred_boxes[mask_pred]
                frame_scores = frame_scores[mask_pred]

                # Filter ground truth by detection distance
                gt_distances = np.linalg.norm(frame_gt_boxes[:, 0], axis=1)
                valid_gt_mask = gt_distances <= max_distance
                frame_gt_boxes = frame_gt_boxes[valid_gt_mask]
                total_gt_count += np.sum(valid_gt_mask)

                if len(frame_gt_boxes) == 0:
                    # No valid ground truth in detection range
                    all_scores.extend(frame_scores)
                    all_matches.extend([False] * len(frame_scores))
                    continue

                # Match predictions to ground truth
                matches, unmatched_preds = match_boxes_by_distance(
                    frame_pred_boxes, frame_gt_boxes, match_threshold
                )

                # Mark which predictions are true positives
                is_match = np.zeros(len(frame_pred_boxes), dtype=bool)
                if len(matches) > 0:
                    is_match[matches[:, 0]] = True

                all_scores.extend(frame_scores)
                all_matches.extend(is_match)

            if len(all_scores) == 0 or total_gt_count == 0:
                threshold_aps.append(0.0)
                continue

            # Sort by confidence score (descending)
            all_scores = np.array(all_scores)
            all_matches = np.array(all_matches)
            sorted_indices = np.argsort(-all_scores)
            sorted_matches = all_matches[sorted_indices]

            # Calculate precision and recall
            tp_cumsum = np.cumsum(sorted_matches)
            fp_cumsum = np.cumsum(~sorted_matches)

            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            recalls = tp_cumsum / total_gt_count

            # Calculate AP
            ap, values = calculate_ap_for_recall_points(precisions, recalls, recall_points)

            threshold_aps.append(ap)
            precisions_values.append(values)

        # Average AP across matching thresholds
        distance_ap = np.mean(threshold_aps)
        distance_aps.append(distance_ap)
        all_values.append(precisions_values)

    return distance_aps[0], threshold_aps, all_values[0]
