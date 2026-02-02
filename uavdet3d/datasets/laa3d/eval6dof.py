import numpy as np
import pandas as pd
import transforms3d.quaternions as txq
import transforms3d.euler as euler
import copy
from scipy.spatial.transform import Rotation as R

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def eval_key_points_error(annos):
    all_error = []

    for i, each_anno in enumerate(annos):
        key_points_2d = each_anno['key_points_2d']
        gt_pts2d = each_anno['gt_pts2d']

        all_error.append(np.abs(key_points_2d - gt_pts2d).reshape(-1))

    all_error = np.concatenate(all_error)

    return all_error


def euler_angle_error(pred, gt):
    """
    计算弧度制下欧拉角的角度误差。

    参数:
    pred (numpy.ndarray): 预测的欧拉角，形状为 (N, 3)。
    gt (numpy.ndarray): 真实的欧拉角，形状为 (N, 3)。

    返回:
    numpy.ndarray: 每个样本的角度误差，形状为 (N,)。
    """
    diff = pred - gt

    diff = np.arctan2(np.sin(diff), np.cos(diff))

    error = np.linalg.norm(diff, axis=1)

    return error


def euler_angle_error_rad(pred, gt):
    diff = pred - gt

    diff = (diff + np.pi) % (2 * np.pi) - np.pi

    error = np.sum(np.abs(diff), axis=1)
    return error


def limit(ang):
    ang = ang % (2 * np.pi)

    ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

    ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

    return ang


def ang_weight(pred, gt):
    a = np.abs(pred - gt)
    b = 2 * np.pi - np.abs(pred - gt)

    res = np.stack([a, b])

    res = np.min(res, axis=0)

    return res


def hungarian_match_allow_unmatched(x: np.ndarray, y: np.ndarray, unmatched_cost: float = 1e5):
    N, M = x.shape[0], y.shape[0]

    cost = cdist(x, y)

    size = max(N, M)
    padded_cost = np.full((size, size), unmatched_cost)
    padded_cost[:N, :M] = cost

    row_ind, col_ind = linear_sum_assignment(padded_cost)

    x_to_y = np.full(N, -1, dtype=int)

    for r, c in zip(row_ind, col_ind):
        if r < N and c < M and padded_cost[r, c] < unmatched_cost:
            x_to_y[r] = c  # 合法匹配

    return x_to_y


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two 2D bounding boxes.
    Boxes are in the format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area


def match_gt(gt_box9d, pred_boxes9d, gt_box2d, pred_box2d, match_iou=0.1):
    """
    Match predicted 3D boxes to ground-truth 3D boxes based on 2D IoU.

    Args:
        gt_box9d: [M, 9] Ground-truth 3D boxes
        pred_boxes9d: [N, 9] Predicted 3D boxes
        gt_box2d: [M, 4] Ground-truth 2D boxes [x1, y1, x2, y2]
        pred_box2d: [N, 4] Predicted 2D boxes
        match_iou: Minimum IoU threshold for a valid match

    Returns:
        matched_gt_box9d: [K, 9] Matched ground-truth 3D boxes
        matched_pred_boxes9d: [K, 9] Matched predicted 3D boxes
    """
    matched_gt = []
    matched_pred = []
    used_preds = set()

    for i, gt_box in enumerate(gt_box2d):
        best_iou = 0
        best_j = -1
        for j, pred_box in enumerate(pred_box2d):
            if j in used_preds:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= match_iou and best_j != -1:
            matched_gt.append(gt_box9d[i])
            matched_pred.append(pred_boxes9d[best_j])
            used_preds.add(best_j)

    if matched_gt:
        return np.stack(matched_gt), np.stack(matched_pred)
    else:
        return np.zeros((0, 9)), np.zeros((0, 9))


def eval_box6d_error(annos, max_dis=8):
    all_orin_error = []

    all_pos_error = []

    all_size_error = []

    for i, each_anno in enumerate(annos):
        gt_box9d = each_anno['gt_box9d']
        pred_boxes9d = each_anno['pred_box9d']

        gt_box2d = each_anno['gt_box2d']
        pred_box2d = each_anno['pred_box2d']

        if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
            continue

        gt_box9d, pred_boxes9d = match_gt(gt_box9d, pred_boxes9d, gt_box2d, pred_box2d, match_iou=0.1)

        if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
            continue

        pred_xyz = pred_boxes9d[:, 0:3]

        gt_xyz = gt_box9d[:, 0:3]

        pred_angle = pred_boxes9d[:, 6:9]
        gt_angle = gt_box9d[:, 6:9]

        dis_error = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)

        dis_error[dis_error > max_dis] = max_dis

        angle_error = val_rotation_enler(pred_angle, gt_angle)

        size_error = np.mean(np.abs(pred_boxes9d[:, 3:6] - gt_box9d[:, 3:6]), axis=-1)

        all_orin_error.append(angle_error)
        all_pos_error.append(dis_error)
        all_size_error.append(size_error)

    if len(all_orin_error) > 0 and len(all_pos_error) > 0 and len(all_size_error) > 0:

        all_orin_error = np.concatenate(all_orin_error)
        all_pos_error = np.concatenate(all_pos_error)
        all_size_error = np.concatenate(all_size_error)
    else:
        all_orin_error = np.array([0])
        all_pos_error = np.array([0])
        all_size_error = np.array([0])

    return all_orin_error, all_pos_error, all_size_error


def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [4,]
        gt_q: [4,]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted = pred_q
        groundtruth = gt_q
    else:
        predicted = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    d = np.abs(np.dot(groundtruth, predicted))
    d = np.minimum(1.0, np.maximum(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error


def val_rotation_enler(pred_q, gt_q):
    rotation_pred = R.from_euler('xyz', pred_q, degrees=False)
    rotation_gt = R.from_euler('xyz', gt_q, degrees=False)

    rotation_pred = rotation_pred.as_quat()
    rotation_gt = rotation_gt.as_quat()

    all_error = [val_rotation(p, q) for p, q in zip(rotation_pred, rotation_gt)]

    all_error = np.array(all_error)

    all_error[all_error > 90] = 180 - all_error[all_error > 90]

    return all_error


def eval_6d_pose(annos, max_dis=100):
    orin_error, pos_error, size_error = eval_box6d_error(annos, max_dis)

    return orin_error, pos_error, size_error
