import numpy as np
import pandas as pd
import transforms3d.quaternions as txq
import transforms3d.euler as euler
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from scipy.spatial.transform import Rotation as R

def eval_key_points_error(annos):

    all_error = []

    for i, each_anno in enumerate(annos) :
        key_points_2d = each_anno['key_points_2d']
        gt_pts2d = each_anno['gt_pts2d']

        all_error.append(np.abs(key_points_2d-gt_pts2d).reshape(-1))

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

def match_gt(gt_box9d, pred_boxes9d):
    new_pred = []

    x_to_y = hungarian_match_allow_unmatched(gt_box9d[:,0:3], pred_boxes9d[:,0:3])

    for i, j in enumerate(x_to_y):

        if j==-1:
            new_pred.append(np.zeros(shape=(10)))
        else:
            new_pred.append(pred_boxes9d[j])

    return gt_box9d, np.array(new_pred)

def eval_box6d_error(annos, max_dis = 8):

    all_orin_error = []

    all_pos_error = []

    for i, each_anno in enumerate(annos) :
        gt_box9d = each_anno['gt_box9d']
        pred_boxes9d = each_anno['pred_box9d']

        pred_xyz = pred_boxes9d[:, 0:3]

        pred_xyz[np.abs(pred_xyz)>max_dis] = max_dis

        pred_boxes9d[:, 0:3] = pred_xyz

        if len(gt_box9d)==0 or len(pred_boxes9d)==0:
            continue
        gt_box9d, pred_boxes9d = match_gt(gt_box9d, pred_boxes9d)

        pred_xyz = pred_boxes9d[:, 0:3]

        gt_xyz = gt_box9d[:, 0:3]

        pred_angle = pred_boxes9d[:, 6:9]
        gt_angle = gt_box9d[:, 6:9]

        dis_error = np.linalg.norm(pred_xyz-gt_xyz,axis=-1)
        angle_error = val_rotation_enler(pred_angle, gt_angle)

        all_orin_error.append(angle_error)
        all_pos_error.append(dis_error)

    if len(all_orin_error)>0 and len(all_pos_error)>0:

        all_orin_error = np.concatenate(all_orin_error)
        all_pos_error = np.concatenate(all_pos_error)
    else:
        all_orin_error=np.array([0])
        all_pos_error=np.array([0])

    return all_orin_error, all_pos_error

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

    # d = abs(np.sum(np.multiply(groundtruth, predicted)))
    # if d != d:
    #     print("d is nan")
    #     raise ValueError
    # if d > 1:
    #     d = 1
    # error = 2 * np.arccos(d) * 180 / np.pi0
    # d     = abs(np.dot(groundtruth, predicted))
    # d     = min(1.0, max(-1.0, d))

    d = np.abs(np.dot(groundtruth, predicted))
    d = np.minimum(1.0, np.maximum(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error

def val_rotation_enler(pred_q, gt_q):
    rotation_pred = R.from_euler('zyx', pred_q, degrees=False)
    rotation_gt = R.from_euler('zyx', gt_q, degrees=False)

    rotation_pred = rotation_pred.as_quat()
    rotation_gt = rotation_gt.as_quat()

    all_error = [val_rotation(p,q) for p,q in zip(rotation_pred,rotation_gt)]

    all_error = np.array(all_error)

    all_error[all_error>90] = 180 - all_error[all_error>90]

    return all_error



def eval_6d_pose(annos, max_dis = 8):


    orin_error, pos_error = eval_box6d_error(annos, max_dis)


    return orin_error, pos_error
