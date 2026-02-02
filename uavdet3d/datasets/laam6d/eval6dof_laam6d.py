import torch
import numpy as np
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


def hungarian_match_allow_unmatched(x, y, unmatched_cost: float = 1e5):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    x = np.asarray(x)
    y = np.asarray(y)

    assert x.ndim == 2 and x.shape[1] == 3, f"x形状错误，应为(N,3)，实际为{x.shape}"
    assert y.ndim == 2 and y.shape[1] == 3, f"y形状错误，应为(M,3)，实际为{y.shape}"

    original_x_len = x.shape[0]

    valid_x = np.all(np.isfinite(x), axis=1)
    valid_y = np.all(np.isfinite(y), axis=1)
    x = x[valid_x]
    y = y[valid_y]
    N, M = x.shape[0], y.shape[0]

    if N == 0 or M == 0:
        return np.full(original_x_len, -1, dtype=int)

    try:
        cost = cdist(x, y)
    except Exception as e:
        print(f"距离计算失败: {e}")
        return np.full(original_x_len, -1, dtype=int)

    cost[~np.isfinite(cost)] = unmatched_cost

    size = max(N, M)
    padded_cost = np.full((size, size), unmatched_cost, dtype=np.float64)
    padded_cost[:N, :M] = cost

    try:
        row_ind, col_ind = linear_sum_assignment(padded_cost)
    except Exception as e:
        print(f"匈牙利算法失败: {e}")
        return np.full(original_x_len, -1, dtype=int)

    x_to_y = np.full(original_x_len, -1, dtype=int)
    original_x_indices = np.where(valid_x)[0]
    original_y_indices = np.where(valid_y)[0]

    for r, c in zip(row_ind, col_ind):
        if r < N and c < M and padded_cost[r, c] < unmatched_cost:
            x_to_y[original_x_indices[r]] = original_y_indices[c]

    return x_to_y


def match_gt_hungarian(gt_box9d, pred_boxes9d):
    new_pred = []

    gt_centers = gt_box9d[:, 0, :]
    pred_centers = pred_boxes9d[:, 0, :]

    x_to_y = hungarian_match_allow_unmatched(gt_centers, pred_centers)

    for i, j in enumerate(x_to_y):
        if j == -1:
            new_pred.append(np.full((9, 3), np.nan, dtype=np.float32))
        else:
            new_pred.append(pred_boxes9d[j])

    return gt_box9d, np.array(new_pred)


def compute_iou(box1, box2):
    # 步骤1：计算“交集框”的坐标（核心：取重叠区域的有效范围）
    x1 = max(box1[0], box2[0])  # 交集框的左上角x（取两个框x1的最大值，确保在两个框内部）
    y1 = max(box1[1], box2[1])  # 交集框的左上角y
    x2 = min(box1[2], box2[2])  # 交集框的右下角x（取两个框x2的最小值，确保在两个框内部）
    y2 = min(box1[3], box2[3])  # 交集框的右下角y

    # 步骤2：计算“交集面积”（若x2 <= x1或y2 <= y1，说明无重叠，交集面积为0）
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0  # 无重叠时直接返回0，避免后续无效计算

    # 步骤3：计算两个框各自的“总面积”
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])  # 框1面积 = 宽 × 高
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])  # 框2面积

    # 步骤4：计算“并集面积”和 IoU（并集 = 两个框面积和 - 交集面积，避免重复计算交集）
    union_area = area1 + area2 - inter_area
    return inter_area / union_area  # IoU = 交集 / 并集


def match_gt_2diou(gt_box9d, pred_boxes9d, gt_box2d, pred_box2d, match_iou=0.1):
    # 初始化：存储匹配结果 + 记录已被使用的预测框（避免重复匹配）
    matched_gt = []  # 存匹配成功的3D真实框
    matched_pred = []  # 存匹配成功的3D预测框
    used_preds = set()  # 存已匹配过的预测框索引（j值），确保一个预测框只匹配一个真实框

    # 步骤1：遍历每个3D真实框（为每个真实框找最好的预测框）
    for i, gt_box_2d in enumerate(gt_box2d):  # i：当前真实框的索引（0~M-1）
        best_iou = 0  # 记录当前真实框能匹配到的最大IoU
        best_j = -1  # 记录最大IoU对应的预测框索引（初始为-1，表示未找到）

        # 步骤2：遍历每个未被使用的预测框
        for j, pred_box_2d in enumerate(pred_box2d):  # j：当前预测框的索引（0~N-1）
            if j in used_preds:  # 跳过已匹配过的预测框
                continue

            # 步骤3：计算当前真实框与预测框的2D IoU
            iou = compute_iou(gt_box_2d, pred_box_2d)

            # 步骤4：更新“最优匹配”（如果当前IoU比之前的最大IoU大）
            if iou > best_iou:
                best_iou = iou  # 更新最大IoU
                best_j = j  # 更新最优预测框的索引

        # 步骤5：判断是否匹配成功（最大IoU ≥ 阈值，且找到最优预测框）
        if best_iou >= match_iou and best_j != -1:
            matched_gt.append(gt_box9d[i])  # 把匹配成功的3D真实框加入列表
            matched_pred.append(pred_boxes9d[best_j])  # 把对应的3D预测框加入列表
            used_preds.add(best_j)  # 标记该预测框为已使用（避免重复匹配）

    # 步骤6：处理匹配结果，返回统一格式
    if matched_gt:  # 若有匹配成功的结果（matched_gt非空）
        # 列表转NumPy数组：(K, 9, 3)（K是匹配对数）
        return np.stack(matched_gt), np.stack(matched_pred)
    else:  # 若没有任何匹配成功的结果（如IoU都低于阈值，或无真实框/预测框）
        # 返回两个空数组，形状为(0, 9, 3)（0表示无数据，9和3保持与有数据时的维度一致）
        return np.zeros((0, 9, 3)), np.zeros((0, 9, 3))


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


def val_rotation_euler_old(pred_q, gt_q):
    rotation_pred = R.from_euler('zyx', pred_q, degrees=False)
    rotation_gt = R.from_euler('zyx', gt_q, degrees=False)

    rotation_pred = rotation_pred.as_quat()
    rotation_gt = rotation_gt.as_quat()

    all_error = [val_rotation(p, q) for p, q in zip(rotation_pred, rotation_gt)]

    all_error = np.array(all_error)

    all_error[all_error > 90] = 180 - all_error[all_error > 90]

    return all_error


def val_rotation_euler(pred_euler, gt_euler):
    """
    输入：
        pred_euler：预测的欧拉角 (3,)，如[a1,a2,a3]
        gt_euler：GT的欧拉角 (3,)
    输出：
        旋转误差（角度）
    """
    try:
        # 1. 欧拉角→旋转对象（关键：确认单位！若输入是角度，degrees=True）
        # 若模型输出的是弧度（范围≈[-3.14,3.14]），用degrees=False；若是角度（≈[-180,180]），用True
        rotation_pred = R.from_euler('zyx', pred_euler, degrees=False)  # 注意旋转顺序是否和GT一致
        rotation_gt = R.from_euler('zyx', gt_euler, degrees=False)

        # 2. 转换为四元数（整体代表旋转，不是分量）
        pred_q = rotation_pred.as_quat()
        gt_q = rotation_gt.as_quat()

        # 3. 单位化四元数（必须步骤）
        pred_q = pred_q / np.linalg.norm(pred_q)
        gt_q = gt_q / np.linalg.norm(gt_q)

        # 4. 计算旋转误差
        d = np.abs(np.dot(gt_q, pred_q))  # 点积的绝对值（避免q和-q的符号问题）
        d = np.clip(d, -1.0, 1.0)  # 确保在[-1,1]内（浮点数误差可能超范围）
        error = 2 * np.arccos(d) * 180 / np.pi  # 转换为角度

        # 5. 取最小角度（旋转误差最大90度，比如120度等价于60度）
        if error > 90:
            error = 180 - error
        return error

    except Exception as e:
        print(f"旋转误差计算失败：{e}，输入欧拉角：pred={pred_euler}, gt={gt_euler}")
        return np.nan  # 出错时返回nan，后续过滤


def eval_box6d_error(annos, max_dis):
    all_orin_error = []
    all_posi_error = []
    all_size_error = []

    for i, each_anno in enumerate(annos):
        gt_box9d = each_anno['gt_box9d']
        pred_boxes9d = each_anno['pred_box9d']

        gt_box2d = each_anno['gt_box2d']
        pred_box2d = each_anno['pred_box2d']

        if isinstance(gt_box9d, torch.Tensor):
            gt_box9d = gt_box9d.cpu().numpy()
        if isinstance(pred_boxes9d, torch.Tensor):
            pred_boxes9d = pred_boxes9d.cpu().numpy()

        if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
            continue

        gt_box9d, pred_boxes9d = match_gt_2diou(gt_box9d, pred_boxes9d, gt_box2d, pred_box2d, match_iou=0.1)

        if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
            continue

        pred_xyz = pred_boxes9d[:, 0, :]
        gt_xyz = gt_box9d[:, 0, :]

        # (center, l, w, h, a1, a2, a3)
        pred_params = convert_box9d_to_box_param(pred_boxes9d)  # (N, 9)
        gt_params = convert_box9d_to_box_param(gt_box9d)  # (N, 9)

        pred_angle = pred_params[:, 6:9]  # (N, 3)
        gt_angle = gt_params[:, 6:9]  # (N, 3)
        angle_error = np.array([val_rotation_euler(pred_angle[i], gt_angle[i]) for i in range(len(pred_angle))])

        dis_error = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)
        dis_error[dis_error > max_dis] = max_dis
        size_error = np.mean(np.abs(pred_params[:, 3:6] - gt_params[:, 3:6]), axis=-1)

        dis_error = dis_error[~np.isnan(dis_error) & ~np.isinf(dis_error)]
        angle_error = angle_error[~np.isnan(angle_error) & ~np.isinf(angle_error)]
        size_error = size_error[~np.isnan(size_error) & ~np.isinf(size_error)]

        if len(dis_error) > 0 and len(angle_error) > 0:
            all_orin_error.append(angle_error)
            all_posi_error.append(dis_error)
            all_size_error.append(size_error)

    if len(all_orin_error) > 0 and len(all_posi_error) > 0:
        all_orin_error = np.concatenate(all_orin_error)
        all_posi_error = np.concatenate(all_posi_error)
        all_size_error = np.concatenate(all_size_error)

    else:
        all_orin_error = np.array([0.0])
        all_posi_error = np.array([0.0])
        all_size_error = np.array([0.0])

    return all_orin_error, all_posi_error, all_size_error


def eval_6d_posi(annos, max_dis=100):
    orin_error, posi_error, size_error = eval_box6d_error(annos, max_dis)

    return orin_error, posi_error, size_error


def convert_9points_to_9params(box9d_points):
    """
    编码：9点框→9参数，返回参数和局部原型（避免依赖实例变量）
    :param box9d_points: (N, 9, 3) 9点框 [中心点, 角点1-8]
    :return:
        box9d_params: (N, 9) 9参数 [x,y,z,l,w,h,a1,a2,a3]
        local_prototypes: 列表，每个元素为(N, 8, 3) 物体坐标系下的归一化角点
    """
    all_box_params = []
    standard_prototype = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ])

    for single_box in box9d_points:
        # 1. 提取中心点和局部角点
        center = single_box[0].copy()  # (3,)
        x, y, z = center
        corners_local = single_box[1:] - center  # (8, 3) 角点相对中心点的偏移

        # 2. 计算旋转矩阵（与解码逻辑一致：基于欧拉角转换）
        # 先通过SVD获取初始旋转矩阵
        H = standard_prototype.T @ corners_local
        U, S, Vt = np.linalg.svd(H)
        rotation_matrix = Vt.T @ U.T
        orthogonal_rot, _ = np.linalg.qr(rotation_matrix)  # 确保正交
        if np.linalg.det(orthogonal_rot) < 0:
            orthogonal_rot[:, 2] *= -1

        # 3. 从旋转矩阵计算欧拉角（zyx顺序）
        r = R.from_matrix(orthogonal_rot)
        angles = r.as_euler('zyx')  # (a1, a2, a3) 对应z,y,x轴旋转
        angle1, angle2, angle3 = angles

        # 4. 用欧拉角重构旋转矩阵（确保与解码时逻辑一致）
        rotation_matrix = R.from_euler('zyx', [angle1, angle2, angle3]).as_matrix()

        # 5. 计算物体自身坐标系下的角点和尺寸
        corners_self = np.dot(corners_local, rotation_matrix)  # (8, 3) 物体坐标系下的坐标
        l = np.max(corners_self[:, 0]) - np.min(corners_self[:, 0])
        w = np.max(corners_self[:, 1]) - np.min(corners_self[:, 1])
        h = np.max(corners_self[:, 2]) - np.min(corners_self[:, 2])

        # 6. 处理零尺寸（避免除零错误）
        l = max(l, 1e-6)
        w = max(w, 1e-6)
        h = max(h, 1e-6)
        scale = np.array([l, w, h])

        # 7. 记录局部原型（归一化角点）
        local_prototype = corners_self / scale  # 物体坐标系下的单位尺寸角点

        # 8. 组合9参数
        box_params = np.array([x, y, z, l, w, h, angle1, angle2, angle3])
        all_box_params.append(box_params)

    return np.array(all_box_params)


def convert_box_opencv_to_world(pts_opencv, extrinsic):
    opencv_to_carla = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    pts_world = []
    for pt in pts_opencv:
        pt_h = np.append(pt, 1.0)  # (4,)
        pt_carla = opencv_to_carla @ pt_h
        pt_world = extrinsic @ pt_carla
        pts_world.append(pt_world[:3])
    return np.array(pts_world, dtype=np.float32)


def convert_box9d_to_box_param(boxes_9d):
    if isinstance(boxes_9d, torch.Tensor):
        boxes_9d = boxes_9d.cpu().numpy()
    boxes_9d = np.asarray(boxes_9d)

    all_box_params = []
    for box in boxes_9d:
        if np.isnan(box).any():
            continue

        center = box[0]
        corners = box[1:9]

        l = np.linalg.norm(corners[0] - corners[1])
        w = np.linalg.norm(corners[1] - corners[2])
        h = np.linalg.norm(corners[0] - corners[4])

        if l < 1e-6 or w < 1e-6 or h < 1e-6:
            print(f"[警告] 边界框尺寸异常: l={l}, w={w}, h={h}，跳过该框")
            continue

        x_axis = corners[1] - corners[0]
        y_axis = corners[3] - corners[0]
        z_axis = corners[4] - corners[0]

        x_norm = np.linalg.norm(x_axis)
        y_norm = np.linalg.norm(y_axis)
        z_norm = np.linalg.norm(z_axis)

        if x_norm < 1e-6 or y_norm < 1e-6 or z_norm < 1e-6:
            print(f"[警告] 轴向量模长为0，跳过该框")
            continue

        rot_mat = np.stack([
            x_axis / x_norm,
            y_axis / y_norm,
            z_axis / z_norm
        ], axis=1)

        try:
            r = R.from_matrix(rot_mat)
            euler = r.as_euler('zyx', degrees=False)
        except:
            print(f"[警告] 旋转矩阵无效，使用默认欧拉角")
            euler = np.zeros(3)

        box_param = np.concatenate([center, [l, w, h], euler])
        all_box_params.append(box_param)

    return np.array(all_box_params) if all_box_params else np.empty((0, 9))
