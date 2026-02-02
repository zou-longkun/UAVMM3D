import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap
import copy
import time


def projectPoints(corners, rotation_mat, translation_vec, intrinsic_mat, distortion_matrix):
    corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))

    extrinsic_mat = np.hstack((rotation_mat, translation_vec.reshape(3, 1)))

    corners_cam = np.dot(extrinsic_mat, corners_homogeneous.T).T

    depth = corners_cam[:, 2]

    corners_norm = corners_cam / corners_cam[:, 2].reshape(-1, 1)

    x = corners_norm[:, 0]
    y = corners_norm[:, 1]

    r2 = x ** 2 + y ** 2

    k1, k2, p1, p2, k3 = distortion_matrix
    radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_distorted = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_distorted = y * radial_distortion + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    corners_2d = np.dot(intrinsic_mat, np.vstack((x_distorted, y_distorted, np.ones_like(x_distorted)))).T[:, :2]

    return corners_2d, depth


def backProject(corners_2D, depth, rotation_mat, translation_vec, intrinsic_mat, distortion_matrix):
    corners_2D_homogeneous = np.hstack((corners_2D, np.ones((corners_2D.shape[0], 1))))

    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)

    corners_norm = np.dot(intrinsic_mat_inv, corners_2D_homogeneous.T).T

    x = corners_norm[:, 0]
    y = corners_norm[:, 1]

    r2 = x ** 2 + y ** 2

    k1, k2, p1, p2, k3 = distortion_matrix
    radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_undistorted = (x - 2 * p1 * x * y - p2 * (r2 + 2 * x ** 2)) / radial_distortion
    y_undistorted = (y - p1 * (r2 + 2 * y ** 2) - 2 * p2 * x * y) / radial_distortion

    points_cam = np.vstack((x_undistorted * depth, y_undistorted * depth, depth)).T

    extrinsic_mat = np.hstack((rotation_mat, translation_vec.reshape(3, 1)))
    extrinsic_mat_homogeneous = np.vstack((extrinsic_mat, np.array([0, 0, 0, 1])))
    extrinsic_mat_inv = np.linalg.inv(extrinsic_mat_homogeneous)

    points_cam_homogeneous = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))
    points3d_homogeneous = np.dot(extrinsic_mat_inv, points_cam_homogeneous.T).T

    points3d = points3d_homogeneous[:, :3]

    return points3d


def backProject_with_opencv_to_world(corners_2D, depth, rotation_mat, translation_vec, intrinsic_mat,
                                     distortion_matrix):
    # === Step 1: 像素 → 归一化相机坐标（OpenCV 相机系） ===
    corners_2D_homogeneous = np.hstack((corners_2D, np.ones((corners_2D.shape[0], 1))))
    intrinsic_inv = np.linalg.inv(intrinsic_mat)
    corners_norm = (intrinsic_inv @ corners_2D_homogeneous.T).T  # shape (N, 3)

    # === Step 2: 去畸变 ===
    x = corners_norm[:, 0]
    y = corners_norm[:, 1]
    r2 = x ** 2 + y ** 2
    k1, k2, p1, p2, k3 = distortion_matrix

    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_undist = (x - 2 * p1 * x * y - p2 * (r2 + 2 * x ** 2)) / radial
    y_undist = (y - p1 * (r2 + 2 * y ** 2) - 2 * p2 * x * y) / radial

    # === Step 3: 得到 OpenCV 相机坐标系下的点 ===
    points_opencv = np.vstack((x_undist * depth, y_undist * depth, depth)).T  # (N, 3)

    # === Step 4: OpenCV → Carla 相机坐标系 ===
    opencv_to_carla = np.linalg.inv(np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]))

    points_opencv_hom = np.hstack([points_opencv, np.ones((points_opencv.shape[0], 1))])  # (N, 4)
    points_carla = (opencv_to_carla @ points_opencv_hom.T).T[:, :3]  # 去掉齐次

    # === Step 5: Carla 相机坐标系 → 世界坐标（使用 extrinsic） ===
    extrinsic_mat = np.hstack((rotation_mat, translation_vec.reshape(3, 1)))  # 3x4
    extrinsic_mat_hom = np.vstack((extrinsic_mat, np.array([0, 0, 0, 1])))  # 4x4

    points_carla_hom = np.hstack([points_carla, np.ones((points_carla.shape[0], 1))])  # (N, 4)
    points_world = (extrinsic_mat_hom @ points_carla_hom.T).T[:, :3]  # (N, 3)

    return points_world


def key_point_encoder(boxes9d, encode_corner=None, intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None,
                      offset=np.array([-0.15, 0.05, 0])):
    corners_local = np.array(encode_corner, dtype=np.float32)
    corners_local += offset

    all_corners_3d = []
    all_corners_2d = []
    if intrinsic_mat is not None:
        intrinsic_mat = np.array(intrinsic_mat, dtype=np.float32)
    if distortion_matrix is not None:
        distortion_matrix = np.array(distortion_matrix, dtype=np.float32)

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box
        corners = corners_local * np.array([l, w, h])

        rotation_matrix = R.from_euler('zyx', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners_rotated = np.dot(corners, rotation_matrix.T)
        corners_cam = corners_rotated + np.array([x, y, z])
        corners_2d, _ = cv2.projectPoints(
            corners_cam,
            rvec=np.zeros(3, dtype=np.float32),
            tvec=np.zeros(3, dtype=np.float32),
            cameraMatrix=intrinsic_mat.reshape(3, 3).astype(np.float32),
            distCoeffs=distortion_matrix.astype(np.float32)
        )
        corners_2d = corners_2d.reshape(-1, 2)

        all_corners_3d.append(corners_cam)
        all_corners_2d.append(corners_2d)

    return np.array(all_corners_3d), np.array(all_corners_2d)


def key_point_decoder(encode_corner,
                      off_set,
                      pred_heat_map=None,  # 1,1,4,W,H
                      pred_res_x=None,  # 1,1,4,W,H
                      pred_res_y=None,  # 1,1,4,W,H
                      new_im_width=None,
                      new_im_hight=None,
                      raw_im_width=None,
                      raw_im_hight=None,
                      stride=None,
                      im_num=None,
                      obj_num=None,
                      size=None,
                      intrinsic=None,
                      distortion=None,
                      PnP_algo='SOLVEPNP_EPNP'):
    im_num = im_num
    obj_num = obj_num

    corners_local = np.array(encode_corner) + off_set

    key_pts_num = len(corners_local)

    key_points_2d = torch.zeros(im_num, obj_num, key_pts_num, 2)

    confidence = torch.zeros(im_num, obj_num)

    for ob_id in range(obj_num):

        all_conf = []
        for k_id in range(key_pts_num):
            this_heatmap = pred_heat_map[0, ob_id, k_id]
            this_res_x = pred_res_x[0, ob_id, k_id]
            this_res_y = pred_res_y[0, ob_id, k_id]

            shape_map = this_heatmap.shape

            flat_x = this_heatmap.flatten()
            values, linear_indices = torch.topk(flat_x, k=1)

            c_num = shape_map[-1]
            rows = linear_indices // c_num
            cols = linear_indices % c_num

            row = rows[0]
            col = cols[0]
            conf = values[0]
            all_conf.append(conf)

            res_x = this_res_x[row.long(), col.long()]
            res_y = this_res_y[row.long(), col.long()]

            y_cor = row.float() + res_y
            x_cor = col.float() + res_x

            delta0 = (new_im_width / raw_im_width / stride)
            delta1 = (new_im_hight / raw_im_hight / stride)

            key_points_2d[0, ob_id, k_id, 0] = (x_cor / delta0)
            key_points_2d[0, ob_id, k_id, 1] = (y_cor / delta1)

        confidence[0, ob_id] = torch.mean(torch.stack(all_conf))

    key_points_2d = key_points_2d.detach().cpu().numpy()
    confidence = confidence.detach().cpu().numpy()

    all_pred_box9d = []

    for im_id in range(im_num):

        all_obj_each_im = []

        for ob_id in range(obj_num):
            # Scale the local corner points according to the target size
            this_corner = corners_local * size[ob_id]

            im_pts = key_points_2d[im_id, ob_id]

            if PnP_algo == 'SOLVEPNP_EPNP':
                success, rvec, tvec = cv2.solvePnP(this_corner, np.array(im_pts, dtype=np.float64), intrinsic[im_id],
                                                   distortion[im_id],
                                                   flags=cv2.SOLVEPNP_EPNP)  # , flags=cv2.SOLVEPNP_AP3P
            elif PnP_algo == 'SOLVEPNP_AP3P':
                success, rvec, tvec = cv2.solvePnP(this_corner, np.array(im_pts, dtype=np.float64), intrinsic[im_id],
                                                   distortion[im_id],
                                                   flags=cv2.SOLVEPNP_AP3P)  # , flags=cv2.SOLVEPNP_AP3P
            else:
                success, rvec, tvec = cv2.solvePnP(this_corner, np.array(im_pts, dtype=np.float64), intrinsic[im_id],
                                                   distortion[im_id],
                                                   flags=cv2.SOLVEPNP_ITERATIVE)  # , flags=cv2.SOLVEPNP_AP3P

            if success:
                # Convert the rotation vector to a rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # Extract Euler angles (rotation angles around x, y, z axes) from the rotation matrix
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                angle1 = np.arctan2(R[2, 1], R[2, 2])  # Rotation around x-axis
                angle2 = np.arctan2(-R[2, 0], sy)  # Rotation around y-axis
                angle3 = np.arctan2(R[1, 0], R[0, 0])  # Rotation around z-axis

                # Combine the translation vector and rotation angles into 9D parameters
                box9d = [tvec[0, 0], tvec[1, 0], tvec[2, 0], size[ob_id][0], size[ob_id][1], size[ob_id][2], angle1,
                         angle2, angle3]
                all_obj_each_im.append(box9d)
            else:
                # If PnP solving fails, return default values
                all_obj_each_im.append([0, 0, 0, size[ob_id][0], size[ob_id][1], size[ob_id][2], 0, 0, 0])

        all_pred_box9d.append(all_obj_each_im)

    all_pred_box9d = np.array(all_pred_box9d)

    pred_boxes9d = all_pred_box9d.reshape(obj_num * im_num, 9)

    key_points_2d = key_points_2d.reshape(obj_num * im_num, key_pts_num, 2),
    confidence = confidence.reshape(obj_num * im_num)

    return key_points_2d, confidence, pred_boxes9d


def center_point_encoder(gt_box9d_with_cls,
                         intrinsic_mat,
                         extrinsic_mat,
                         distortion_matrix,
                         new_im_width=None,
                         new_im_hight=None,
                         raw_im_width=None,
                         raw_im_hight=None,
                         stride=None,
                         im_num=None,
                         class_name_config=None,
                         center_rad=None):
    # intrinsic_mat = np.array(intrinsic_mat[0])
    cls_num = len(class_name_config)
    gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []

    scale_heatmap_w = 1.0 / stride
    scale_heatmap_h = 1.0 / stride

    xyz = copy.deepcopy(gt_box9d_with_cls[:, 0:3])
    
    # 函数入口参数检查（已禁用调试打印）
    
    for im_id in range(im_num):
        hm_height = new_im_hight // stride
        hm_width = new_im_width // stride

        this_heat_map = torch.zeros(cls_num, hm_height, hm_width)
        this_res_map = np.zeros(shape=(2, hm_height, hm_width))
        this_dis_map = np.zeros(shape=(1, hm_height, hm_width))
        this_size_map = np.zeros(shape=(3, hm_height, hm_width))
        this_angle_map = np.zeros(shape=(6, hm_height, hm_width))
        
        # 跟踪有效的目标数量
        valid_target_count = 0
        z_filtered_count = 0
        out_of_bounds_count = 0

        for obj_i, obj in enumerate(gt_box9d_with_cls):
            this_cls = int(obj[-1])
            X, Y, Z = xyz[obj_i]
            
            if Z <= 1e-6:
                z_filtered_count += 1
                continue

            fx, fy = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
            cx, cy = intrinsic_mat[0, 2], intrinsic_mat[1, 2]
            u = (X / Z) * fx + cx
            v = (Y / Z) * fy + cy

            u_heatmap = u * scale_heatmap_w
            v_heatmap = v * scale_heatmap_h
            center_heatmap = [u_heatmap, v_heatmap]

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(
                this_heat_map[this_cls],
                center_heatmap,
                center_rad
            )

            this_res_map[0], this_res_map[1] = draw_res_to_heatmap(
                this_res_map[0],
                this_res_map[1],
                center_heatmap
            )

            try:
                h_idx = int(v_heatmap)
                w_idx = int(u_heatmap)

                if 0 <= h_idx < hm_height and 0 <= w_idx < hm_width:
                    this_dis_map[0, h_idx, w_idx] = Z
                    l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]
                    this_size_map[0, h_idx, w_idx] = l
                    this_size_map[1, h_idx, w_idx] = w
                    this_size_map[2, h_idx, w_idx] = h

                    this_angle_map[0, h_idx, w_idx] = np.cos(a1)
                    this_angle_map[1, h_idx, w_idx] = np.sin(a1)
                    this_angle_map[2, h_idx, w_idx] = np.cos(a2)
                    this_angle_map[3, h_idx, w_idx] = np.sin(a2)
                    this_angle_map[4, h_idx, w_idx] = np.cos(a3)
                    this_angle_map[5, h_idx, w_idx] = np.sin(a3)
                    
                    valid_target_count += 1
                else:
                    out_of_bounds_count += 1

            except Exception as e:
                print(f"处理目标 {obj_i} 时出错: {e}")
                continue
        
        # 每个图像的统计（已禁用调试打印）

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis.append(this_dis_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)
        
    # 最终输出统计（已禁用调试打印）

    return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis), np.array(gt_dim), np.array(gt_rot)


def nms_pytorch(confidence_map, max_num, distance_threshold=5):
    """
    PyTorch implementation of Non-Maximum Suppression

    Args:
        confidence_map (torch.Tensor): 2D confidence map with shape (H, W)
        max_num (int): Maximum number of points to return
        distance_threshold (float): Pixel distance threshold for suppression (default: 5)

    Returns:
        tuple: (confidences, linear_indices)
            - confidences: Selected confidence values, shape (k,) where k <= max_num
            - linear_indices: Linear indices of selected points in flattened array, shape (k,)
    """
    device = confidence_map.device
    H, W = confidence_map.shape

    # Flatten the confidence map
    flat_conf = confidence_map.flatten()

    # Get initial top-k candidates
    initial_k = min(max_num, flat_conf.numel())  # Get more candidates initially
    top_conf, top_indices = torch.topk(flat_conf, k=initial_k)

    # Convert linear indices to 2D coordinates
    top_y = top_indices // W
    top_x = top_indices % W
    top_coords = torch.stack([top_x, top_y], dim=1).float()  # Shape: (initial_k, 2)

    # NMS process
    selected_mask = torch.ones(len(top_indices), dtype=torch.bool, device=device)

    for i in range(len(top_indices)):
        if not selected_mask[i]:
            continue

        # Current point coordinates
        current_coord = top_coords[i:i + 1]  # Shape: (1, 2)

        # Calculate distances to all remaining points
        remaining_coords = top_coords[i + 1:]  # Shape: (remaining, 2)
        if len(remaining_coords) == 0:
            break

        # Euclidean distance
        distances = torch.norm(remaining_coords - current_coord, dim=1)  # Shape: (remaining,)

        # Suppress points within distance threshold
        suppress_mask = distances < distance_threshold
        selected_mask[i + 1:] = selected_mask[i + 1:] & (~suppress_mask)

    # Get final selected results
    selected_indices = torch.where(selected_mask)[0][:max_num]

    # Return results in the requested format
    confi = top_conf[selected_indices]
    linear_indices = top_indices[selected_indices]

    return confi, linear_indices

# 二阶段版本
# def _robust_depth_from_patch(patch: np.ndarray, invalid_val: float = 0.0):
#     """
#     patch: 2D depth patch, depth unit == patch unit (here: normalized)
#     returns:
#       z_med (float or None),
#       valid_count (int),
#       spread (float or None)  # p90 - p10
#     """
#     vals = patch.reshape(-1)
#     vals = vals[vals > invalid_val]
#     if vals.size == 0:
#         return None, 0, None

#     vals = np.sort(vals)
#     cnt = int(vals.size)
#     if cnt == 1:
#         return float(vals[0]), 1, 0.0

#     p10 = float(vals[int(0.1 * (cnt - 1))])
#     p90 = float(vals[int(0.9 * (cnt - 1))])
#     spread = p90 - p10
#     z_med = float(np.median(vals))
#     return z_med, cnt, spread

# def _point_in_box(x: float, y: float, box):
#     x1, y1, x2, y2 = box
#     return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)

# def _point_in_box_center(x: float, y: float, box, ratio: float = 0.5):
#     x1, y1, x2, y2 = box
#     cx = 0.5 * (x1 + x2)
#     cy = 0.5 * (y1 + y2)
#     w = max(1.0, (x2 - x1))
#     h = max(1.0, (y2 - y1))
#     cw = 0.5 * w * ratio
#     ch = 0.5 * h * ratio
#     return (x >= cx - cw) and (x <= cx + cw) and (y >= cy - ch) and (y <= cy + ch)

# def _boxes_gate_mask(xs, ys, boxes2d, center_only=True, center_ratio=0.5):
#     """Return boolean mask (K,) for points inside any box (or its center region)."""
#     if boxes2d is None or len(boxes2d) == 0:
#         return np.ones_like(xs, dtype=bool)

#     boxes = np.asarray(boxes2d, dtype=np.float32).reshape(-1, 4)
#     out = np.zeros_like(xs, dtype=bool)
#     for i in range(xs.shape[0]):
#         x = float(xs[i]); y = float(ys[i])
#         ok = False
#         for b in boxes:
#             if center_only:
#                 if _point_in_box_center(x, y, b, ratio=center_ratio):
#                     ok = True
#                     break
#             else:
#                 if _point_in_box(x, y, b):
#                     ok = True
#                     break
#         out[i] = ok
#     return out

# def lidar_hard_calibrate_depth(x_pix: np.ndarray, y_pix: np.ndarray,
#                                depth_pred: np.ndarray,
#                                lidar_depth_map: np.ndarray,          # (H,W) normalized
#                                hm_conf: np.ndarray = None,           # (K,) optional heatmap confidence at each point
#                                boxes2d: np.ndarray = None,           # (N,4) pixels optional gate
#                                # ---- gates ----
#                                win: int = 4,  
#                                invalid_val: float = 0.0,
#                                min_count: int = 5,
#                                max_spread: float = 0.05,             # normalized, 0 disables if <=0 or None
#                                hm_thr: float = 0.35,                 # 0 disables if <=0 or None
#                                delta0: float = 0.02,                 # normalized abs delta floor
#                                rel_delta: float = 0.15,              # normalized relative delta
#                                # ---- box gate ----
#                                box_center_only: bool = True,
#                                box_center_ratio: float = 0.5,
#                                debug: bool = False,):
#     """
#     HARD replace depth_pred[i] <- median(lidar patch) if:
#       - inside boxes2d (if provided) (or their center region)
#       - hm_conf[i] >= hm_thr (if provided)
#       - patch has enough valid points
#       - patch spread small enough
#       - lidar vs pred delta within threshold (abs + relative)
#     All in normalized unit.
#     """
#     H, W = lidar_depth_map.shape[:2]
#     K = depth_pred.shape[0]

#     xs = x_pix.reshape(-1).astype(np.float32)
#     ys = y_pix.reshape(-1).astype(np.float32)
#     out = depth_pred.copy().astype(np.float32)

#     # gate by boxes
#     box_mask = _boxes_gate_mask(xs, ys, boxes2d, center_only=box_center_only, center_ratio=box_center_ratio)

#     used = 0
#     cnts, sprs, deltas = [], [], []
#     rej_oob = rej_box = rej_hm = rej_cnt = rej_spr = rej_delta = 0

#     for i in range(K):
#         x = float(xs[i]); y = float(ys[i])

#         if not box_mask[i]:
#             rej_box += 1
#             continue

#         if hm_conf is not None and hm_thr is not None and hm_thr > 0:
#             if float(hm_conf[i]) < hm_thr:
#                 rej_hm += 1
#                 continue

#         xi = int(round(x)); yi = int(round(y))
#         if xi < 0 or xi >= W or yi < 0 or yi >= H:
#             rej_oob += 1
#             continue

#         x0, x1 = max(0, xi - win), min(W, xi + win + 1)
#         y0, y1 = max(0, yi - win), min(H, yi + win + 1)

#         patch = lidar_depth_map[y0:y1, x0:x1]
#         z_lidar, cnt, spr = _robust_depth_from_patch(patch, invalid_val=invalid_val)
#         if z_lidar is None or cnt < min_count:
#             rej_cnt += 1
#             continue

#         if max_spread is not None and max_spread > 0 and spr is not None and spr > max_spread:
#             rej_spr += 1
#             continue

#         z_pred = float(out[i])
#         # abs + relative delta gate（防止替换到背景/穿透）
#         abs_thr = float(delta0) if (delta0 is not None and delta0 > 0) else 0.0
#         rel_thr = float(rel_delta) if (rel_delta is not None and rel_delta > 0) else 0.0
#         thr = max(abs_thr, rel_thr * max(z_lidar, z_pred, 1e-6))
#         if thr > 0 and abs(z_lidar - z_pred) > thr:
#             rej_delta += 1
#             continue

#         out[i] = z_lidar
#         used += 1
#         cnts.append(cnt)
#         sprs.append(float(spr if spr is not None else 0.0))
#         deltas.append(abs(z_lidar - z_pred))

#     if debug:
#         print(f"[LiDAR HARD] used={used}/{K} win={2*win+1} hm_thr={hm_thr} min_count={min_count} max_spread={max_spread}")
#         print(f"  rej: oob={rej_oob} box={rej_box} hm={rej_hm} cnt={rej_cnt} spr={rej_spr} delta={rej_delta}")
#         if used > 0:
#             print(f"  cnt: mean={np.mean(cnts):.3f} p50={np.median(cnts):.3f} max={np.max(cnts):.0f}")
#             print(f"  spr: mean={np.mean(sprs):.4f} p50={np.median(sprs):.4f} max={np.max(sprs):.4f}")
#             print(f"  |Δ|: mean={np.mean(deltas):.4f} p50={np.median(deltas):.4f} max={np.max(deltas):.4f}")

#     return out

# def center_point_decoder(hm,
#                          center_res,
#                          center_dis,
#                          dim,
#                          rot,
#                          intrinsic_mat,
#                          extrinsic_mat,
#                          distortion_matrix,
#                          new_im_width=None,
#                          new_im_height=None,
#                          raw_im_width=None,
#                          raw_im_height=None,
#                          stride=None,
#                          im_num=None,
#                          max_num=10,
#                          lidar_depth_map=None,              # (H,W) in SAME SCALE as center_dis (normalized 0~1)
#                          # ---- LiDAR hard replace params ----
#                          lidar_win=4,
#                          lidar_min_count=2,
#                          lidar_max_spread=0.05,             # normalized spread threshold (if MAX_DIS=150 => 0.05~7.5m)
#                          lidar_hm_thr=0.35,
#                          lidar_delta0=0.02,                 # normalized abs delta floor (0.02=>3m if 150m)
#                          lidar_rel_delta=0.15,
#                          lidar_box_center_only=True,
#                          lidar_box_center_ratio=0.5,
#                          lidar_debug=False,
#                          # ---- two-stage toggle ----
#                          lidar_two_stage=True):
#     """
#     Two-stage decoder:
#       stage1: decode with predicted depth -> get boxes2d
#       stage2: if lidar available, hard replace depth ONLY at peak points gated by stage1 boxes2d, then decode again
#     All depth are normalized (0~1). Your outer post_processing can still * MAX_DIS to meters AFTER this if你想。
#     """

#     hm = torch.tensor(hm) if isinstance(hm, np.ndarray) else hm
#     center_res = torch.tensor(center_res) if isinstance(center_res, np.ndarray) else center_res
#     center_dis = torch.tensor(center_dis) if isinstance(center_dis, np.ndarray) else center_dis
#     dim = torch.tensor(dim) if isinstance(dim, np.ndarray) else dim
#     rot = torch.tensor(rot) if isinstance(rot, np.ndarray) else rot

#     pred_boxes9d_all = []
#     confidence_all = []

#     for im_id in range(im_num):
#         im_mat = intrinsic_mat
#         ex_mat = extrinsic_mat
#         dis_mat = distortion_matrix

#         this_hm = hm[im_id]  # (cls, Hf, Wf)
#         this_hm_conf, cls = this_hm.max(dim=0)  # (Hf,Wf)

#         this_center_res = center_res[im_id]
#         this_center_dis = center_dis[im_id]     # (1,Hf,Wf) normalized
#         this_dim = dim[im_id]
#         this_rot = rot[im_id]

#         Hf, Wf = this_hm_conf.shape
#         confi_t, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)
#         confi = confi_t.detach().cpu().numpy()

#         rows = (linear_indices // Wf).long()
#         cols = (linear_indices % Wf).long()

#         res_x = this_center_res[0, rows, cols]
#         res_y = this_center_res[1, rows, cols]

#         # heatmap coords -> pixel coords (distorted pixel)
#         u_heatmap = cols.float() + res_x
#         v_heatmap = rows.float() + res_y
#         x_cor = (u_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)  # (K,1)
#         y_cor = (v_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)  # (K,1)

#         # also grab hm conf at the same peaks (for gate)
#         hm_k = this_hm_conf[rows, cols].detach().cpu().numpy().reshape(-1)   # (K,)

#         # depth pred normalized at peaks
#         depth = this_center_dis[0, rows, cols].detach().cpu().numpy().reshape(-1).astype(np.float32)  # (K,)

#         # others
#         cls_values = cls[rows, cols].detach().cpu().numpy().reshape(-1, 1)
#         l = this_dim[0, rows, cols].detach().cpu().numpy().reshape(-1, 1)
#         w = this_dim[1, rows, cols].detach().cpu().numpy().reshape(-1, 1)
#         h = this_dim[2, rows, cols].detach().cpu().numpy().reshape(-1, 1)

#         a1 = torch.atan2(this_rot[1, rows, cols], this_rot[0, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
#         a2 = torch.atan2(this_rot[3, rows, cols], this_rot[2, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
#         a3 = torch.atan2(this_rot[5, rows, cols], this_rot[4, rows, cols]).detach().cpu().numpy().reshape(-1, 1)

#         # ===================== undistort pixel coords =====================
#         if dis_mat is not None and np.any(dis_mat != 0):
#             k1, k2, p1, p2, k3 = dis_mat
#             fx = im_mat[0, 0]; fy = im_mat[1, 1]
#             cx = im_mat[0, 2]; cy = im_mat[1, 2]

#             x_norm_dist = (x_cor - cx) / fx
#             y_norm_dist = (y_cor - cy) / fy
#             r2 = x_norm_dist ** 2 + y_norm_dist ** 2

#             radial = 1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
#             x_tan = 2 * p1 * x_norm_dist * y_norm_dist + p2 * (r2 + 2 * x_norm_dist ** 2)
#             y_tan = p1 * (r2 + 2 * y_norm_dist ** 2) + 2 * p2 * x_norm_dist * y_norm_dist

#             x_norm_undist = (x_norm_dist - x_tan) / radial
#             y_norm_undist = (y_norm_dist - y_tan) / radial

#             x_cor = x_norm_undist * fx + cx
#             y_cor = y_norm_undist * fy + cy
#         # ================================================================

#         # =============== stage1 decode -> boxes2d (for gate) ===============
#         # compute XYZ in camera(OpenCV) using current depth (normalized)
#         fx = im_mat[0, 0]; fy = im_mat[1, 1]
#         cx = im_mat[0, 2]; cy = im_mat[1, 2]
#         x_norm = (x_cor - cx) / fx
#         y_norm = (y_cor - cy) / fy
#         depth_col = depth.reshape(-1, 1)
#         X = x_norm * depth_col
#         Y = y_norm * depth_col
#         Z = depth_col
#         points = np.column_stack([X, Y, Z])
#         points_world = encode_box_centers_to_world(points, ex_mat)

#         stage1_box9d = np.concatenate([points_world, l, w, h, a1, a2, a3, cls_values], axis=-1)

#         boxes2d_gate = None
#         if lidar_two_stage and (lidar_depth_map is not None):
#             # stage1 boxes for gate
#             try:
#                 pred_boxes_params = stage1_box9d[:, :-1]  # drop cls
#                 pred_box9d_9points = convert_9params_to_9points(pred_boxes_params)
#                 boxes2d_gate, _ = box9d_to_2d(
#                     boxes9d=pred_box9d_9points,
#                     intrinsic_mat=im_mat,
#                     extrinsic_mat=ex_mat
#                 )
#             except Exception:
#                 boxes2d_gate = None
#         # ================================================================

#         # =============== stage2: lidar hard replace (normalized) ==========
#         if lidar_depth_map is not None:
#             depth = lidar_hard_calibrate_depth(
#                 x_pix=x_cor.reshape(-1),
#                 y_pix=y_cor.reshape(-1),
#                 depth_pred=depth,
#                 lidar_depth_map=lidar_depth_map,   # normalized
#                 hm_conf=hm_k,
#                 boxes2d=boxes2d_gate,              # comes from stage1
#                 win=lidar_win,
#                 invalid_val=0.0,
#                 min_count=lidar_min_count,
#                 max_spread=lidar_max_spread,
#                 hm_thr=lidar_hm_thr,
#                 delta0=lidar_delta0,
#                 rel_delta=lidar_rel_delta,
#                 box_center_only=lidar_box_center_only,
#                 box_center_ratio=lidar_box_center_ratio,
#                 debug=lidar_debug,
#             )

#             # recompute XYZ with calibrated depth
#             depth_col = depth.reshape(-1, 1)
#             X = x_norm * depth_col
#             Y = y_norm * depth_col
#             Z = depth_col
#             points = np.column_stack([X, Y, Z])
#             points_world = encode_box_centers_to_world(points, ex_mat)
#         # ================================================================

#         # final boxes
#         this_box9d = np.concatenate([points_world, l, w, h, a1, a2, a3, cls_values], axis=-1)
#         pred_boxes9d_all.append(this_box9d)
#         confidence_all.append(confi)

#     return np.concatenate(pred_boxes9d_all), np.concatenate(confidence_all)



# 只要 LiDAR patch 通过 min_count / max_spread / max_delta，就会把该点的 depth_pred[i] 直接替换成 LiDAR 的 median
# def _robust_depth_from_patch(patch: np.ndarray, invalid_val: float = 0.0):
#     """return (median_depth, valid_count, p90-p10 spread). depth单位随patch一致"""
#     vals = patch.reshape(-1)
#     vals = vals[vals > invalid_val]
#     if vals.size == 0:
#         return None, 0, None
#     vals = np.sort(vals)
#     cnt = vals.size
#     p10 = vals[int(0.1 * (cnt - 1))] if cnt > 1 else vals[0]
#     p90 = vals[int(0.9 * (cnt - 1))] if cnt > 1 else vals[0]
#     spread = float(p90 - p10)
#     return float(np.median(vals)), int(cnt), spread

# def lidar_hard_calibrate_depth(x_pix: np.ndarray, y_pix: np.ndarray, depth_pred: np.ndarray,
#                                lidar_depth_map: np.ndarray,
#                                win: int = 4,                # 半径，4=>9x9窗口
#                                invalid_val: float = 0.0,
#                                min_count: int = 3,
#                                max_spread: float = None,    # 米单位，比如 2.0；None表示不启用
#                                max_delta: float = None,     # 米单位，比如 10.0；None表示不启用
#                                debug: bool = False):
#     """对每个候选点：若LiDAR可靠 => 直接替换depth_pred[i]=median(lidar_patch)"""
#     H, W = lidar_depth_map.shape[:2]
#     K = depth_pred.shape[0]
#     out = depth_pred.copy()

#     used = 0
#     deltas = []
#     cnts = []
#     sprs = []

#     for i in range(K):
#         x = int(round(float(x_pix[i])))
#         y = int(round(float(y_pix[i])))
#         if x < 0 or x >= W or y < 0 or y >= H:
#             continue

#         x0, x1 = max(0, x - win), min(W, x + win + 1)
#         y0, y1 = max(0, y - win), min(H, y + win + 1)
#         patch = lidar_depth_map[y0:y1, x0:x1]

#         z_lidar, cnt, spr = _robust_depth_from_patch(patch, invalid_val=invalid_val)
#         if z_lidar is None or cnt < min_count:
#             continue
#         if max_spread is not None and spr is not None and spr > max_spread:
#             continue
#         if max_delta is not None and abs(z_lidar - float(depth_pred[i])) > max_delta:
#             continue

#         out[i] = z_lidar
#         used += 1
#         deltas.append(abs(z_lidar - float(depth_pred[i])))
#         cnts.append(cnt)
#         if spr is not None:
#             sprs.append(spr)

#     if debug:
#         print(f"[LiDAR calib HARD] used={used}/{K} win={2*win+1} min_count={min_count} max_spread={max_spread} max_delta={max_delta}")
#         if used > 0:
#             print(f"  cnt: mean={np.mean(cnts):.3f} p50={np.median(cnts):.3f} max={np.max(cnts):.0f}")
#             print(f"  spr: mean={np.mean(sprs) if sprs else 0:.3f} p50={np.median(sprs) if sprs else 0:.3f} max={np.max(sprs) if sprs else 0:.3f}")
#             print(f"  |Δ|: mean={np.mean(deltas):.3f} p50={np.median(deltas):.3f} max={np.max(deltas):.3f}")

#     return out

# def center_point_decoder(hm,
#                          center_res,
#                          center_dis,
#                          dim,
#                          rot,
#                          intrinsic_mat,
#                          extrinsic_mat,
#                          distortion_matrix,
#                          new_im_width=None,
#                          new_im_height=None,
#                          raw_im_width=None,
#                          raw_im_height=None,
#                          stride=None,
#                          im_num=None,
#                          max_num=10,
#                          lidar_depth_map=None,
#                          # ---- LiDAR calib params ----
#                          lidar_win=4,            # 9x9
#                          lidar_min_count=3,
#                          lidar_max_spread=2.0,   # meters; None to disable
#                          lidar_max_delta=10.0,   # meters; None to disable
#                          lidar_debug=False):
#     pred_boxes9d = []
#     all_confidence = []

#     hm = torch.tensor(hm) if isinstance(hm, np.ndarray) else hm
#     center_res = torch.tensor(center_res) if isinstance(center_res, np.ndarray) else center_res
#     center_dis = torch.tensor(center_dis) if isinstance(center_dis, np.ndarray) else center_dis
#     dim = torch.tensor(dim) if isinstance(dim, np.ndarray) else dim
#     rot = torch.tensor(rot) if isinstance(rot, np.ndarray) else rot

#     for im_id in range(im_num):
#         im_mat = intrinsic_mat
#         ex_mat = extrinsic_mat
#         dis_mat = distortion_matrix

#         this_hm = hm[im_id]  # (cls_num, Hf, Wf)
#         this_hm_conf, cls = this_hm.max(dim=0)

#         this_center_res = center_res[im_id]
#         this_center_dis = center_dis[im_id]
#         this_dim = dim[im_id]
#         this_rot = rot[im_id]

#         Hf, Wf = this_hm_conf.shape
#         confi_t, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)
#         confi = confi_t.detach().cpu().numpy()

#         rows = (linear_indices // Wf).long()
#         cols = (linear_indices % Wf).long()

#         res_x = this_center_res[0, rows, cols]
#         res_y = this_center_res[1, rows, cols]

#         cls_values = cls[rows, cols].detach().cpu().numpy().reshape(-1, 1)

#         # 预测深度（米）
#         depth = this_center_dis[0, rows, cols].detach().cpu().numpy().reshape(-1)  # (K,)

#         # 其他回归
#         l = this_dim[0, rows, cols].detach().cpu().numpy().reshape(-1, 1)
#         w = this_dim[1, rows, cols].detach().cpu().numpy().reshape(-1, 1)
#         h = this_dim[2, rows, cols].detach().cpu().numpy().reshape(-1, 1)

#         a1 = torch.atan2(this_rot[1, rows, cols], this_rot[0, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
#         a2 = torch.atan2(this_rot[3, rows, cols], this_rot[2, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
#         a3 = torch.atan2(this_rot[5, rows, cols], this_rot[4, rows, cols]).detach().cpu().numpy().reshape(-1, 1)

#         # heatmap坐标 -> 像素坐标（畸变前）
#         u_heatmap = cols.float() + res_x
#         v_heatmap = rows.float() + res_y
#         x_cor = (u_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)  # (K,1)
#         y_cor = (v_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)  # (K,1)

#         # ===================== 畸变校正（先做这个！）=====================
#         if dis_mat is not None and np.any(dis_mat != 0):
#             k1, k2, p1, p2, k3 = dis_mat
#             fx = im_mat[0, 0]; fy = im_mat[1, 1]
#             cx = im_mat[0, 2]; cy = im_mat[1, 2]

#             x_norm_dist = (x_cor - cx) / fx
#             y_norm_dist = (y_cor - cy) / fy
#             r2 = x_norm_dist ** 2 + y_norm_dist ** 2

#             radial = 1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
#             x_tan = 2 * p1 * x_norm_dist * y_norm_dist + p2 * (r2 + 2 * x_norm_dist ** 2)
#             y_tan = p1 * (r2 + 2 * y_norm_dist ** 2) + 2 * p2 * x_norm_dist * y_norm_dist

#             x_norm_undist = (x_norm_dist - x_tan) / radial
#             y_norm_undist = (y_norm_dist - y_tan) / radial

#             x_cor = x_norm_undist * fx + cx
#             y_cor = y_norm_undist * fy + cy
#         # =============================================================

#         # ===== LiDAR 推理硬校准：用“去畸变后的像素坐标”去索引 depth map =====
#         if lidar_depth_map is not None:
#             # lidar_hard_calibrate_depth 期望 x/y 是 (K,) 或 (K,1) 都行，但内部我建议用 (K,)
#             x_pix = x_cor.reshape(-1)  # (K,)
#             y_pix = y_cor.reshape(-1)  # (K,)

#             depth = lidar_hard_calibrate_depth(
#                 x_pix=x_pix,
#                 y_pix=y_pix,
#                 depth_pred=depth,                 # (K,)
#                 lidar_depth_map=lidar_depth_map,  # (H,W) meters
#                 win=lidar_win,
#                 invalid_val=0.0,
#                 min_count=lidar_min_count,
#                 max_spread=lidar_max_spread,
#                 max_delta=lidar_max_delta,
#                 debug=lidar_debug
#             )
#         # =====================================================================

#         # 计算 X,Y,Z（此时 x_cor/y_cor 已去畸变）
#         fx = im_mat[0, 0]; fy = im_mat[1, 1]
#         cx = im_mat[0, 2]; cy = im_mat[1, 2]

#         x_norm = (x_cor - cx) / fx
#         y_norm = (y_cor - cy) / fy

#         depth_col = depth.reshape(-1, 1)
#         X = x_norm * depth_col
#         Y = y_norm * depth_col
#         Z = depth_col

#         points = np.column_stack([X, Y, Z])
#         points_world = encode_box_centers_to_world(points, ex_mat)

#         this_box9d = np.concatenate([points_world, l, w, h, a1, a2, a3, cls_values], axis=-1)
#         pred_boxes9d.append(this_box9d)
#         all_confidence.append(confi)

#     return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)



# 新增hm_conf门控，只有被检测“比较确信是目标中心”的 peak（hm_conf >= hm_thr）才允许用 LiDAR 去替换深度
# def _robust_depth_from_patch(patch: np.ndarray, invalid_val: float = 0.0):
#     """
#     patch: 2D depth patch, unit == patch unit (HERE: meters)
#     returns:
#       z_med (float or None),
#       valid_count (int),
#       spread (float or None)  # p90 - p10, meters
#     """
#     vals = patch.reshape(-1).astype(np.float32)
#     vals = vals[vals > invalid_val]
#     if vals.size == 0:
#         return None, 0, None

#     vals.sort()
#     cnt = int(vals.size)
#     if cnt == 1:
#         return float(vals[0]), 1, 0.0

#     p10 = float(vals[int(0.1 * (cnt - 1))])
#     p90 = float(vals[int(0.9 * (cnt - 1))])
#     spread = p90 - p10
#     z_med = float(np.median(vals))
#     return z_med, cnt, spread

# def lidar_hard_calibrate_depth(
#     x_pix: np.ndarray,
#     y_pix: np.ndarray,
#     depth_pred: np.ndarray,
#     lidar_depth_map: np.ndarray,      # (H,W) meters
#     win: int = 4,                     # radius, 4 => 9x9
#     invalid_val: float = 0.0,
#     min_count: int = 3,
#     max_spread: float = 2.0,          # meters; None/<=0 disables
#     max_delta: float = 10.0,          # meters; None/<=0 disables
#     hm_conf: np.ndarray = None,       # (K,) optional
#     hm_thr: float = 0.35,             # only replace if hm_conf >= hm_thr; None/<=0 disables
#     debug: bool = False
# ):
#     """
#     For each candidate i:
#       if LiDAR patch is reliable => replace depth_pred[i] = median(lidar_patch)

#     Reliability gates (ALL in meters):
#       - (optional) hm_conf gate: require hm_conf[i] >= hm_thr
#       - min_count valid pixels in patch
#       - (optional) spread gate: p90-p10 <= max_spread
#       - (optional) delta gate: |z_lidar - z_pred| <= max_delta
#     """
#     H, W = lidar_depth_map.shape[:2]
#     xs = np.asarray(x_pix, dtype=np.float32).reshape(-1)
#     ys = np.asarray(y_pix, dtype=np.float32).reshape(-1)
#     out = np.asarray(depth_pred, dtype=np.float32).reshape(-1).copy()

#     K = out.shape[0]
#     used = 0
#     deltas, cnts, sprs = [], [], []
#     rej_hm = rej_oob = rej_cnt = rej_spr = rej_delta = 0

#     use_hm_gate = (hm_conf is not None) and (hm_thr is not None) and (hm_thr > 0)
#     if use_hm_gate:
#         hm_conf = np.asarray(hm_conf, dtype=np.float32).reshape(-1)
#         if hm_conf.shape[0] != K:
#             raise ValueError(f"hm_conf shape mismatch: {hm_conf.shape} vs K={K}")

#     use_spread_gate = (max_spread is not None) and (max_spread > 0)
#     use_delta_gate = (max_delta is not None) and (max_delta > 0)

#     for i in range(K):
#         # ✅ 关键：没检测到目标就不替换（避免背景 LiDAR 深度把预测搞乱）
#         if use_hm_gate and float(hm_conf[i]) < float(hm_thr):
#             rej_hm += 1
#             continue

#         xi = int(round(float(xs[i])))
#         yi = int(round(float(ys[i])))
#         if xi < 0 or xi >= W or yi < 0 or yi >= H:
#             rej_oob += 1
#             continue

#         x0, x1 = max(0, xi - win), min(W, xi + win + 1)
#         y0, y1 = max(0, yi - win), min(H, yi + win + 1)
#         patch = lidar_depth_map[y0:y1, x0:x1]

#         z_lidar, cnt, spr = _robust_depth_from_patch(patch, invalid_val=invalid_val)
#         if z_lidar is None or cnt < int(min_count):
#             rej_cnt += 1
#             continue

#         if use_spread_gate and spr is not None and float(spr) > float(max_spread):
#             rej_spr += 1
#             continue

#         z_pred = float(out[i])
#         if use_delta_gate and abs(z_lidar - z_pred) > float(max_delta):
#             rej_delta += 1
#             continue

#         out[i] = float(z_lidar)
#         used += 1
#         deltas.append(abs(z_lidar - z_pred))
#         cnts.append(cnt)
#         sprs.append(float(spr if spr is not None else 0.0))

#     if debug:
#         print(f"[LiDAR calib HARD@decoder] used={used}/{K} win={2*win+1} min_count={min_count} "
#               f"hm_thr={hm_thr if use_hm_gate else None} max_spread={max_spread if use_spread_gate else None} "
#               f"max_delta={max_delta if use_delta_gate else None}")
#         print(f"  rej: hm={rej_hm} oob={rej_oob} cnt={rej_cnt} spr={rej_spr} delta={rej_delta}")
#         if used > 0:
#             print(f"  cnt: mean={np.mean(cnts):.3f} p50={np.median(cnts):.3f} max={np.max(cnts):.0f}")
#             print(f"  spr: mean={np.mean(sprs):.3f} p50={np.median(sprs):.3f} max={np.max(sprs):.3f}")
#             print(f"  |Δ|: mean={np.mean(deltas):.3f} p50={np.median(deltas):.3f} max={np.max(deltas):.3f}")

#     return out

# def center_point_decoder(
#     hm,
#     center_res,
#     center_dis,
#     dim,
#     rot,
#     intrinsic_mat,
#     extrinsic_mat,
#     distortion_matrix,
#     new_im_width=None,
#     new_im_height=None,
#     raw_im_width=None,
#     raw_im_height=None,
#     stride=None,
#     im_num=None,
#     max_num=10,
#     lidar_depth_map=None,          # (H,W) meters
#     # ---- LiDAR calib params (ALL meters except hm_thr) ----
#     lidar_win=4,
#     lidar_min_count=5,
#     lidar_max_spread=None,          # meters
#     lidar_max_delta=15.0,          # meters
#     lidar_hm_thr=0.35,             # heatmap conf gate
#     lidar_debug=True
# ):
#     pred_boxes9d = []
#     all_confidence = []

#     hm = torch.tensor(hm) if isinstance(hm, np.ndarray) else hm
#     center_res = torch.tensor(center_res) if isinstance(center_res, np.ndarray) else center_res
#     center_dis = torch.tensor(center_dis) if isinstance(center_dis, np.ndarray) else center_dis
#     dim = torch.tensor(dim) if isinstance(dim, np.ndarray) else dim
#     rot = torch.tensor(rot) if isinstance(rot, np.ndarray) else rot

#     for im_id in range(im_num):
#         im_mat = intrinsic_mat
#         ex_mat = extrinsic_mat
#         dis_mat = distortion_matrix

#         this_hm = hm[im_id]  # (cls_num, Hf, Wf)
#         this_hm_conf, cls = this_hm.max(dim=0)  # (Hf,Wf)

#         this_center_res = center_res[im_id]
#         this_center_dis = center_dis[im_id]     # IMPORTANT: meters already
#         this_dim = dim[im_id]
#         this_rot = rot[im_id]

#         Hf, Wf = this_hm_conf.shape
#         confi_t, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)
#         confi = confi_t.detach().cpu().numpy()

#         rows = (linear_indices // Wf).long()
#         cols = (linear_indices % Wf).long()

#         res_x = this_center_res[0, rows, cols]
#         res_y = this_center_res[1, rows, cols]

#         # ✅ hm confidence for each selected peak (gate lidar replace)
#         hm_k = this_hm_conf[rows, cols].detach().cpu().numpy().reshape(-1)

#         cls_values = cls[rows, cols].detach().cpu().numpy().reshape(-1, 1)

#         # depth at peaks (meters)
#         depth = this_center_dis[0, rows, cols].detach().cpu().numpy().reshape(-1).astype(np.float32)

#         # other regressions
#         l = this_dim[0, rows, cols].detach().cpu().numpy().reshape(-1, 1)
#         w = this_dim[1, rows, cols].detach().cpu().numpy().reshape(-1, 1)
#         h = this_dim[2, rows, cols].detach().cpu().numpy().reshape(-1, 1)

#         a1 = torch.atan2(this_rot[1, rows, cols], this_rot[0, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
#         a2 = torch.atan2(this_rot[3, rows, cols], this_rot[2, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
#         a3 = torch.atan2(this_rot[5, rows, cols], this_rot[4, rows, cols]).detach().cpu().numpy().reshape(-1, 1)

#         # heatmap coords -> pixel coords (distorted)
#         u_heatmap = cols.float() + res_x
#         v_heatmap = rows.float() + res_y
#         x_cor = (u_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)
#         y_cor = (v_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)

#         # ---- undistort pixel coords (so lidar indexing aligns with your depth map if it's undistorted) ----
#         if dis_mat is not None and np.any(dis_mat != 0):
#             k1, k2, p1, p2, k3 = dis_mat
#             fx = im_mat[0, 0]; fy = im_mat[1, 1]
#             cx = im_mat[0, 2]; cy = im_mat[1, 2]

#             x_norm_dist = (x_cor - cx) / fx
#             y_norm_dist = (y_cor - cy) / fy
#             r2 = x_norm_dist ** 2 + y_norm_dist ** 2

#             radial = 1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
#             x_tan = 2 * p1 * x_norm_dist * y_norm_dist + p2 * (r2 + 2 * x_norm_dist ** 2)
#             y_tan = p1 * (r2 + 2 * y_norm_dist ** 2) + 2 * p2 * x_norm_dist * y_norm_dist

#             x_norm_undist = (x_norm_dist - x_tan) / radial
#             y_norm_undist = (y_norm_dist - y_tan) / radial

#             x_cor = x_norm_undist * fx + cx
#             y_cor = y_norm_undist * fy + cy

#         # ---- HARD lidar replace ONLY at confident peaks ----
#         if lidar_depth_map is not None:
#             depth = lidar_hard_calibrate_depth(
#                 x_pix=x_cor.reshape(-1),
#                 y_pix=y_cor.reshape(-1),
#                 depth_pred=depth,
#                 lidar_depth_map=lidar_depth_map,   # meters
#                 win=lidar_win,
#                 invalid_val=0.0,
#                 min_count=lidar_min_count,
#                 max_spread=lidar_max_spread,
#                 max_delta=lidar_max_delta,
#                 hm_conf=hm_k,
#                 hm_thr=lidar_hm_thr,
#                 debug=lidar_debug
#             )

#         # ---- XYZ in camera ----
#         fx = im_mat[0, 0]; fy = im_mat[1, 1]
#         cx = im_mat[0, 2]; cy = im_mat[1, 2]

#         x_norm = (x_cor - cx) / fx
#         y_norm = (y_cor - cy) / fy

#         depth_col = depth.reshape(-1, 1)
#         X = x_norm * depth_col
#         Y = y_norm * depth_col
#         Z = depth_col

#         points = np.column_stack([X, Y, Z])
#         points_world = encode_box_centers_to_world(points, ex_mat)

#         this_box9d = np.concatenate([points_world, l, w, h, a1, a2, a3, cls_values], axis=-1)
#         pred_boxes9d.append(this_box9d)
#         all_confidence.append(confi)

#     return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)


# 聚类版本，改进LiDAR 的median，使用距离近的聚类median去替换预测深度
def _robust_depth_from_patch(
    patch: np.ndarray,
    invalid_val: float = 0.0,
    # ---- near-cluster params (meters) ----
    near_k: int = 10,                 # 只看最近的 K 个点来找“目标簇”
    min_cluster: int = 3,             # 至少 3 个点才算目标簇（你说的2-3个，我建议先用3更稳）
    near_spread_thr: float = 0.6,     # 小簇内部最大深度差(米)，越小越严格
    bg_sep_thr: float = 3.0,          # 背景中位数 - 近簇深度 >= 该值(米)，才认为近簇是“目标”
    bg_min_count: int = 5,            # 背景点太少就不做分离判定（或用弱判定）
):
    """
    patch: 2D depth patch, unit == meters
    returns:
      z_pick (float or None): 选出来用于替换的深度（近簇median）
      valid_count (int): patch内有效点总数
      info (dict): 方便debug的统计信息
        - z_cluster, cluster_cnt, cluster_spread
        - z_bg_med, sep_to_bg
    """
    vals = patch.reshape(-1).astype(np.float32)
    vals = vals[vals > float(invalid_val)]
    if vals.size == 0:
        return None, 0, {}

    vals.sort()
    cnt = int(vals.size)
    if cnt < int(min_cluster):
        return None, cnt, {"reason": "too_few_valid"}

    K = int(min(near_k, cnt))
    near = vals[:K]  # 最近的K个点

    # 在 near 中找一个长度=min_cluster 的“最紧子段”（max-min最小）
    m = int(min_cluster)
    best_spread = None
    best_j = None
    for j in range(0, K - m + 1):
        sp = float(near[j + m - 1] - near[j])
        if best_spread is None or sp < best_spread:
            best_spread = sp
            best_j = j

    if best_j is None:
        return None, cnt, {"reason": "no_cluster"}

    cluster = near[best_j: best_j + m]
    z_cluster = float(np.median(cluster))
    cluster_spread = float(cluster[-1] - cluster[0])

    # 近簇内部必须足够“接近”
    if near_spread_thr is not None and near_spread_thr > 0 and cluster_spread > float(near_spread_thr):
        return None, cnt, {
            "reason": "cluster_spread_too_large",
            "z_cluster": z_cluster,
            "cluster_cnt": m,
            "cluster_spread": cluster_spread
        }

    # 背景分离：用剩余点的 median 做“背景典型深度”
    info = {
        "z_cluster": z_cluster,
        "cluster_cnt": m,
        "cluster_spread": cluster_spread,
    }

    bg = vals[K:]  # 剩余认为偏背景（更远的点）
    if bg.size >= int(bg_min_count):
        z_bg_med = float(np.median(bg))
        sep = float(z_bg_med - z_cluster)
        info["z_bg_med"] = z_bg_med
        info["sep_to_bg"] = sep

        if bg_sep_thr is not None and bg_sep_thr > 0 and sep < float(bg_sep_thr):
            # 背景不够“远”，近簇可能不是目标（可能是噪声/路面/同一平面）
            info["reason"] = "bg_not_separated"
            return None, cnt, info
    else:
        # 背景点太少：不做分离判定（弱通过），但把信息记下来
        info["z_bg_med"] = None
        info["sep_to_bg"] = None
        info["reason"] = "bg_too_few_skip_sep"

    return z_cluster, cnt, info

def lidar_hard_calibrate_depth(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    depth_pred: np.ndarray,
    lidar_depth_map: np.ndarray,      # (H,W) meters
    win: int = 4,                     # radius, 4 => 9x9
    invalid_val: float = 0.0,
    min_count: int = 3,               # patch内总有效点最低要求（先过滤“太稀”）
    # ---- old gates (meters) ----
    max_delta: float = 15.0,          # |z_lidar - z_pred| <= max_delta
    hm_conf: np.ndarray = None,       # (K,) optional
    hm_thr: float = 0.35,             # hm gate
    # ---- NEW: near-cluster params (meters) ----
    near_k: int = 10,
    min_cluster: int = 3,
    near_spread_thr: float = 0.6,
    bg_sep_thr: float = 3.0,
    bg_min_count: int = 5,
    # ---- debug ----
    debug: bool = False
):
    """
    用“最近点小簇”替换 depth_pred[i]：
      - hm_conf gate（可选）
      - patch 有足够 valid 点
      - 在 patch 里能找到：最近点里的紧簇(min_cluster) 且簇spread小
      - 且该紧簇与背景 median 分离 >= bg_sep_thr
      - 且与预测差值不离谱 (max_delta)
    """
    H, W = lidar_depth_map.shape[:2]
    xs = np.asarray(x_pix, dtype=np.float32).reshape(-1)
    ys = np.asarray(y_pix, dtype=np.float32).reshape(-1)
    out = np.asarray(depth_pred, dtype=np.float32).reshape(-1).copy()
    K = out.shape[0]

    used = 0
    deltas, cnts = [], []
    rej_hm = rej_oob = rej_cnt = rej_pick = rej_delta = 0

    use_hm_gate = (hm_conf is not None) and (hm_thr is not None) and (hm_thr > 0)
    if use_hm_gate:
        hm_conf = np.asarray(hm_conf, dtype=np.float32).reshape(-1)
        if hm_conf.shape[0] != K:
            raise ValueError(f"hm_conf shape mismatch: {hm_conf.shape} vs K={K}")

    use_delta_gate = (max_delta is not None) and (max_delta > 0)

    # debug stats
    dbg_reasons = {}

    for i in range(K):
        if use_hm_gate and float(hm_conf[i]) < float(hm_thr):
            rej_hm += 1
            continue

        xi = int(round(float(xs[i])))
        yi = int(round(float(ys[i])))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            rej_oob += 1
            continue

        x0, x1 = max(0, xi - win), min(W, xi + win + 1)
        y0, y1 = max(0, yi - win), min(H, yi + win + 1)
        patch = lidar_depth_map[y0:y1, x0:x1]

        # 总有效点先过滤一次
        vals = patch.reshape(-1).astype(np.float32)
        vals = vals[vals > float(invalid_val)]
        if int(vals.size) < int(min_count):
            rej_cnt += 1
            continue

        z_pick, vcnt, info = _robust_depth_from_patch(
            patch,
            invalid_val=invalid_val,
            near_k=near_k,
            min_cluster=min_cluster,
            near_spread_thr=near_spread_thr,
            bg_sep_thr=bg_sep_thr,
            bg_min_count=bg_min_count,
        )
        if z_pick is None:
            rej_pick += 1
            if debug:
                r = info.get("reason", "unknown")
                dbg_reasons[r] = dbg_reasons.get(r, 0) + 1
            continue

        z_pred = float(out[i])
        if use_delta_gate and abs(float(z_pick) - z_pred) > float(max_delta):
            rej_delta += 1
            if debug:
                dbg_reasons["delta_too_large"] = dbg_reasons.get("delta_too_large", 0) + 1
            continue

        out[i] = float(z_pick)
        used += 1
        deltas.append(abs(float(z_pick) - z_pred))
        cnts.append(int(vcnt))

    if debug:
        print(
            f"[LiDAR calib NEAR-CLUSTER@decoder] used={used}/{K} "
            f"win={2*win+1} min_count={min_count} hm_thr={hm_thr if use_hm_gate else None} "
            f"near_k={near_k} min_cluster={min_cluster} near_spread_thr={near_spread_thr} bg_sep_thr={bg_sep_thr} "
            f"max_delta={max_delta if use_delta_gate else None}"
        )
        print(f"  rej: hm={rej_hm} oob={rej_oob} cnt={rej_cnt} pick={rej_pick} delta={rej_delta}")
        if debug and dbg_reasons:
            print(f"  pick_fail_reasons: {dbg_reasons}")
        if used > 0:
            print(f"  cnt: mean={np.mean(cnts):.3f} p50={np.median(cnts):.3f} max={np.max(cnts):.0f}")
            print(f"  |Δ|: mean={np.mean(deltas):.3f} p50={np.median(deltas):.3f} max={np.max(deltas):.3f}")

    return out

def center_point_decoder(
    hm,
    center_res,
    center_dis,
    dim,
    rot,
    intrinsic_mat,
    extrinsic_mat,
    distortion_matrix,
    new_im_width=None,
    new_im_height=None,
    raw_im_width=None,
    raw_im_height=None,
    stride=None,
    im_num=None,
    max_num=10,
    lidar_depth_map=None,          # (H,W) meters
    # ---- LiDAR calib params (ALL meters except hm_thr) ----
    lidar_win=4,
    lidar_min_count=5,
    lidar_max_delta=15.0,
    lidar_hm_thr=0.35,
    # ---- NEW: near-cluster params ----
    lidar_near_k=10,
    lidar_min_cluster=3,
    lidar_near_spread_thr=0.2,
    lidar_bg_sep_thr=1.5,
    lidar_bg_min_count=5,
    lidar_debug=True
):
    pred_boxes9d = []
    all_confidence = []

    hm = torch.tensor(hm) if isinstance(hm, np.ndarray) else hm
    center_res = torch.tensor(center_res) if isinstance(center_res, np.ndarray) else center_res
    center_dis = torch.tensor(center_dis) if isinstance(center_dis, np.ndarray) else center_dis
    dim = torch.tensor(dim) if isinstance(dim, np.ndarray) else dim
    rot = torch.tensor(rot) if isinstance(rot, np.ndarray) else rot

    for im_id in range(im_num):
        im_mat = intrinsic_mat
        ex_mat = extrinsic_mat
        dis_mat = distortion_matrix

        this_hm = hm[im_id]  # (cls_num, Hf, Wf)
        this_hm_conf, cls = this_hm.max(dim=0)  # (Hf,Wf)

        this_center_res = center_res[im_id]
        this_center_dis = center_dis[im_id]     # ✅ meters already
        this_dim = dim[im_id]
        this_rot = rot[im_id]

        Hf, Wf = this_hm_conf.shape
        confi_t, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)
        confi = confi_t.detach().cpu().numpy()

        rows = (linear_indices // Wf).long()
        cols = (linear_indices % Wf).long()

        res_x = this_center_res[0, rows, cols]
        res_y = this_center_res[1, rows, cols]

        # hm confidence at selected peaks (for gate)
        hm_k = this_hm_conf[rows, cols].detach().cpu().numpy().reshape(-1)

        cls_values = cls[rows, cols].detach().cpu().numpy().reshape(-1, 1)

        # depth at peaks (meters)
        depth = this_center_dis[0, rows, cols].detach().cpu().numpy().reshape(-1).astype(np.float32)

        # other regressions
        l = this_dim[0, rows, cols].detach().cpu().numpy().reshape(-1, 1)
        w = this_dim[1, rows, cols].detach().cpu().numpy().reshape(-1, 1)
        h = this_dim[2, rows, cols].detach().cpu().numpy().reshape(-1, 1)

        a1 = torch.atan2(this_rot[1, rows, cols], this_rot[0, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
        a2 = torch.atan2(this_rot[3, rows, cols], this_rot[2, rows, cols]).detach().cpu().numpy().reshape(-1, 1)
        a3 = torch.atan2(this_rot[5, rows, cols], this_rot[4, rows, cols]).detach().cpu().numpy().reshape(-1, 1)

        # heatmap coords -> pixel coords (distorted)
        u_heatmap = cols.float() + res_x
        v_heatmap = rows.float() + res_y
        x_cor = (u_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)
        y_cor = (v_heatmap * stride).detach().cpu().numpy().reshape(-1, 1)

        # ---- undistort pixel coords (keep your original logic) ----
        if dis_mat is not None and np.any(dis_mat != 0):
            k1, k2, p1, p2, k3 = dis_mat
            fx = im_mat[0, 0]; fy = im_mat[1, 1]
            cx = im_mat[0, 2]; cy = im_mat[1, 2]

            x_norm_dist = (x_cor - cx) / fx
            y_norm_dist = (y_cor - cy) / fy
            r2 = x_norm_dist ** 2 + y_norm_dist ** 2

            radial = 1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
            x_tan = 2 * p1 * x_norm_dist * y_norm_dist + p2 * (r2 + 2 * x_norm_dist ** 2)
            y_tan = p1 * (r2 + 2 * y_norm_dist ** 2) + 2 * p2 * x_norm_dist * y_norm_dist

            x_norm_undist = (x_norm_dist - x_tan) / radial
            y_norm_undist = (y_norm_dist - y_tan) / radial

            x_cor = x_norm_undist * fx + cx
            y_cor = y_norm_undist * fy + cy

        # ---- HARD lidar replace at peaks using NEAR-CLUSTER ----
        if lidar_depth_map is not None:
            depth = lidar_hard_calibrate_depth(
                x_pix=x_cor.reshape(-1),
                y_pix=y_cor.reshape(-1),
                depth_pred=depth,
                lidar_depth_map=lidar_depth_map,   # meters
                win=lidar_win,
                invalid_val=0.0,
                min_count=lidar_min_count,
                max_delta=lidar_max_delta,
                hm_conf=hm_k,
                hm_thr=lidar_hm_thr,
                near_k=lidar_near_k,
                min_cluster=lidar_min_cluster,
                near_spread_thr=lidar_near_spread_thr,
                bg_sep_thr=lidar_bg_sep_thr,
                bg_min_count=lidar_bg_min_count,
                debug=lidar_debug
            )

        # ---- XYZ in camera ----
        fx = im_mat[0, 0]; fy = im_mat[1, 1]
        cx = im_mat[0, 2]; cy = im_mat[1, 2]

        x_norm = (x_cor - cx) / fx
        y_norm = (y_cor - cy) / fy

        depth_col = depth.reshape(-1, 1)
        X = x_norm * depth_col
        Y = y_norm * depth_col
        Z = depth_col

        points = np.column_stack([X, Y, Z])
        points_world = encode_box_centers_to_world(points, ex_mat)

        this_box9d = np.concatenate([points_world, l, w, h, a1, a2, a3, cls_values], axis=-1)
        pred_boxes9d.append(this_box9d)
        all_confidence.append(confi)

    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)


# 最初版本
# def center_point_decoder(hm,
#                          center_res,
#                          center_dis,
#                          dim,
#                          rot,
#                          intrinsic_mat,
#                          extrinsic_mat,
#                          distortion_matrix,
#                          new_im_width=None,
#                          new_im_height=None,
#                          raw_im_width=None,
#                          raw_im_height=None,
#                          stride=None,
#                          im_num=None,
#                          max_num=10, 
#                          lidar_depth_map=None):
#     pred_boxes9d = []
#     all_confidence = []

#     hm = torch.tensor(hm) if isinstance(hm, np.ndarray) else hm
#     center_res = torch.tensor(center_res) if isinstance(center_res, np.ndarray) else center_res
#     center_dis = torch.tensor(center_dis) if isinstance(center_dis, np.ndarray) else center_dis
#     dim = torch.tensor(dim) if isinstance(dim, np.ndarray) else dim
#     rot = torch.tensor(rot) if isinstance(rot, np.ndarray) else rot

#     for im_id in range(im_num):
#         im_mat = intrinsic_mat
#         ex_mat = extrinsic_mat
#         dis_mat = distortion_matrix  # 畸变矩阵（k1,k2,p1,p2,k3）
#         this_hm = hm[im_id]  # (cls_num, hm_height, hm_width)

#         this_hm_conf, cls = this_hm.max(dim=0)
#         this_center_res = center_res[im_id]
#         this_center_dis = center_dis[im_id]
#         this_dim = dim[im_id]
#         this_rot = rot[im_id]
#         shape_hm = this_hm_conf.shape
#         confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=3)
#         confi = confi.detach().cpu().numpy()
#         c_num = shape_hm[-1]
#         rows = linear_indices // c_num  # K
#         cols = linear_indices % c_num  # K 
#         rows_ind = rows.long()
#         cols_ind = cols.long()

#         res_x = this_center_res[0, rows_ind, cols_ind]
#         res_y = this_center_res[1, rows_ind, cols_ind]

#         cls_values = cls[rows_ind, cols_ind].cpu().numpy().reshape(-1, 1)
#         depth = this_center_dis[0, rows_ind, cols_ind].cpu().numpy()
#         l = this_dim[0, rows_ind, cols_ind].cpu().numpy().reshape(-1, 1)
#         w = this_dim[1, rows_ind, cols_ind].cpu().numpy().reshape(-1, 1)
#         h = this_dim[2, rows_ind, cols_ind].cpu().numpy().reshape(-1, 1)

#         a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind]).cpu().numpy().reshape(-1, 1)
#         a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind]).cpu().numpy().reshape(-1, 1)
#         a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind]).cpu().numpy().reshape(-1, 1)

#         # 1. 从热力图索引和残差还原 u_heatmap 和 v_heatmap
#         u_heatmap = cols.float() + res_x  # 对应编码器的 u_heatmap = u * (1/stride)
#         v_heatmap = rows.float() + res_y  # 对应编码器的 v_heatmap = v * (1/stride)

#         # 2. 还原为图像像素坐标（仅乘以stride，与编码器严格互逆）
#         x_cor = u_heatmap * stride  # 对应编码器的 u = u_heatmap * stride
#         y_cor = v_heatmap * stride  # 对应编码器的 v = v_heatmap * stride

#         # 3. 转换为numpy数组
#         x_cor = x_cor.cpu().numpy().reshape(-1, 1)
#         y_cor = y_cor.cpu().numpy().reshape(-1, 1)

#         # ===================== 畸变校正 =====================
#         # 仅当畸变矩阵非空/非全0时执行校正，否则跳过
#         if dis_mat is not None and np.any(dis_mat != 0):
#             # 提取OpenCV标准畸变参数：k1, k2, p1, p2, k3（顺序固定）
#             k1, k2, p1, p2, k3 = dis_mat
            
#             # 步骤1：将像素坐标转换为归一化相机坐标（去畸变的前置步骤）
#             fx = im_mat[0, 0]
#             fy = im_mat[1, 1]
#             cx = im_mat[0, 2]
#             cy = im_mat[1, 2]
#             x_norm_dist = (x_cor - cx) / fx  # 畸变的归一化x
#             y_norm_dist = (y_cor - cy) / fy  # 畸变的归一化y

#             # 步骤2：计算径向畸变和切向畸变校正项（OpenCV标准公式）
#             r2 = x_norm_dist ** 2 + y_norm_dist ** 2  # 径向距离平方
#             # 径向畸变校正因子
#             radial = 1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
#             # 切向畸变校正
#             x_tan = 2 * p1 * x_norm_dist * y_norm_dist + p2 * (r2 + 2 * x_norm_dist ** 2)
#             y_tan = p1 * (r2 + 2 * y_norm_dist ** 2) + 2 * p2 * x_norm_dist * y_norm_dist

#             # 步骤3：计算去畸变后的归一化坐标
#             x_norm_undist = (x_norm_dist - x_tan) / radial
#             y_norm_undist = (y_norm_dist - y_tan) / radial

#             # 步骤4：还原为去畸变后的像素坐标（用于后续计算，保持逻辑统一）
#             x_cor = x_norm_undist * fx + cx
#             y_cor = y_norm_undist * fy + cy
#         # =====================================================================

#         # 4. 计算X、Y（与编码器公式互逆，此时x_cor/y_cor已去畸变）
#         fx = im_mat[0, 0]
#         fy = im_mat[1, 1]
#         cx = im_mat[0, 2]
#         cy = im_mat[1, 2]

#         x_norm = (x_cor - cx) / fx  # 对应编码器 (X/Z) = (u - cx)/fx
#         y_norm = (y_cor - cy) / fy  # 对应编码器 (Y/Z) = (v - cy)/fy

#         depth = depth.reshape(-1, 1)
#         X = x_norm * depth
#         Y = y_norm * depth
#         Z = depth

#         points = np.column_stack([X, Y, Z])
#         points_world = encode_box_centers_to_world(points, ex_mat)

#         # x,y,z,l,w,h,a1,a2,a3,cls
#         this_box9d = np.concatenate([points_world, l, w, h, a1, a2, a3, cls_values], axis=-1)
#         pred_boxes9d.append(this_box9d)
#         all_confidence.append(confi)

#     return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)

# def center_point_decoder(hm,
#                          center_res,
#                          center_dis,
#                          dim,
#                          rot,
#                          intrinsic_mat,
#                          extrinsic_mat,
#                          distortion_matrix,
#                          new_im_width=None,
#                          new_im_height=None,
#                          raw_im_width=None,
#                          raw_im_height=None,
#                          stride=None,
#                          im_num=None,
#                          max_num=10):
#     pred_boxes9d = []
#     all_confidence = []

#     hm = torch.tensor(hm) if isinstance(hm, np.ndarray) else hm
#     center_res = torch.tensor(center_res) if isinstance(center_res, np.ndarray) else center_res
#     center_dis = torch.tensor(center_dis) if isinstance(center_dis, np.ndarray) else center_dis
#     dim = torch.tensor(dim) if isinstance(dim, np.ndarray) else dim
#     rot = torch.tensor(rot) if isinstance(rot, np.ndarray) else rot

#     for im_id in range(im_num):
#         im_mat = intrinsic_mat
#         ex_mat = extrinsic_mat
#         dis_mat = distortion_matrix

#         this_hm = hm[im_id]  # (cls_num, hm_height, hm_width)
#         this_hm_conf, cls_map = this_hm.max(0)

#         this_center_res = center_res[im_id]
#         this_center_dis = center_dis[im_id]
#         this_dim = dim[im_id]
#         this_rot = rot[im_id]

#         shape_hm = this_hm_conf.shape

#         # Call NMS instead of simple topk

#         confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=2)  # (N,) (N,)

#         confi = confi.detach().cpu().numpy()

#         c_num = shape_hm[-1]
#         rows = linear_indices // c_num  # K
#         cols = linear_indices % c_num  # K

#         rows_ind = rows.long()
#         cols_ind = cols.long()

#         res_x = this_center_res[0, rows_ind, cols_ind]
#         res_y = this_center_res[1, rows_ind, cols_ind]

#         cls = cls_map[rows_ind, cols_ind].cpu().numpy()
#         cls = cls.reshape(-1, 1)

#         depth = this_center_dis[0, rows_ind, cols_ind].detach().cpu().numpy()
#         y_cor = rows + res_y
#         x_cor = cols + res_x
#         y_cor = y_cor * stride
#         x_cor = x_cor * stride
#         y_cor = y_cor.detach().cpu().numpy()
#         x_cor = x_cor.detach().cpu().numpy()

#         x_cor = x_cor.reshape(-1, 1)
#         y_cor = y_cor.reshape(-1, 1)

#         points = backProject_with_opencv_to_world(np.concatenate([x_cor, y_cor], -1), depth, ex_mat[:3, :3],
#                                                   ex_mat[:3, 3], im_mat, dis_mat)

#         l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

#         a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind])
#         a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind])
#         a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind])

#         l = l.detach().cpu().numpy().reshape(-1, 1)
#         w = w.detach().cpu().numpy().reshape(-1, 1)
#         h = h.detach().cpu().numpy().reshape(-1, 1)
#         a1 = a1.detach().cpu().numpy().reshape(-1, 1)
#         a2 = a2.detach().cpu().numpy().reshape(-1, 1)
#         a3 = a3.detach().cpu().numpy().reshape(-1, 1)

#         this_box9d = np.concatenate([points, l, w, h, a1, a2, a3, cls], -1)

#         pred_boxes9d.append(this_box9d)
#         all_confidence.append(confi)

#     return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)


def encode_box_centers_to_world(centers_cv, extrinsic_mat):
    """
    批量将OpenCV相机坐标系下的中心点转换为世界坐标系
    
    参数：
        centers_cv: (N, 3)，OpenCV相机系坐标（X右、Y下、Z前），支持批量点
        extrinsic_mat: 4x4齐次外参矩阵（Carla相机系→世界系的转换矩阵）
    
    返回：
        centers_world: (N, 3)，世界坐标系坐标，与输入点一一对应
    """
    # 定义OpenCV→Carla相机系的转换矩阵（与carla_to_opencv互逆）
    opencv_to_carla = np.linalg.inv(np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]))

    # 转换为齐次坐标（添加w=1），支持批量处理 (N, 3) → (N, 4)
    centers_cv_hom = np.hstack([centers_cv, np.ones((centers_cv.shape[0], 1))])

    # Step 1: OpenCV相机系 → Carla相机系（批量矩阵乘法）
    centers_carla = (opencv_to_carla @ centers_cv_hom.T).T  # 结果为 (N, 4)

    # Step 2: Carla相机系 → 世界坐标系（批量应用外参矩阵）
    centers_world_hom = (extrinsic_mat @ centers_carla.T).T  # 结果为 (N, 4)

    return centers_world_hom[:, :3].astype(np.float32)  # 取前3列，形状 (N, 3)


all_object_encoders = {'key_point_encoder': key_point_encoder,
                       'key_point_decoder': key_point_decoder,
                       'center_point_encoder': center_point_encoder,
                       'center_point_decoder': center_point_decoder}
