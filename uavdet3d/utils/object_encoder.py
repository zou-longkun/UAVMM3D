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
    # === Step 1: ���� �� ��һ��������꣨OpenCV ���ϵ�� ===
    corners_2D_homogeneous = np.hstack((corners_2D, np.ones((corners_2D.shape[0], 1))))
    intrinsic_inv = np.linalg.inv(intrinsic_mat)
    corners_norm = (intrinsic_inv @ corners_2D_homogeneous.T).T  # shape (N, 3)

    # === Step 2: ȥ���� ===
    x = corners_norm[:, 0]
    y = corners_norm[:, 1]
    r2 = x ** 2 + y ** 2
    k1, k2, p1, p2, k3 = distortion_matrix

    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_undist = (x - 2 * p1 * x * y - p2 * (r2 + 2 * x ** 2)) / radial
    y_undist = (y - p1 * (r2 + 2 * y ** 2) - 2 * p2 * x * y) / radial

    # === Step 3: �õ� OpenCV �������ϵ�µĵ� ===
    points_opencv = np.vstack((x_undist * depth, y_undist * depth, depth)).T  # (N, 3)

    # === Step 4: OpenCV �� Carla �������ϵ ===
    opencv_to_carla = np.linalg.inv(np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]))

    points_opencv_hom = np.hstack([points_opencv, np.ones((points_opencv.shape[0], 1))])  # (N, 4)
    points_carla = (opencv_to_carla @ points_opencv_hom.T).T[:, :3]  # ȥ�����

    # === Step 5: Carla �������ϵ �� �������꣨ʹ�� extrinsic�� ===
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

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box

        # Scale the corner points to match the box dimensions
        corners = corners_local * np.array([l, w, h])

        # Rotate the corner points
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)

        # Translate the corner points to the box center
        corners += np.array([x, y, z])

        # corners_2d, _ = cv2.projectPoints(corners, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], intrinsic_mat,
        #                                   distortion_matrix)
        # corners_2d = corners_2d.reshape(-1, 2)

        # all_corners_3d.append(corners)
        # all_corners_2d.append(corners_2d)

        pts_cam = corners.T  # (3, N)
        proj = (intrinsic_mat @ pts_cam).T  # (N, 3)
        proj = proj[:, :2] / proj[:, 2:3]  # ��һ��

        all_corners_3d.append(corners)
        all_corners_2d.append(proj)

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

    for im_id in range(im_num):
        for ob_id in range(obj_num):

            all_conf = []
            for k_id in range(key_pts_num):
                this_heatmap = pred_heat_map[im_id, ob_id, k_id]
                this_res_x = pred_res_x[im_id, ob_id, k_id]
                this_res_y = pred_res_y[im_id, ob_id, k_id]

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

                key_points_2d[im_id, ob_id, k_id, 0] = (x_cor / delta0)
                key_points_2d[im_id, ob_id, k_id, 1] = (y_cor / delta1)

            confidence[im_id, ob_id] = torch.mean(torch.stack(all_conf))

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
    cls_num = len(class_name_config)

    gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []

    for im_id in range(im_num):
        in_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        xyz = copy.deepcopy(gt_box9d_with_cls[:, 0, :])

        # ptx_im, depth = projectPoints(xyz, ex_mat[:3, :3], ex_mat[:3, 3], in_mat,dis_mat*0)#
        pts_cam = xyz.T  # (3, N)
        pts_proj = (in_mat @ pts_cam).T  # (N, 3)
        ptx_im = pts_proj[:, :2] / pts_proj[:, 2:3]  # (N, 2)
        depth = pts_proj[:, 2:3]
        ptx_im = ptx_im.reshape(-1, 2)
        depth = depth.reshape(-1, 1)

        this_heat_map = torch.zeros(cls_num, new_im_hight // stride, new_im_width // stride)

        this_res_map = np.zeros(shape=(2, new_im_hight // stride, new_im_width // stride))

        this_dis_map = np.zeros(shape=(1, new_im_hight // stride, new_im_width // stride))

        this_size_map = np.zeros(shape=(3, new_im_hight // stride, new_im_width // stride))

        this_angle_map = np.zeros(shape=(6, new_im_hight // stride, new_im_width // stride))

        for obj_i, obj in enumerate(gt_box9d_with_cls):

            this_cls = int(obj[-1, 0])

            center = ptx_im[obj_i]

            dis = depth[obj_i]

            center[0] *= (new_im_width / raw_im_width / stride)
            center[1] *= (new_im_hight / raw_im_hight / stride)

            this_heat_map[this_cls] = draw_gaussian_to_heatmap(this_heat_map[this_cls], center, center_rad)

            this_res_map[0], this_res_map[1] = draw_res_to_heatmap(this_res_map[0], this_res_map[1], center)

            try:

                this_dis_map[0, int(center[1]), int(center[0])] = dis

                l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]

                this_size_map[0, int(center[1]), int(center[0])] = l
                this_size_map[1, int(center[1]), int(center[0])] = w
                this_size_map[2, int(center[1]), int(center[0])] = h

                this_angle_map[0, int(center[1]), int(center[0])] = np.cos(a1)
                this_angle_map[1, int(center[1]), int(center[0])] = np.sin(a1)
                this_angle_map[2, int(center[1]), int(center[0])] = np.cos(a2)
                this_angle_map[3, int(center[1]), int(center[0])] = np.sin(a2)
                this_angle_map[4, int(center[1]), int(center[0])] = np.cos(a3)
                this_angle_map[5, int(center[1]), int(center[0])] = np.sin(a3)

            except:

                continue

        gt_hm.append(this_heat_map.cpu().numpy())
        gt_center_res.append(this_res_map)
        gt_center_dis.append(this_dis_map)
        gt_dim.append(this_size_map)
        gt_rot.append(this_angle_map)

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


def center_point_decoder(hm,
                         center_res,
                         center_dis,
                         dim,
                         rot,
                         intrinsic_mat,
                         extrinsic_mat,
                         distortion_matrix,
                         new_im_width=None,
                         new_im_hight=None,
                         raw_im_width=None,
                         raw_im_hight=None,
                         stride=None,
                         im_num=None,
                         max_num=10):
    pred_boxes9d = []
    all_confidence = []

    for im_id in range(im_num):
        im_mat = intrinsic_mat[im_id]
        ex_mat = extrinsic_mat[im_id]
        dis_mat = distortion_matrix[im_id]

        this_hm = hm[im_id]  # c, W, H
        this_hm_conf, cls_map = this_hm.max(0)

        this_center_res = center_res[im_id]
        this_center_dis = center_dis[im_id]
        this_dim = dim[im_id]
        this_rot = rot[im_id]

        shape_hm = this_hm_conf.shape

        # Call NMS instead of simple topk

        confi, linear_indices = nms_pytorch(this_hm_conf, max_num, distance_threshold=2)  # (N,) (N,)

        confi = confi.detach().cpu().numpy()

        c_num = shape_hm[-1]
        rows = linear_indices // c_num  # K
        cols = linear_indices % c_num  # K

        rows_ind = rows.long()
        cols_ind = cols.long()

        res_x = this_center_res[0, rows_ind, cols_ind]
        res_y = this_center_res[1, rows_ind, cols_ind]

        cls = cls_map[rows_ind, cols_ind].cpu().numpy()
        cls = cls.reshape(-1, 1)

        depth = this_center_dis[0, rows_ind, cols_ind].detach().cpu().numpy()
        y_cor = rows + res_y
        x_cor = cols + res_x
        y_cor = y_cor.detach().cpu().numpy()
        x_cor = x_cor.detach().cpu().numpy()

        delta0 = (new_im_width / raw_im_width / stride)
        delta1 = (new_im_hight / raw_im_hight / stride)

        x_cor /= delta0
        y_cor /= delta1

        x_cor = x_cor.reshape(-1, 1)
        y_cor = y_cor.reshape(-1, 1)

        points = backProject_with_opencv_to_world(np.concatenate([x_cor, y_cor], -1), depth, ex_mat[:3, :3],
                                                  ex_mat[:3, 3], im_mat, dis_mat * 0)

        l, w, h = this_dim[0, rows_ind, cols_ind], this_dim[1, rows_ind, cols_ind], this_dim[2, rows_ind, cols_ind]

        a1 = torch.atan2(this_rot[1, rows_ind, cols_ind], this_rot[0, rows_ind, cols_ind])
        a2 = torch.atan2(this_rot[3, rows_ind, cols_ind], this_rot[2, rows_ind, cols_ind])
        a3 = torch.atan2(this_rot[5, rows_ind, cols_ind], this_rot[4, rows_ind, cols_ind])

        l = l.detach().cpu().numpy().reshape(-1, 1)
        w = w.detach().cpu().numpy().reshape(-1, 1)
        h = h.detach().cpu().numpy().reshape(-1, 1)
        a1 = a1.detach().cpu().numpy().reshape(-1, 1)
        a2 = a2.detach().cpu().numpy().reshape(-1, 1)
        a3 = a3.detach().cpu().numpy().reshape(-1, 1)

        this_box9d = np.concatenate([points, l, w, h, a1, a2, a3, cls], -1)

        pred_boxes9d.append(this_box9d)
        all_confidence.append(confi)

    return np.concatenate(pred_boxes9d), np.concatenate(all_confidence)


all_object_encoders = {'key_point_encoder': key_point_encoder,
                       'key_point_decoder': key_point_decoder,
                       'center_point_encoder': center_point_encoder,
                       'center_point_decoder': center_point_decoder}
