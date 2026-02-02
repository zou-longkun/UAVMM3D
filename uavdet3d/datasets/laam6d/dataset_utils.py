import numpy as np
import torch
import os
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from PIL import Image


def project_lidar_and_get_uvz_rgb_tag(
        lidar_points,  # (N, 5): x, y, z, intensity, tag
        lidar_extrinsic,  # 4x4
        camera_extrinsic,  # 4x4
        camera_intrinsic,  # 3x3
        image  # (H, W, 3)
):
    carla_to_opencv = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    h, w = image.shape[:2]

    xyz = lidar_points[:, :3]
    tags = lidar_points[:, 4]

    # Step 1: LiDAR → World
    lidar_homo = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    points_world = (lidar_extrinsic @ lidar_homo.T).T

    # Step 2: World → Camera
    cam_ext_inv = np.linalg.inv(camera_extrinsic)
    points_cam = (cam_ext_inv @ points_world.T).T
    points_cam_xyz = points_cam[:, :3]

    # Step 3: Camera → OpenCV
    points_cam_homo = np.hstack([points_cam_xyz, np.ones((points_cam_xyz.shape[0], 1))])
    points_opencv = (carla_to_opencv @ points_cam_homo.T).T[:, :3]

    in_front = points_opencv[:, 2] > 0
    points_opencv = points_opencv[in_front]
    points_cam_xyz = points_cam_xyz[in_front]
    tags = tags[in_front]

    uv, _ = cv2.projectPoints(points_opencv, np.eye(3), np.zeros(3), camera_intrinsic, None)
    if uv is None:
        print("[DEBUG ERROR] LIDAR UV = NONE")
        return np.array([])
    uv = uv.reshape(-1, 2)

    results = []
    for i in range(uv.shape[0]):
        u, v = uv[i]
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < w and 0 <= v_int < h:
            z = points_cam_xyz[i, 2]
            r, g, b = image[v_int, u_int].tolist()
            tag = tags[i]
            x_world, y_world, z_world = points_world[i, :3]
            results.append([u, v, z, x_world, y_world, z_world, r, g, b, tag])

    return np.array(results)  # shape: (M, 10)


def radar_to_velocity_heatmap(
        radar_data,
        radar_extrinsic,
        camera_extrinsic,
        camera_intrinsic,
        image_shape=(720, 1280),
        method='max'  # 可选：'max'、'mean'、'sum'
):
    # 解析雷达数据
    velocity = radar_data[:, 0]
    azim = radar_data[:, 1]
    alt = radar_data[:, 2]
    depth = radar_data[:, 3]

    # 极坐标 → 笛卡尔
    x = depth * np.cos(alt) * np.cos(azim)
    y = depth * np.cos(alt) * np.sin(azim)
    z = depth * np.sin(alt)
    radar_xyz = np.stack([x, y, z], axis=1)

    # 雷达 → 世界
    radar_homo = np.hstack([radar_xyz, np.ones((radar_xyz.shape[0], 1))])
    pts_world = (radar_extrinsic @ radar_homo.T).T[:, :3]

    # 世界 → 相机（再转 Carla → OpenCV）
    cam_extr_inv = np.linalg.inv(camera_extrinsic)
    world_homo = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
    pts_cam = (cam_extr_inv @ world_homo.T).T[:, :3]

    carla_to_opencv = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    pts_cam_homo = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    pts_opencv = (carla_to_opencv @ pts_cam_homo.T).T[:, :3]

    # 投影
    uv, _ = cv2.projectPoints(pts_opencv, np.zeros(3), np.zeros(3), camera_intrinsic, None)
    if uv is None:
        print("[Debug ERROR] RADAR UV = NONE")
        return np.zeros((1, image_shape[0], image_shape[1]), dtype=np.float32)
    # 关键修复：确保uv是二维数组 (N, 2)
    uv = uv.squeeze()  # 先去除多余维度
    # 如果是一维数组（单一点的情况），转为二维数组
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)
    uv = uv.astype(int)  # 转换为整数坐标

    # 构建热力图
    H, W = image_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32) if method == 'mean' else None

    for i in range(len(uv)):
        # 额外安全检查：确保每个元素都是长度为2的坐标
        if uv[i].size != 2:
            continue  # 跳过无效坐标
        u, v = uv[i]
        # 检查坐标是否在图像范围内
        if 0 <= u < W and 0 <= v < H:
            v_val = velocity[i]
            if method == 'max':
                heatmap[v, u] = max(heatmap[v, u], v_val)
            elif method == 'sum':
                heatmap[v, u] += v_val
            elif method == 'mean':
                heatmap[v, u] += v_val
                count_map[v, u] += 1.0

    if method == 'mean':
        valid = count_map > 0
        heatmap[valid] /= count_map[valid]

    # 可选平滑
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

    return heatmap[np.newaxis, ...]  # shape: (1, H, W)


def draw_box9d_on_image(boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
                        intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):
    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)

    # Carla → OpenCV 相机坐标变换
    carla_to_opencv = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    extrinsic_inv = np.linalg.inv(extrinsic_mat)

    for i, box in enumerate(boxes9d):
        box_cv = []
        for pt in box:
            if isinstance(pt, torch.Tensor):
                pt = pt.cpu().numpy()  # 仅对 PyTorch 张量进行转换
            else:
                pt = np.array(pt)  # 处理其他类型（如列表）
            pt_hom = np.append(pt, 1.0)  # 世界坐标 → 齐次
            pt_cam = extrinsic_inv @ pt_hom  # 世界 → 相机
            pt_cv = carla_to_opencv @ pt_cam  # Carla 相机 → OpenCV 相机
            box_cv.append(pt_cv[:3])

        box_cv = np.array(box_cv, dtype=np.float32)  # (9, 3)

        corners_2d, _ = cv2.projectPoints(box_cv[1:], np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        middle_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
        top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]

        for edge in bottom_edges:
            color = (255, 0, 0)
            idx1, idx2 = edge
            pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
            pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
            cv2.line(image, pt1, pt2, color, 1)
        for idx, edge in enumerate(middle_edges):
            intensity = int(255 - (idx / 3) * 150)
            color = (0, intensity, 0)
            idx1, idx2 = edge
            pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
            pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
            cv2.line(image, pt1, pt2, color, 1)
        for edge in top_edges:
            color = (0, 0, 255)
            idx1, idx2 = edge
            pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
            pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
            cv2.line(image, pt1, pt2, color, 1)

        diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]
        mid_points = []
        for i, j in diagonals:
            pt1, pt2 = box_cv[1:][i], box_cv[1:][j]
            mid = (pt1 + pt2) / 2
            mid_points.append(mid)
            proj_pts, _ = cv2.projectPoints(np.vstack([pt1, pt2]), np.eye(3), np.zeros(3), intrinsic_mat,
                                            distortion_matrix)
            p1, p2 = proj_pts.reshape(-1, 2).astype(int)
            cv2.line(image, tuple(p1), tuple(p2), (0, 255, 255), 1)

    return image


# def draw_box9d_on_image_gt(boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
#                            intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):
#     if intrinsic_mat is None:
#         intrinsic_mat = np.array([[img_width, 0, img_width / 2],
#                                   [0, img_height, img_height / 2],
#                                   [0, 0, 1]], dtype=np.float32)

#     if extrinsic_mat is None:
#         extrinsic_mat = np.eye(4)

#     if distortion_matrix is None:
#         distortion_matrix = np.zeros(5, dtype=np.float32)

#     # Carla → OpenCV 相机坐标变换
#     carla_to_opencv = np.array([
#         [0, 1, 0, 0],
#         [0, 0, -1, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 1]
#     ])
#     extrinsic_inv = np.linalg.inv(extrinsic_mat)

#     for i, box in enumerate(boxes9d):
#         box_cv = []
#         for pt in box:
#             if isinstance(pt, torch.Tensor):
#                 pt = pt.cpu().numpy()  # 仅对 PyTorch 张量进行转换
#             else:
#                 pt = np.array(pt)  # 处理其他类型（如列表）
#             pt_hom = np.append(pt, 1.0)  # 世界坐标 → 齐次
#             pt_cam = extrinsic_inv @ pt_hom  # 世界 → 相机
#             pt_cv = carla_to_opencv @ pt_cam  # Carla 相机 → OpenCV 相机
#             box_cv.append(pt_cv[:3])

#         box_cv = np.array(box_cv, dtype=np.float32)  # (9, 3)
        
#         # 计算框的深度信息（使用OpenCV相机坐标系中的z值，即box_cv[0, 2]，这是框的中心点）
#         # 在OpenCV坐标系中，z值表示沿光轴向前的距离，正值为物体在相机前方
#         box_depth = box_cv[0, 2] if len(box_cv) > 0 else 0.0

#         corners_2d, _ = cv2.projectPoints(box_cv[1:], np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
#         corners_2d = corners_2d.reshape(-1, 2).astype(int)

#         bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
#         middle_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
#         top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]

#         for edge in bottom_edges:
#             color = (255, 255, 255)
#             idx1, idx2 = edge
#             pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
#             pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
#             try:
#                 cv2.line(image, pt1, pt2, color, 1)
#             except Exception as e:
#                 # 报错时打印相关信息
#                 print(f"Line drawing failed: {e}")
#                 print(f"pt1 value: {pt1}, type: {type(pt1)}")
#                 print(f"pt2 value: {pt2}, type: {type(pt2)}")

#         for idx, edge in enumerate(middle_edges):
#             intensity = int(255 - (idx / 3) * 150)
#             color = (255, 255, 255)
#             idx1, idx2 = edge
#             pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
#             pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
#             cv2.line(image, pt1, pt2, color, 1)
#         for edge in top_edges:
#             color = (255, 255, 255)
#             idx1, idx2 = edge
#             pt1 = (int(corners_2d[idx1, 0]), int(corners_2d[idx1, 1]))
#             pt2 = (int(corners_2d[idx2, 0]), int(corners_2d[idx2, 1]))
#             cv2.line(image, pt1, pt2, color, 1)

#         diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]
#         mid_points = []
#         for i, j in diagonals:
#             pt1, pt2 = box_cv[1:][i], box_cv[1:][j]
#             mid = (pt1 + pt2) / 2
#             mid_points.append(mid)
#             proj_pts, _ = cv2.projectPoints(np.vstack([pt1, pt2]), np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
#             p1, p2 = proj_pts.reshape(-1, 2).astype(int)
#             cv2.line(image, tuple(p1), tuple(p2), (255, 255, 255), 1)
        
#         # 在框的上方显示深度信息
#         if len(corners_2d) > 0:
#             # 计算框的中心点上方位置，用于显示深度信息
#             # 找到边界框的最小和最大y坐标，以确定框的顶部
#             y_coords = corners_2d[:, 1]
#             min_y = np.min(y_coords)
#             # 找到y坐标最小的点的x坐标的平均值
#             top_points = corners_2d[y_coords == min_y]
#             avg_x = np.mean(top_points[:, 0])
            
#             # 文本位置在框的上方，稍微偏移一点
#             text_pos = (int(avg_x), max(0, min_y - 10))
            
#             # 格式化深度信息文本，保留2位小数
#             depth_text = f"Depth: {box_depth:.2f}m"
            
#             # 确保文本在图像范围内
#             text_size, _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#             text_pos = (max(0, min(text_pos[0] - text_size[0] // 2, image.shape[1] - text_size[0])), max(text_size[1], text_pos[1]))
            
#             # 绘制文本背景以提高可读性
#             cv2.rectangle(image, 
#                          (text_pos[0] - 5, text_pos[1] - text_size[1] - 5), 
#                          (text_pos[0] + text_size[0] + 5, text_pos[1] + 5), 
#                          (0, 0, 0), -1)
            
#             # 绘制深度文本
#             cv2.putText(image, depth_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     return image

def draw_box9d_on_image_gt(boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
                           intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):
    # 初始化默认参数
    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)

    # Carla → OpenCV 相机坐标变换矩阵
    carla_to_opencv = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    extrinsic_inv = np.linalg.inv(extrinsic_mat)

    # 获取图像实际尺寸
    if image is not None and len(image.shape) >= 2:
        actual_h, actual_w = image.shape[:2]
    else:
        actual_h, actual_w = int(img_height), int(img_width)

    for box in boxes9d:
        box_cv = []
        # 转换3D点坐标（修复笔误：将pt_cv.append改为box_cv.append）
        for pt in box:
            if isinstance(pt, torch.Tensor):
                pt = pt.cpu().numpy()
            else:
                pt = np.array(pt, dtype=np.float32)
            
            pt_hom = np.append(pt, 1.0)
            pt_cam = extrinsic_inv @ pt_hom
            # 核心修复：append到box_cv（原笔误写为pt_cv.append）
            box_cv.append((carla_to_opencv @ pt_cam)[:3])

        box_cv = np.array(box_cv, dtype=np.float32)
        
        # 跳过无效3D框（数量不足或全为无效值）
        if len(box_cv) <= 1 or not np.isfinite(box_cv).any():
            continue

        # 3D点投影到2D图像平面
        corners_2d, _ = cv2.projectPoints(box_cv[1:], np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2)

        # 过滤全NaN的2D坐标，直接跳过当前框
        if not np.isfinite(corners_2d).any():
            continue

        # 定义边界框各类边缘连接
        bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        middle_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
        top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
        diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]

        # 通用线条绘制函数（内部静默处理无效情况）
        def draw_edge(edges):
            for edge in edges:
                idx1, idx2 = edge
                # 跳过索引越界
                if idx1 >= len(corners_2d) or idx2 >= len(corners_2d):
                    continue
                
                x1, y1 = corners_2d[idx1]
                x2, y2 = corners_2d[idx2]
                
                # 仅处理有效坐标
                if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
                    # 转换为整数并裁剪到图像范围
                    pt1 = (int(round(x1)), int(round(y1)))
                    pt2 = (int(round(x2)), int(round(y2)))
                    pt1 = (max(0, min(actual_w - 1, pt1[0])), max(0, min(actual_h - 1, pt1[1])))
                    pt2 = (max(0, min(actual_w - 1, pt2[0])), max(0, min(actual_h - 1, pt2[1])))
                    # 静默绘制（忽略异常）
                    try:
                        cv2.line(image, pt1, pt2, (255, 255, 255), 1)
                    except:
                        pass

        # 绘制底部、中间、顶部边缘
        draw_edge(bottom_edges)
        draw_edge(middle_edges)
        draw_edge(top_edges)

        # 绘制对角线
        for i, j in diagonals:
            # 跳过索引越界
            if i >= len(box_cv[1:]) or j >= len(box_cv[1:]):
                continue
            
            pt1_3d, pt2_3d = box_cv[1:][i], box_cv[1:][j]
            # 3D点投影到2D
            proj_pts, _ = cv2.projectPoints(np.vstack([pt1_3d, pt2_3d]), np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
            p1, p2 = proj_pts.reshape(-1, 2)
            
            # 仅处理有效坐标
            if np.isfinite(p1).all() and np.isfinite(p2).all():
                # 转换为整数并裁剪到图像范围
                p1_int = (int(round(p1[0])), int(round(p1[1])))
                p2_int = (int(round(p2[0])), int(round(p2[1])))
                p1_int = (max(0, min(actual_w - 1, p1_int[0])), max(0, min(actual_h - 1, p1_int[1])))
                p2_int = (max(0, min(actual_w - 1, p2_int[0])), max(0, min(actual_h - 1, p2_int[1])))
                # 静默绘制（忽略异常）
                try:
                    cv2.line(image, p1_int, p2_int, (255, 255, 255), 1)
                except:
                    pass

        # 绘制深度文本（仅在有有效坐标时）
        valid_coords = corners_2d[np.isfinite(corners_2d).all(axis=1)]
        if len(valid_coords) > 0:
            # 获取有效深度值
            box_depth = box_cv[0, 2] if np.isfinite(box_cv[0, 2]) else 0.0
            # 计算文本显示位置
            y_coords = valid_coords[:, 1]
            min_y = np.nanmin(y_coords)
            top_points = valid_coords[y_coords == min_y]
            avg_x = np.nanmean(top_points[:, 0]) if len(top_points) > 0 else 0
            
            # 调整文本位置（确保在图像内）
            text_x = int(round(avg_x))
            text_y = max(20, int(round(min_y)) - 10)
            text_x = max(0, min(text_x - 30, actual_w - 60))  # 固定文本宽度偏移
            
            # 格式化深度文本
            depth_text = f"Depth: {box_depth:.2f}m"
            # 静默绘制文本（忽略异常）
            try:
                # 绘制黑色背景（提高可读性）
                cv2.rectangle(image, (text_x - 5, text_y - 15), (text_x + 65, text_y + 5), (0, 0, 0), -1)
                # 绘制白色文本
                cv2.putText(image, depth_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except:
                pass

    return image

def convert_9params_to_9points(box9d_params):
    """
    预测阶段可直接使用的版本：不依赖 local_prototypes。
    输入: (N, 9) -> [x,y,z,l,w,h,a1,a2,a3]
    输出: (N, 9, 3) -> [center, eight corners] in world frame
    角点顺序采用固定的立方体原型，保证与投影/可视化一致。
    """
    box9d_params = np.asarray(box9d_params, dtype=np.float32)
    if box9d_params.size == 0:
        return np.empty((0, 9, 3), dtype=np.float32)

    out = []
    for p in box9d_params:
        x, y, z, l, w, h, a1, a2, a3 = p.astype(np.float32)

        # 1) 局部坐标系下的固定角点（以中心为原点）
        # 底面: 0-3, 顶面: 4-7
        corners_obj = np.array([
            [-l / 2, -w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [l / 2, w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [l / 2, w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
        ], dtype=np.float32)

        # 2) 旋转矩阵（zyx）
        Rm = R.from_euler('zyx', [a1, a2, a3], degrees=False).as_matrix().astype(np.float32)
        # 保持右手系（健壮性处理）
        if np.linalg.det(Rm) < 0:
            Rm[:, 2] *= -1

        # 3) 旋转 + 平移到世界坐标
        corners_world = (Rm @ corners_obj.T).T + np.array([x, y, z], dtype=np.float32)

        # 4) 拼 9 点（中心点 + 8 角点）
        out.append(np.vstack((np.array([x, y, z], dtype=np.float32), corners_world)))

    return np.stack(out, axis=0).astype(np.float32)


def generate_world_coords_map(lidar_proj_info, image_shape):
    h, w = image_shape
    # x_world, y_world, z_world
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    z_map = np.zeros((h, w), dtype=np.float32)

    for data in lidar_proj_info:
        u, v, z_carla, x_world, y_world, z_world, r, g, b, tag = data
        u_int, v_int = int(round(u)), int(round(v))

        if 0 <= u_int < w and 0 <= v_int < h:
            if z_map[v_int, u_int] == 0 or z_carla < z_map[v_int, u_int]:
                x_map[v_int, u_int] = x_world
                y_map[v_int, u_int] = y_world
                z_map[v_int, u_int] = z_world

    world_coords_map = np.stack([x_map, y_map, z_map], axis=-1)
    return world_coords_map  # (H, W, 3)


def convert_9points_to_9params(box9d_points):
    all_box_params = []

    for single_box in box9d_points:
        center = single_box[0].copy()
        x, y, z = center
        corners_local = single_box[1:] - center  # (8, 3)

        # 1. 计算旋转矩阵（物体自身坐标系）
        x_axis = corners_local[0] / np.linalg.norm(corners_local[0]) if np.linalg.norm(
            corners_local[0]) != 0 else np.array([1, 0, 0])
        y_axis = corners_local[1] / np.linalg.norm(corners_local[1]) if np.linalg.norm(
            corners_local[1]) != 0 else np.array([0, 1, 0])
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) != 0 else np.array([0, 0, 1])
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] *= -1

        # 2. 将局部角点投影到物体自身坐标系（获取沿x/y/z轴的坐标）
        corners_self = np.dot(corners_local, rotation_matrix)  # (8, 3)，每个点在物体坐标系下的坐标

        # 3. 计算物体自身坐标系下的尺寸（l: x轴长度, w: y轴长度, h: z轴长度）
        l = np.max(corners_self[:, 0]) - np.min(corners_self[:, 0])
        w = np.max(corners_self[:, 1]) - np.min(corners_self[:, 1])
        h = np.max(corners_self[:, 2]) - np.min(corners_self[:, 2])

        # 4. 计算欧拉角
        r = R.from_matrix(rotation_matrix)
        angles = r.as_euler('zyx')
        angle1, angle2, angle3 = angles

        box_params = np.array([x, y, z, l, w, h, angle1, angle2, angle3])
        all_box_params.append(box_params)

    return np.array(all_box_params)


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
            print(f"[Warning] Abnormal bounding box size: l={l}, w={w}, h={h}, skipping this box")
            continue

        x_axis = corners[1] - corners[0]
        y_axis = corners[3] - corners[0]
        z_axis = corners[4] - corners[0]

        x_norm = np.linalg.norm(x_axis)
        y_norm = np.linalg.norm(y_axis)
        z_norm = np.linalg.norm(z_axis)

        if x_norm < 1e-6 or y_norm < 1e-6 or z_norm < 1e-6:
            print(f"[Warning] Axis vector norm is 0, skipping this box")
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
            print(f"[Warning] Invalid rotation matrix, using default Euler angles")
            euler = np.zeros(3)

        box_param = np.concatenate([center, [l, w, h], euler])
        all_box_params.append(box_param)

    return np.array(all_box_params) if all_box_params else np.empty((0, 9))


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
    Calculate the angular error of Euler angles in radians.

    Args:
    pred (numpy.ndarray): Predicted Euler angles, shape (N, 3).
    gt (numpy.ndarray): Ground truth Euler angles, shape (N, 3).

    Returns:
    numpy.ndarray: Angular error for each sample, shape (N,).
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
        print(f"Distance calculation failed: {e}")
        return np.full(original_x_len, -1, dtype=int)

    cost[~np.isfinite(cost)] = unmatched_cost

    size = max(N, M)
    padded_cost = np.full((size, size), unmatched_cost, dtype=np.float64)
    padded_cost[:N, :M] = cost

    try:
        row_ind, col_ind = linear_sum_assignment(padded_cost)
    except Exception as e:
        print(f"Hungarian algorithm failed: {e}")
        return np.full(original_x_len, -1, dtype=int)

    x_to_y = np.full(original_x_len, -1, dtype=int)
    original_x_indices = np.where(valid_x)[0]
    original_y_indices = np.where(valid_y)[0]

    for r, c in zip(row_ind, col_ind):
        if r < N and c < M and padded_cost[r, c] < unmatched_cost:
            x_to_y[original_x_indices[r]] = original_y_indices[c]

    return x_to_y


def match_gt(gt_box9d, pred_boxes9d):
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
    # error = 2 * np.arccos(d) * 180 / np.pi
    # d     = abs(np.dot(groundtruth, predicted))
    # d     = min(1.0, max(-1.0, d))

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


def val_rotation_euler(pred_euler, gt_euler):  # Function name corrected, input is explicitly Euler angles
    """
    Input:
        pred_euler: Predicted Euler angles (3,), e.g., [a1,a2,a3]
        gt_euler: Ground truth Euler angles (3,)
    Output:
        Rotation error (degrees)
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
        print(f"Rotation error calculation failed: {e}, input Euler angles: pred={pred_euler}, gt={gt_euler}")
        return np.nan  # Return nan on error, filter later


def visualize_multi_modal_align_old(
        rgb_image,
        ir_image,
        dvs_image,
        velocity_image,
        world_coords_map,
        world_coords_map_original,
        save_path,
        sample_coords=None):
    """
    可视化多模态图像（RGB、IR、DVS、Velocity）与世界坐标映射图的对齐效果
    Args:
        rgb_image: RGB图像 (H, W, 3)
        ir_image: 红外图像 (H, W) 或 (H, W, 3)
        dvs_image: DVS事件图像 (H, W, 3)
        velocity_image: 雷达速度热力图 (H, W) 或 (H, W, 3)
        world_coords_map: 世界坐标映射图 (H, W, 3)
        save_path: 保存路径
        sample_coords: 待验证的采样坐标列表 [(u1, v1), (u2, v2), ...]，默认随机生成3个
    """
    # 确保所有图像尺寸一致
    h, w = rgb_image.shape[:2]
    assert ir_image.shape[:2] == (h, w), f"IR size mismatch: {ir_image.shape[:2]} vs {(h, w)}"
    assert dvs_image.shape[:2] == (h, w), f"DVS size mismatch: {dvs_image.shape[:2]} vs {(h, w)}"
    assert velocity_image.shape[:2] == (h, w), f"Velocity size mismatch: {velocity_image.shape[:2]} vs {(h, w)}"
    assert world_coords_map.shape[:2] == (h, w), f"World coordinate map size mismatch: {world_coords_map.shape[:2]} vs {(h, w)}"

    # 预处理单通道图像为3通道（便于统一显示）
    if len(ir_image.shape) == 2:
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
    if len(velocity_image.shape) == 2:
        velocity_image = cv2.applyColorMap(velocity_image, cv2.COLORMAP_JET)  # 热力图上色

    # 提取世界坐标X通道用于可视化
    x_coords = world_coords_map[..., 0]
    valid_mask = x_coords != 0
    x_normalized = np.zeros_like(x_coords, dtype=np.uint8)
    if np.any(valid_mask):
        x_valid = x_coords[valid_mask]
        x_normalized[valid_mask] = ((x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-8)) * 255
    world_vis = cv2.applyColorMap(x_normalized, cv2.COLORMAP_VIRIDIS)  # 世界坐标可视化

    x_coords = world_coords_map_original[..., 0]
    valid_mask = x_coords != 0
    x_normalized = np.zeros_like(x_coords, dtype=np.uint8)
    if np.any(valid_mask):
        x_valid = x_coords[valid_mask]
        x_normalized[valid_mask] = ((x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-8)) * 255
    world_vis2 = cv2.applyColorMap(x_normalized, cv2.COLORMAP_VIRIDIS)  # 世界坐标可视化

    # 生成采样坐标（默认随机选3个有效点）
    if sample_coords is None:
        sample_coords = []
        if np.any(valid_mask):
            # 从有效坐标中随机选3个
            valid_uv = np.argwhere(valid_mask)  # (v, u) 格式
            if len(valid_uv) >= 3:
                sample_indices = np.random.choice(len(valid_uv), 3, replace=False)
                sample_coords = [(valid_uv[i][1], valid_uv[i][0]) for i in sample_indices]  # 转为 (u, v)
            else:
                # 若无足够有效点，选图像中心附近
                sample_coords = [(w // 2, h // 2), (w // 3, h // 3), (2 * w // 3, 2 * h // 3)]
        else:
            sample_coords = [(w // 2, h // 2), (w // 3, h // 3), (2 * w // 3, 2 * h // 3)]

    # 从查询结果中选一个Noto Sans CJK字体文件（比如常规体）
    noto_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    # Verify if the path exists (prevent spelling errors)
    if not os.path.exists(noto_font_path):
        print(f"Warning: Font file does not exist, please check the path! Current path: {noto_font_path}")
    else:
        # Load font and apply globally
        noto_font = FontProperties(fname=noto_font_path)
        plt.rcParams["font.family"] = noto_font.get_name()  # Use the actual name of the font file
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

    # 创建画布（2行3列布局）
    plt.figure(figsize=(18, 12))
    modalities = [
        ("RGB", cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)),
        ("IR", cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB)),
        ("DVS", cv2.cvtColor(dvs_image, cv2.COLOR_BGR2RGB)),
        ("Radar HM", cv2.cvtColor(velocity_image, cv2.COLOR_BGR2RGB)),
        ("World Coordinate X Channel", cv2.cvtColor(world_vis, cv2.COLOR_BGR2RGB)),
        ("World Coordinate AA Channel", cv2.cvtColor(world_vis2, cv2.COLOR_BGR2RGB)),
    ]

    # 绘制每个模态并标注采样坐标
    for i, (title, img) in enumerate(modalities, 1):
        plt.subplot(2, 3, i)
        plt.title(title, fontsize=16)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-modal alignment visualization saved to: {save_path}")


def visualize_multi_modal_align_new(
        rgb_image,
        ir_image,
        dvs_image,
        velocity_image,
        world_coords_map,
        world_coords_map_original,
        save_path,
        sample_coords=None):
    """
    Visualize multi-modal images, force specify Noto font through font file path to solve Chinese display issue
    """
    # Ensure all images have the same dimensions
    h, w = rgb_image.shape[:2]
    assert ir_image.shape[:2] == (h, w), f"IR size mismatch: {ir_image.shape[:2]} vs {(h, w)}"
    assert dvs_image.shape[:2] == (h, w), f"DVS size mismatch: {dvs_image.shape[:2]} vs {(h, w)}"
    assert velocity_image.shape[:2] == (h, w), f"Velocity size mismatch: {velocity_image.shape[:2]} vs {(h, w)}"
    assert world_coords_map.shape[:2] == (h, w), f"World coordinate map size mismatch: {world_coords_map.shape[:2]} vs {(h, w)}"

    # Preprocess single-channel images to 3-channel
    if len(ir_image.shape) == 2:
        ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
    if len(velocity_image.shape) == 2:
        velocity_image = cv2.applyColorMap(velocity_image, cv2.COLORMAP_JET)

    # Extract X channel from world coordinates for visualization
    x_coords = world_coords_map[..., 0]
    valid_mask = x_coords != 0
    x_normalized = np.zeros_like(x_coords, dtype=np.uint8)
    if np.any(valid_mask):
        x_valid = x_coords[valid_mask]
        x_normalized[valid_mask] = ((x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-8)) * 255
    world_vis = cv2.applyColorMap(x_normalized, cv2.COLORMAP_VIRIDIS)

    x_coords = world_coords_map_original[..., 0]
    valid_mask = x_coords != 0
    x_normalized = np.zeros_like(x_coords, dtype=np.uint8)
    if np.any(valid_mask):
        x_valid = x_coords[valid_mask]
        x_normalized[valid_mask] = ((x_valid - x_valid.min()) / (x_valid.max() - x_valid.min() + 1e-8)) * 255
    world_vis2 = cv2.applyColorMap(x_normalized, cv2.COLORMAP_VIRIDIS)

    # -------------------------- Key modification: Specify font through file path --------------------------
    # 1. Clear font cache (ensure Matplotlib reloads fonts)
    cache_dir = matplotlib.get_cachedir()
    for f in os.listdir(cache_dir):
        if f.startswith('fontlist'):
            os.remove(os.path.join(cache_dir, f))

    # 2. Manually specify Noto font file path (use a file that exists in your system)
    # Select a Noto Sans CJK font file from the query results (e.g., regular)
    noto_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    # 验证路径是否存在（防止拼写错误）
    if not os.path.exists(noto_font_path):
        print(f"Warning: Font file does not exist, please check the path! Current path: {noto_font_path}")
    else:
        # 加载字体并应用到全局
        noto_font = FontProperties(fname=noto_font_path)
        plt.rcParams["font.family"] = noto_font.get_name()  # 用字体文件的实际名称
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号问题

    # Core layout parameters
    plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(
        2, 6,
        figure=plt.gcf(),
        hspace=0.02,
        height_ratios=[5, 3]
    )

    # Five modalities (with titles)
    modalities = [
        ("RGB", cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)),
        ("IR", cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB)),
        ("DVS", cv2.cvtColor(dvs_image, cv2.COLOR_BGR2RGB)),
        ("Radar HM", cv2.cvtColor(velocity_image, cv2.COLOR_BGR2RGB)),
        ("Lidar", cv2.cvtColor(world_vis, cv2.COLOR_BGR2RGB)),
    ]

    # 第一行大图
    ax1 = plt.subplot(gs[0, 0:3])
    ax1.set_title(modalities[0][0], fontsize=16, pad=5, fontproperties=noto_font)  # 显式指定字体
    ax1.imshow(modalities[0][1])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.subplot(gs[0, 3:6])
    ax2.set_title(modalities[1][0], fontsize=16, pad=5, fontproperties=noto_font)
    ax2.imshow(modalities[1][1])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # 第二行小图
    ax3 = plt.subplot(gs[1, 0:2])
    ax3.set_title(modalities[2][0], fontsize=14, pad=3, fontproperties=noto_font)
    ax3.imshow(modalities[2][1])
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax4 = plt.subplot(gs[1, 2:4])
    ax4.set_title(modalities[3][0], fontsize=14, pad=3, fontproperties=noto_font)
    ax4.imshow(modalities[3][1])
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = plt.subplot(gs[1, 4:6])
    ax5.set_title(modalities[4][0], fontsize=14, pad=3, fontproperties=noto_font)  # Title with specified font
    ax5.imshow(modalities[4][1])
    ax5.set_xticks([])
    ax5.set_yticks([])

    plt.savefig(
        save_path,
        dpi=350,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()
    print(f"Multi-modal alignment visualization saved to: {save_path}")


def decode_box_centers_from_world(centers_world, extrinsic_inv):
    carla_to_opencv = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    centers_world = np.asarray(centers_world).reshape(-1, 3)
    centers_hom = np.hstack([centers_world, np.ones((centers_world.shape[0], 1))])  # (N, 4)
    centers_cam = (extrinsic_inv @ centers_hom.T).T  # (N, 4)
    centers_cv = (carla_to_opencv @ centers_cam.T).T  # (N, 4)
    return centers_cv[:, :3].astype(np.float32)


def xyz_to_uv(xyz_coords, img_width=1920., img_height=1080.,
              intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None,
              return_average=True):
    """
    Convert XYZ world coordinates to UV pixel coordinates on the image, and optionally calculate average offset.

    Args:
        xyz_coords: 3D coordinates, can be a single coordinate (list/array) or a list of multiple coordinates (N, 3)
        img_width: Image width
        img_height: Image height
        intrinsic_mat: Camera intrinsic matrix
        extrinsic_mat: Camera extrinsic matrix
        distortion_matrix: Distortion coefficient matrix
        return_average: Whether to return the average offset of all points

    Returns:
        If return_average=True: Average UV coordinates (1, 2)
        Otherwise: UV coordinates of all points (N, 2)
    """
    # 设置默认参数
    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)

    # 坐标变换矩阵 (Carla → OpenCV)
    carla_to_opencv = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    # 计算外参矩阵的逆矩阵
    extrinsic_inv = np.linalg.inv(extrinsic_mat)

    # 处理输入坐标，确保是二维数组形式 (N, 3)
    coords = np.asarray(xyz_coords)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    # 转换坐标
    uv_coords = []
    for pt in coords:
        # 转换为齐次坐标 (x, y, z, 1)
        pt_hom = np.append(pt, 1.0)

        # 世界坐标 → 相机坐标
        pt_cam = extrinsic_inv @ pt_hom

        # Carla相机坐标 → OpenCV相机坐标
        pt_cv = carla_to_opencv @ pt_cam

        # 投影到图像平面获取UV坐标
        proj_pt, _ = cv2.projectPoints(pt_cv[:3], np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
        uv = proj_pt.reshape(2).astype(int)
        uv_coords.append(uv)

    uv_array = np.array(uv_coords)

    # 如果需要，计算并返回平均值
    if return_average and len(uv_array) > 0:
        return np.mean(uv_array, axis=0, keepdims=True).astype(int)
    else:
        return uv_array


def register_images_by_center(rgb_img, ir_img, dvs_img,
                              center_rgb, center_ir, center_dvs,
                              intrinsic_rgb, intrinsic_ir, intrinsic_dvs):
    """
    保持RGB图像不变，通过平移IR和DVS图像使其中心点与RGB对齐
    平移方式：左侧超出部分截断，右侧不足部分用黑色填充

    参数:
        rgb_img: RGB图像（保持不变）
        ir_img: IR图像（需要平移）
        dvs_img: DVS图像（需要平移）
        center_rgb: RGB中心点坐标 (u, v)
        center_ir: IR中心点坐标 (u, v)
        center_dvs: DVS中心点坐标 (u, v)
        内参矩阵: 3x3矩阵
    """
    # 提取中心点坐标 (u, v)
    center_rgb = (int(center_rgb[0][0]), int(center_rgb[0][1]))
    center_ir = (int(center_ir[0][0]), int(center_ir[0][1]))
    center_dvs = (int(center_dvs[0][0]), int(center_dvs[0][1]))

    # print("\n===== 初始中心点信息 =====")
    # print(f"RGB中心点: {center_rgb}")
    # print(f"IR中心点: {center_ir}")
    # print(f"DVS中心点: {center_dvs}")

    # 获取图像尺寸（假设所有图像尺寸相同）
    h, w = rgb_img.shape[:2]
    # print(f"图像尺寸: {w}x{h}")

    # 计算需要平移的像素数（以RGB为基准）
    # 正值：需要向右平移；负值：需要向左平移
    ir_shift_u = center_rgb[0] - center_ir[0]
    ir_shift_v = center_rgb[1] - center_ir[1]

    dvs_shift_u = center_rgb[0] - center_dvs[0]
    dvs_shift_v = center_rgb[1] - center_dvs[1]

    # print(f"\n===== 平移量计算 =====")
    # print(f"IR需要平移: 水平{ir_shift_u}px, 垂直{ir_shift_v}px")
    # print(f"DVS需要平移: 水平{dvs_shift_u}px, 垂直{dvs_shift_v}px")

    # ------------------------------
    # 平移IR图像
    # ------------------------------
    # 创建平移矩阵 [1,0,dx; 0,1,dy]
    ir_M = np.float32([[1, 0, ir_shift_u], [0, 1, ir_shift_v]])
    # 执行平移：左侧超出部分截断，右侧不足部分用黑色填充
    aligned_ir = cv2.warpAffine(
        ir_img,
        ir_M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,  # 超出部分用常数填充
        borderValue=0  # 填充黑色
    )

    # ------------------------------
    # 平移DVS图像
    # ------------------------------
    dvs_M = np.float32([[1, 0, dvs_shift_u], [0, 1, dvs_shift_v]])
    aligned_dvs = cv2.warpAffine(
        dvs_img,
        dvs_M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # ------------------------------
    # 计算平移后的中心点（应该与RGB一致）
    # ------------------------------
    ir_new_center = (center_ir[0] + ir_shift_u, center_ir[1] + ir_shift_v)
    dvs_new_center = (center_dvs[0] + dvs_shift_u, center_dvs[1] + dvs_shift_v)

    # ------------------------------
    # 更新内参（主点坐标同步更新）
    # ------------------------------
    new_intrinsic_rgb = intrinsic_rgb.copy()  # RGB内参不变
    new_intrinsic_ir = intrinsic_ir.copy()
    new_intrinsic_dvs = intrinsic_dvs.copy()

    # 内参主点坐标随平移量调整
    new_intrinsic_ir[0, 2] += ir_shift_u  # cx
    new_intrinsic_ir[1, 2] += ir_shift_v  # cy

    new_intrinsic_dvs[0, 2] += dvs_shift_u
    new_intrinsic_dvs[1, 2] += dvs_shift_v

    # ------------------------------
    # 输出结果验证
    # ------------------------------
    # print(f"\n===== 对齐结果 =====")
    # print(f"RGB中心点: {center_rgb}")
    # print(f"IR平移后中心点: {ir_new_center}")
    # print(f"DVS平移后中心点: {dvs_new_center}")

    return {
        'rgb': rgb_img,  # RGB保持不变
        'ir': aligned_ir,
        'dvs': aligned_dvs,
        'intrinsics': {
            'rgb': new_intrinsic_rgb,
            'ir': new_intrinsic_ir,
            'dvs': new_intrinsic_dvs
        },
        'shift_info': {
            'ir': (ir_shift_u, ir_shift_v),
            'dvs': (dvs_shift_u, dvs_shift_v)
        }
    }


def eval_box6d_error(annos, max_dis):
    all_orin_error = []
    all_pos_error = []

    for i, each_anno in enumerate(annos):
        gt_box9d = each_anno['gt_boxes']
        pred_boxes9d = each_anno['pred_boxes']

        if isinstance(gt_box9d, torch.Tensor):
            gt_box9d = gt_box9d.cpu().numpy()
        if isinstance(pred_boxes9d, torch.Tensor):
            pred_boxes9d = pred_boxes9d.cpu().numpy()

        if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
            continue

        gt_box9d, pred_boxes9d = match_gt(gt_box9d, pred_boxes9d)

        if len(gt_box9d) == 0 or len(pred_boxes9d) == 0:
            continue

        pred_xyz = pred_boxes9d[:, 0, :]
        gt_xyz = gt_box9d[:, 0, :]

        # (center, l, w, h, a1, a2, a3)
        pred_params = convert_box9d_to_box_param(pred_boxes9d)  # (N, 9)
        gt_params = convert_box9d_to_box_param(gt_box9d)  # (N, 9)

        pred_angle = pred_params[:, 6:9]  # (N, 3)
        gt_angle = gt_params[:, 6:9]  # (N, 3)

        dis_error = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)
        dis_error[dis_error > max_dis] = max_dis

        # angle_error = val_rotation_euler(pred_angle, gt_angle)
        angle_error = np.array([val_rotation_euler(pred_angle[i], gt_angle[i])
                                for i in range(len(pred_angle))])

        dis_error = dis_error[~np.isnan(dis_error) & ~np.isinf(dis_error)]
        angle_error = angle_error[~np.isnan(angle_error) & ~np.isinf(angle_error)]

        if len(dis_error) > 0 and len(angle_error) > 0:
            all_orin_error.append(angle_error)
            all_pos_error.append(dis_error)

    if len(all_orin_error) > 0 and len(all_pos_error) > 0:
        all_orin_error = np.concatenate(all_orin_error)
        all_pos_error = np.concatenate(all_pos_error)
    else:
        all_orin_error = np.array([0.0])
        all_pos_error = np.array([0.0])

    return all_orin_error, all_pos_error


def eval_6d_pose(annos, max_dis):

    orin_error, pos_error = eval_box6d_error(annos, max_dis)

    return orin_error, pos_error


def evaluation_by_name(annos, metric_root_path, cls_name, max_ass_dis):

    orin_error, pos_error = eval_6d_pose(annos, max_dis=max_ass_dis)

    plt.figure(figsize=(10, 4))
    plt.scatter(np.arange(0, len(pos_error)), pos_error, s=5)
    plt.title(f"{cls_name} Position Error")
    plt.xlabel("Instance Index")
    plt.ylabel("Position Error (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(metric_root_path, f'{cls_name}_pos_error.png'))
    plt.close()

    print(f"{cls_name}_orin_error： {orin_error.mean():.4f}")
    print(f"{cls_name}_pos_error: {pos_error.mean():.4f}")
    print(f"{cls_name}_orin_error_median: {np.median(orin_error):.4f}")
    print(f"{cls_name}_pos_error_median: {np.median(pos_error):.4f}")
    print(f"{cls_name}_orin_error_min： {orin_error.min():.4f}")
    print(f"{cls_name}_pos_error_min: {pos_error.min():.4f}")
    print(f"{cls_name}_orin_error_max： {orin_error.max():.4f}")
    print(f"{cls_name}_pos_error_max: {pos_error.max():.4f}")

    return_str = f'''
        {cls_name}: 
        orin_error： {orin_error.mean():.4f}
        pos_error: {pos_error.mean():.4f}
        orin_error_median: {np.median(orin_error):.4f}
        pos_error_median: {np.median(pos_error):.4f}
        orin_error_min: {orin_error.min():.4f}
        pos_error_min: {pos_error.min():.4f}
        orin_error_max: {orin_error.max():.4f}
        pos_error_max: {pos_error.max():.4f}
    '''

    pd.DataFrame({f'{cls_name}_orin_error': orin_error.tolist()}).to_csv(
        os.path.join(metric_root_path, f'{cls_name}_orin_error.csv'), index=False)
    pd.DataFrame({f'{cls_name}_pos_error': pos_error.tolist()}).to_csv(
        os.path.join(metric_root_path, f'{cls_name}_pos_error.csv'), index=False)
    pd.DataFrame({
        f'{cls_name}_orin_error': [orin_error.mean()],
        f'{cls_name}_pos_error': [pos_error.mean()]
    }).to_csv(os.path.join(metric_root_path, f'{cls_name}_mean_error.csv'), index=False)

    return return_str


def get_rgb_image(data_dict):
        try:
            # data_dict['image'] 形状为 (5, 3, H, W)，第一个模态为RGB
            rgb_data = data_dict['image'][0]  # (3, H, W)

            # 转换为 (H, W, 3)
            rgb_image = rgb_data.transpose(1, 2, 0)  # 现在形状是 (H, W, 3)

            # 处理数据类型：确保是float32并归一化到[0,1]
            if rgb_image.dtype != np.float32:
                rgb_image = rgb_image.astype(np.float32)

            # 确保数据在合理范围内
            if np.max(rgb_image) > 1.0:
                # 如果值超过1.0，假设是[0,255]范围，转换到[0,1]
                rgb_image = rgb_image / 255.0

            # 处理可能的负值
            rgb_image = np.clip(rgb_image, 0.0, 1.0)

            # 转换为uint8 [0,255]
            rgb_image = (rgb_image * 255).astype(np.uint8)

            # 确保是RGB格式（如果是BGR则转换）
            if rgb_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            return rgb_image
        except Exception as e:
            print(f"Extracting RGB image failed: {e}")
            print(f"RGB data shape: {data_dict['image'][0].shape}, data type: {data_dict['image'][0].dtype}")
            return None


def visualize_encode_decode(data_dict, gt_box9d, pred_box9d_9points, save_dir, save_prefix):
        # 确保根目录存在（不创建子文件夹）
        os.makedirs(save_dir, exist_ok=True)  # 只创建根目录，不递归

        # 生成文件名：移除原 save_prefix 中的路径分隔符，避免创建子文件夹
        # 例如将 "Town01_Opt/carla_data/00004/xxx" 转为 "Town01_Opt_carla_data_00004_xxx"
        safe_prefix = save_prefix.replace('/', '_').replace('\\', '_')  # 替换所有路径分隔符
        full_save_path = os.path.join(save_dir, f"{safe_prefix}_compare.png")  # 直接保存在根目录

        # 获取 RGB 图像
        rgb_image = get_rgb_image(data_dict)
        if rgb_image is None:
            print("Unable to get RGB image, skipping visualization")
            return

        # 2. 绘制真实边界框
        intrinsic = data_dict['intrinsic'][0]  # 取第一个相机内参
        extrinsic = data_dict['extrinsic'][0]  # 取第一个相机外参
        distortion = data_dict['distortion'][0]

        new_im_width, new_im_height = data_dict['new_im_size']

        # 绘制原始 3D 框
        rgb_with_gt = draw_box9d_on_image_gt(
            gt_box9d,
            rgb_image.copy(),
            img_width=new_im_width,
            img_height=new_im_height,
            intrinsic_mat=intrinsic,
            extrinsic_mat=extrinsic,
            distortion_matrix=distortion
        )

        # 绘制解码后的 3D 框
        rgb_with_pred = draw_box9d_on_image_gt(
            pred_box9d_9points,
            rgb_image.copy(),
            img_width=new_im_width,
            img_height=new_im_height,
            intrinsic_mat=intrinsic,
            extrinsic_mat=extrinsic,
            distortion_matrix=distortion
        )

        # 拼接并保存
        combined = np.hstack([rgb_with_gt, rgb_with_pred])
        try:
            Image.fromarray(combined).save(full_save_path)
            print(f"Comparison image saved: {full_save_path}")
        except Exception as e:
            print(f"Failed to save comparison image: {e}")
