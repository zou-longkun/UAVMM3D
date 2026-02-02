import numpy as np
from uavdet3d.utils.object_encoder_laam6d import all_object_encoders, center_point_decoder
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap
import torch
import cv2
from functools import partial
import copy
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt


class DataPreProcessorLAAm6d():
    def __init__(self, dataset_cfg, training):
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.data_processor_queue = []
        self.class_name_config = self.dataset_cfg.CLASS_NAMES

        # 初始化数据处理器队列
        for cur_cfg in self.dataset_cfg.DATA_PRE_PROCESSOR:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

        # 标准立方体原型（物体自身坐标系下的8个角点，归一化到[-0.5, 0.5]）
        # 格式：[底面左下, 底面右下, 底面右上, 底面左上, 顶面左下, 顶面右下, 顶面右上, 顶面左上]
        self.standard_prototype = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ])

    def convert_box9d_to_heatmap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_heatmap, config=config)
        center_rad = self.dataset_cfg.CENTER_RAD
        offset = config.OFFSET
        key_point_encoder = all_object_encoders[config.ENCODER]
        encode_corner = np.array(config.ENCODER_CORNER)

        intrinsic = data_dict['intrinsic']
        extrinsic = data_dict['extrinsic']
        distortion = data_dict['distortion']
        raw_w, raw_h = data_dict['raw_im_size']
        new_w, new_h = data_dict['new_im_size']
        stride = data_dict['stride']
        box9d = data_dict['gt_boxes']  # (N, 9, 3)

        # 转换9点框为9参数（无local_prototype）
        box9d_params = self.convert_9points_to_9params(box9d)
        centers_world = box9d_params[:, :3]
        extrinsic_inv = np.linalg.inv(extrinsic[0])
        centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)
        box9d_params[:, :3] = centers_cv

        # 编码关键点
        pts3d, pts2d = key_point_encoder(
            box9d_params,
            encode_corner=encode_corner,
            intrinsic_mat=intrinsic,
            extrinsic_mat=extrinsic,
            distortion_matrix=distortion,
            offset=offset
        )

        num_obj, num_pts = pts2d.shape[0], pts2d.shape[1]
        H, W = new_h // stride, new_w // stride

        # 初始化热力图和残差
        heatmap = torch.zeros((num_obj, num_pts, H, W), dtype=torch.float32)
        residual_x = torch.zeros_like(heatmap)
        residual_y = torch.zeros_like(heatmap)

        for ob_id in range(num_obj):
            for k in range(num_pts):
                pt = pts2d[ob_id, k].copy()
                pt[0] *= (new_w / raw_w / stride)
                pt[1] *= (new_h / raw_h / stride)

                # 绘制高斯热力图
                heatmap[ob_id, k] = draw_gaussian_to_heatmap(heatmap[ob_id, k], pt, center_rad)

                # 绘制残差
                res_x_np = residual_x[ob_id, k].cpu().numpy()
                res_y_np = residual_y[ob_id, k].cpu().numpy()
                res_x_np, res_y_np = draw_res_to_heatmap(res_x_np, res_y_np, pt)
                residual_x[ob_id, k] = torch.from_numpy(res_x_np)
                residual_y[ob_id, k] = torch.from_numpy(res_y_np)

        data_dict['gt_heatmap'] = heatmap
        data_dict['gt_res_x'] = residual_x
        data_dict['gt_res_y'] = residual_y
        data_dict['gt_pts2d'] = pts2d
        return data_dict

    def convert_box9d_to_centermap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_centermap, config=config)

        center_rad = self.dataset_cfg.CENTER_RAD
        offset = config.OFFSET
        stride = data_dict['stride']
        raw_im_size = data_dict['raw_im_size']
        new_im_size = data_dict['new_im_size']
        gt_name = data_dict['gt_names']
        gt_box9d = data_dict['gt_boxes']  # (N, 9, 3)
        intrinsic = data_dict['intrinsic'][0]
        extrinsic = data_dict['extrinsic'][0]
        distortion = data_dict['distortion'][0]
        center_point_encoder = all_object_encoders[config.ENCODER]

        if len(gt_box9d) == 0:
            H, W = new_im_size[1] // stride, new_im_size[0] // stride
            data_dict['hm'] = np.zeros((len(self.class_name_config), H, W), dtype=np.float32)
            data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
            data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
            data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
            data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
            return data_dict

        # 转换9点框为9参数
        box9d_params, local_prototypes = self.convert_9points_to_9params(gt_box9d)

        extrinsic_inv = np.linalg.inv(extrinsic)
        centers_world = box9d_params[:, :3]
        centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)
        box9d_params[:, :3] = centers_cv

        # 拼接类别索引
        cls_index = self.map_name_to_index(gt_name, self.class_name_config).reshape(-1, 1)
        gt_box9d_with_cls = np.concatenate([box9d_params, cls_index], axis=1)

        # 生成中心图相关特征
        H, W = new_im_size[1] // stride, new_im_size[0] // stride
        gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = center_point_encoder(
            gt_box9d_with_cls,
            intrinsic,
            extrinsic,
            distortion,
            new_im_size[0],
            new_im_size[1],
            raw_im_size[0],
            raw_im_size[1],
            stride,
            im_num=self.dataset_cfg.IM_NUM,
            class_name_config=self.class_name_config,
            center_rad=center_rad
        )

        data_dict['hm'] = gt_hm
        data_dict['center_res'] = gt_center_res
        data_dict['center_dis'] = gt_center_dis / self.dataset_cfg.MAX_DIS
        data_dict['dim'] = gt_dim / self.dataset_cfg.MAX_SIZE
        data_dict['rot'] = gt_rot

        # self._save_visualization(data_dict, save_dir, save_prefix)

        # 解码验证（可选）
        # pred_boxes9d, pred_conf = center_point_decoder(
        #     gt_hm,
        #     gt_center_res,
        #     gt_center_dis,
        #     gt_dim,
        #     gt_rot,
        #     intrinsic,
        #     extrinsic,
        #     distortion,
        #     new_im_width=new_im_size[0],
        #     new_im_hight=new_im_size[1],
        #     raw_im_width=raw_im_size[0],
        #     raw_im_hight=raw_im_size[1],
        #     stride=stride,
        #     im_num=1,
        #     max_num=7
        # )
        # pred_boxes_params = pred_boxes9d[:, :-1]  # 移除类别
        # pred_box9d = self.convert_9params_to_9points(pred_boxes_params, local_prototypes)
        #
        # print(gt_box9d)
        # print(pred_box9d)

        # # 可视化对比
        # self._visualize_encode_decode(
        #     data_dict,
        #     gt_box9d,
        #     pred_box9d,
        #     save_dir="encode_decode_vis",
        #     save_prefix=f"{data_dict.get('seq_id')}_{data_dict.get('frame_id')}"
        # )

        return data_dict

    def filter_box_outside(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_box_outside, config=config)

        raw_im_width, raw_im_height = data_dict['raw_im_size']
        intrinsic = data_dict['intrinsic']
        extrinsic = data_dict['extrinsic']
        distortion = data_dict['distortion']
        boxes = data_dict['gt_boxes']
        names = data_dict['gt_names']
        diffs = data_dict.get('gt_diffs', np.zeros(len(boxes)))

        if len(boxes) > 0:
            centers = boxes[:, 0]
            is_valid = ~np.isnan(centers).any(axis=1) & (np.abs(centers) < 1e9).all(axis=1)
        else:
            is_valid = np.array([], dtype=bool)

        if len(boxes) > 0 and np.any(is_valid):
            valid_boxes = boxes[is_valid]
            valid_names = names[is_valid] if len(names) > 0 else []
            valid_diffs = diffs[is_valid] if len(diffs) > 0 else []

            centers_world = valid_boxes[:, 0]
            extrinsic_inv = np.linalg.inv(extrinsic[0])
            centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)

            # 投影到图像验证是否在视野内
            rvec, _ = cv2.Rodrigues(np.eye(3))
            tvec = np.zeros((3, 1), dtype=np.float32)
            corners_2d, _ = cv2.projectPoints(centers_cv, rvec, tvec, intrinsic[0], distortion[0])
            corners_2d = corners_2d.reshape(-1, 2).astype(int)

            mask_x = (corners_2d[:, 0] > 0) & (corners_2d[:, 0] < raw_im_width)
            mask_y = (corners_2d[:, 1] > 0) & (corners_2d[:, 1] < raw_im_height)
            mask = mask_x & mask_y

            final_mask = np.zeros(len(boxes), dtype=bool)
            final_mask[is_valid] = mask
        else:
            final_mask = np.zeros(len(boxes), dtype=bool)

        data_dict['gt_boxes'] = boxes[final_mask]
        data_dict['gt_names'] = names[final_mask] if len(names) > 0 else []
        data_dict['gt_diffs'] = diffs[final_mask] if len(diffs) > 0 else []
        return data_dict

    def map_name_to_index(self, gt_names, class_name_config):
        return np.array([class_name_config.index(name) for name in gt_names], dtype=np.int32)

    def decode_box_centers_from_world(self, centers_world, extrinsic_inv):
        carla_to_opencv = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        centers_world = np.asarray(centers_world).reshape(-1, 3)
        centers_hom = np.hstack([centers_world, np.ones((centers_world.shape[0], 1))])  # (N, 4)
        centers_cam = (extrinsic_inv @ centers_hom.T).T  # 世界→相机坐标系 (N, 4)
        centers_cv = (carla_to_opencv @ centers_cam.T).T  # Carla→OpenCV坐标系 (N, 4)
        return centers_cv[:, :3].astype(np.float32)  # 取前三维

    def image_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalization, config=config)

        image = data_dict['image']  # (5, 3, H, W)
        for i in range(image.shape[0]):
            for c in range(image.shape[1]):
                img = image[i, c]
                # 处理NaN/Inf
                if np.isnan(img).any() or np.isinf(img).any():
                    finite_vals = img[np.isfinite(img)]
                    max_val = np.max(finite_vals) if len(finite_vals) > 0 else 0.0
                    img = np.nan_to_num(img, nan=0.0, posinf=max_val, neginf=0.0)
                # 归一化
                if i in [0, 1, 2]:  # RGB模态
                    image[i, c] = img / 255.0
                elif i == 3:  # 深度模态
                    image[i, c] = img / self.dataset_cfg.MAX_DIS
                elif i == 4:  # 其他模态
                    max_val = np.max(img)
                    image[i, c] = img / max_val if max_val > 0 else img

        data_dict['image'] = image.astype(np.float32)
        return data_dict

    def __call__(self, data_dict):
        self.col(data_dict)
        for func in self.data_processor_queue:
            data_dict = func(data_dict)
        return data_dict

    def col(self, data_dict):
        # 统一相机参数格式为列表
        data_dict['intrinsic'] = [data_dict['intrinsic'].astype(np.float32)]
        data_dict['extrinsic'] = [data_dict['extrinsic'].astype(np.float32)]
        data_dict['distortion'] = [data_dict['distortion'].astype(np.float32)]

    def draw_box9d_on_image_gt(self, boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
                               intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):

        # 从图像获取实际宽高（优先于参数，避免不一致）
        img_height_actual, img_width_actual = image.shape[:2]
        img_width = img_width_actual if img_width_actual > 0 else img_width
        img_height = img_height_actual if img_height_actual > 0 else img_height

        # 初始化默认相机参数
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
        ], dtype=np.float32)
        extrinsic_inv = np.linalg.inv(extrinsic_mat)

        for i, box in enumerate(boxes9d):
            box_cv = []
            valid_box = True  # 标记当前框是否有效

            # 1. 转换坐标并检查有效性（过滤相机后方的点）
            for pt in box:
                if isinstance(pt, torch.Tensor):
                    pt = pt.cpu().numpy()
                else:
                    pt = np.array(pt, dtype=np.float32)

                # 世界坐标→相机坐标（齐次变换）
                pt_hom = np.append(pt, 1.0)  # (4,)
                pt_cam = extrinsic_inv @ pt_hom  # 世界→相机坐标系
                pt_cv = carla_to_opencv @ pt_cam  # Carla→OpenCV相机坐标系

                # 检查相机坐标系下的Z值（必须>0，否则在相机后方，投影无效）
                if pt_cv[2] <= 1e-3:  # Z≤0，标记框无效
                    valid_box = False
                    break

                box_cv.append(pt_cv[:3])  # 保留3D坐标

            # 跳过无效框（存在相机后方的点）
            if not valid_box or len(box_cv) < 9:
                print(f"跳过无效框 {i}（存在相机后方的点或坐标不完整）")
                continue

            box_cv = np.array(box_cv, dtype=np.float32)  # (9, 3)

            # 2. 3D→2D投影并处理异常坐标
            corners_2d, _ = cv2.projectPoints(
                box_cv[1:],  # 8个角点（跳过中心点）
                np.eye(3),  # 无额外旋转
                np.zeros(3),  # 无额外平移
                intrinsic_mat,
                distortion_matrix
            )
            corners_2d = corners_2d.reshape(-1, 2)  # (8, 2)

            # 处理异常值（NaN/Inf/超出图像范围）
            mask_nan = np.isnan(corners_2d) | np.isinf(corners_2d)
            if np.any(mask_nan):
                # 按位置替换：x异常则用img_width/2，y异常则用img_height/2
                corners_2d = np.where(
                    mask_nan,
                    [img_width / 2, img_height / 2],  # 替换值（x和y分别对应）
                    corners_2d
                )
                print(f"框 {i} 存在异常坐标，已替换为图像中心")

            # 步骤2：裁剪坐标到图像范围内（避免极端值）
            corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, img_width - 1)  # x∈[0, width-1]
            corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, img_height - 1)  # y∈[0, height-1]

            # 转换为整数坐标（确保无溢出）
            corners_2d = corners_2d.astype(int)

            # 3. 绘制框的边缘（按原逻辑保留连线顺序）
            bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            middle_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
            top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
            diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]

            # 绘制底部、中间、顶部边缘
            for edge in bottom_edges + middle_edges + top_edges:
                idx1, idx2 = edge
                pt1 = (corners_2d[idx1, 0], corners_2d[idx1, 1])
                pt2 = (corners_2d[idx2, 0], corners_2d[idx2, 1])
                cv2.line(image, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)  # 抗锯齿

            # 绘制对角线（修改后）
            for idx1, idx2 in diagonals:
                # 获取3D角点并检查有效性
                pt3d_1 = box_cv[1:][idx1]  # 第1个3D角点
                pt3d_2 = box_cv[1:][idx2]  # 第2个3D角点

                # 检查3D点是否有异常（NaN/Inf或Z≤0）
                if (np.isnan(pt3d_1).any() or np.isinf(pt3d_1).any() or pt3d_1[2] <= 1e-3 or
                        np.isnan(pt3d_2).any() or np.isinf(pt3d_2).any() or pt3d_2[2] <= 1e-3):
                    print(f"框 {i} 的对角线角点无效，跳过绘制")
                    continue

                # 3D→2D投影
                proj_pts, _ = cv2.projectPoints(
                    np.vstack([pt3d_1, pt3d_2]),
                    np.eye(3),
                    np.zeros(3),
                    intrinsic_mat,
                    distortion_matrix
                )
                proj_pts = proj_pts.reshape(-1, 2)  # (2, 2)：两个端点的2D坐标

                # 处理投影后的异常值（NaN/Inf）
                mask_nan = np.isnan(proj_pts) | np.isinf(proj_pts)
                if np.any(mask_nan):
                    # 用图像中心替换异常值
                    proj_pts = np.where(mask_nan, [img_width / 2, img_height / 2], proj_pts)
                    print(f"框 {i} 的对角线投影有异常值，已替换")

                # 裁剪坐标到图像范围内
                proj_pts[:, 0] = np.clip(proj_pts[:, 0], 0, img_width - 1)
                proj_pts[:, 1] = np.clip(proj_pts[:, 1], 0, img_height - 1)

                # 转换为整数（此时已无NaN）
                p1 = (int(proj_pts[0, 0]), int(proj_pts[0, 1]))
                p2 = (int(proj_pts[1, 0]), int(proj_pts[1, 1]))

                cv2.line(image, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def _get_rgb_image(self, data_dict):
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
            print(f"提取RGB图像失败: {e}")
            print(f"RGB数据形状: {data_dict['image'][0].shape}, 数据类型: {data_dict['image'][0].dtype}")
            return None

    # 修正：生成热力图可视化并确保格式正确
    def _get_heatmap_image(self, heatmap, class_names):
        try:
            # heatmap 形状为 (num_classes, H, W)
            num_classes, h, w = heatmap.shape

            # 合并所有类别的热力图（取最大值）
            combined_heatmap = np.max(heatmap, axis=0)  # 形状 (H, W)

            # 处理数据类型和范围
            if combined_heatmap.dtype != np.float32:
                combined_heatmap = combined_heatmap.astype(np.float32)

            # 归一化到[0,255]
            max_val = np.max(combined_heatmap)
            if max_val > 0:
                combined_heatmap = (combined_heatmap / max_val * 255).astype(np.uint8)
            else:
                combined_heatmap = np.zeros_like(combined_heatmap, dtype=np.uint8)

            # 应用颜色映射
            heatmap_colored = cv2.applyColorMap(combined_heatmap, cv2.COLORMAP_JET)
            # 转换为RGB格式
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            return heatmap_colored
        except Exception as e:
            print(f"生成热力图失败: {e}")
            print(f"热力图形状: {heatmap.shape}, 数据类型: {heatmap.dtype}")
            return None

    # 保存可视化结果的主函数，增加形状检查
    def _save_visualization(self, data_dict, save_dir, save_prefix):
        # 1. 提取RGB图像并绘制真实边界框
        rgb_image = self._get_rgb_image(data_dict)
        if rgb_image is None:
            print("无法获取有效的RGB图像，跳过保存")
            return

        # 检查RGB图像形状
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            print(f"无效的RGB图像形状: {rgb_image.shape}，跳过保存")
            return

        # 2. 绘制真实边界框
        gt_boxes = data_dict['gt_boxes']
        intrinsic = data_dict['intrinsic'][0]  # 取第一个相机内参
        extrinsic = data_dict['extrinsic'][0]  # 取第一个相机外参
        distortion = data_dict['distortion'][0]
        new_im_width, new_im_height = data_dict['new_im_size']

        if len(gt_boxes) > 0:
            rgb_with_gt = self.draw_box9d_on_image_gt(
                gt_boxes,
                rgb_image.copy(),
                img_width=new_im_width,
                img_height=new_im_height,
                intrinsic_mat=intrinsic,
                extrinsic_mat=extrinsic,
                distortion_matrix=distortion
            )
        else:
            rgb_with_gt = rgb_image.copy()

        # 3. 生成热力图
        heatmap = self._get_heatmap_image(data_dict['hm'][0], self.class_name_config)
        if heatmap is None:
            print("无法获取有效的热力图，跳过保存")
            return

        # 检查热力图形状
        if len(heatmap.shape) != 3 or heatmap.shape[2] != 3:
            print(f"无效的热力图形状: {heatmap.shape}，跳过保存")
            return

        # 4. 拼接RGB图和热力图（水平拼接）
        # 确保两者尺寸一致
        heatmap_resized = cv2.resize(heatmap, (rgb_with_gt.shape[1], rgb_with_gt.shape[0]))
        combined = np.hstack((rgb_with_gt, heatmap_resized))

        # 确保拼接后的图像格式正确
        if combined.dtype != np.uint8:
            combined = (combined / np.max(combined) * 255).astype(np.uint8)
        else:
            # 确保值在[0,255]范围内
            combined = np.clip(combined, 0, 255)

        # 5. 保存拼接后的图像
        save_path = os.path.join(save_dir, f"{save_prefix}_rgb_with_gt_and_heatmap.png")
        try:
            img = Image.fromarray(combined)
            img.save(save_path)
            print(f"已保存带边界框和热力图的拼接图像: {save_path}")
        except Exception as e:
            print(f"保存图像失败: {e}")
            print(f"拼接后图像形状: {combined.shape}, 数据类型: {combined.dtype}")

    def _visualize_encode_decode(self, data_dict, gt_box9d, pred_box9d_9points, save_dir, save_prefix):
        # 确保根目录存在（不创建子文件夹）
        os.makedirs(save_dir, exist_ok=True)  # 只创建根目录，不递归

        # 生成文件名：移除原 save_prefix 中的路径分隔符，避免创建子文件夹
        # 例如将 "Town01_Opt/carla_data/00004/xxx" 转为 "Town01_Opt_carla_data_00004_xxx"
        safe_prefix = save_prefix.replace('/', '_').replace('\\', '_')  # 替换所有路径分隔符
        full_save_path = os.path.join(save_dir, f"{safe_prefix}_compare.png")  # 直接保存在根目录

        # 获取 RGB 图像
        rgb_image = self._get_rgb_image(data_dict)
        if rgb_image is None:
            print("无法获取RGB图像，跳过可视化")
            return

        # 2. 绘制真实边界框
        intrinsic = data_dict['intrinsic'][0]  # 取第一个相机内参
        extrinsic = data_dict['extrinsic'][0]  # 取第一个相机外参
        distortion = data_dict['distortion'][0]

        new_im_width, new_im_height = data_dict['new_im_size']

        # 绘制原始 3D 框
        rgb_with_gt = self.draw_box9d_on_image_gt(
            gt_box9d,
            rgb_image.copy(),
            img_width=new_im_width,
            img_height=new_im_height,
            intrinsic_mat=intrinsic,
            extrinsic_mat=extrinsic,
            distortion_matrix=distortion
        )

        # 绘制解码后的 3D 框
        rgb_with_pred = self.draw_box9d_on_image_gt(
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
            print(f"已保存对比图: {full_save_path}")
        except Exception as e:
            print(f"保存对比图失败: {e}")

    def convert_9points_to_9params(self, box9d_points):
        """
        编码：9点框→9参数，返回参数和局部原型（避免依赖实例变量）
        :param box9d_points: (N, 9, 3) 9点框 [中心点, 角点1-8]
        :return:
            box9d_params: (N, 9) 9参数 [x,y,z,l,w,h,a1,a2,a3]
            local_prototypes: 列表，每个元素为(N, 8, 3) 物体坐标系下的归一化角点
        """
        all_box_params = []
        local_prototypes = []  # 局部变量存储，不依赖self

        for single_box in box9d_points:
            # 1. 提取中心点和局部角点
            center = single_box[0].copy()  # (3,)
            x, y, z = center
            corners_local = single_box[1:] - center  # (8, 3) 角点相对中心点的偏移

            # 2. 计算旋转矩阵（与解码逻辑一致：基于欧拉角转换）
            # 先通过SVD获取初始旋转矩阵
            H = self.standard_prototype.T @ corners_local
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
            local_prototypes.append(local_prototype)

            # 8. 组合9参数
            box_params = np.array([x, y, z, l, w, h, angle1, angle2, angle3])
            all_box_params.append(box_params)

        return np.array(all_box_params), local_prototypes

    def convert_9params_to_9points(self, box9d_params, local_prototypes):
        """
        解码：9参数→9点框，需传入编码时的局部原型
        :param box9d_params: (N, 9) 9参数 [x,y,z,l,w,h,a1,a2,a3]
        :param local_prototypes: 编码时返回的局部原型列表（长度=N）
        :return: (N, 9, 3) 9点框 [中心点, 角点1-8]
        """
        if len(box9d_params) != len(local_prototypes):
            raise ValueError("参数数量与局部原型数量不匹配，请确保一一对应")

        all_box_points = []
        for i, params in enumerate(box9d_params):
            # 1. 解析9参数
            x, y, z = params[0:3]  # 中心点坐标
            l, w, h = params[3:6]  # 尺寸（长度、宽度、高度）
            angle1, angle2, angle3 = params[6:9]  # zyx顺序欧拉角

            # 2. 获取编码时的局部原型（确保尺寸缩放一致）
            local_prototype = local_prototypes[i]  # (8, 3) 归一化角点

            # 3. 生成物体自身坐标系下的角点（缩放回实际尺寸）
            corners_self = local_prototype * np.array([l, w, h])  # (8, 3)

            # 4. 生成旋转矩阵（与编码逻辑一致）
            rot_mat = R.from_euler('zyx', [angle1, angle2, angle3], degrees=False).as_matrix()
            if np.linalg.det(rot_mat) < 0:
                rot_mat[:, 2] *= -1  # 确保右手坐标系

            # 5. 转换到世界坐标系（旋转+平移）
            corners_world_local = (rot_mat @ corners_self.T).T  # 旋转后的局部坐标
            corners_world = corners_world_local + np.array([x, y, z])  # 世界坐标系绝对坐标

            # 6. 组合为9点框
            box9d = np.vstack([[x, y, z], corners_world])
            all_box_points.append(box9d)

        return np.stack(all_box_points) if len(all_box_points) > 0 else np.array([])
