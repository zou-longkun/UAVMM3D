from uavdet3d.utils.object_encoder_laam6d import all_object_encoders, center_point_decoder
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap
import torch
import cv2
from functools import partial
import copy
import os
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt


class DataPreProcessorLidarFusion():
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
    
    def map_name_to_index(self, gt_names, class_name_config):
        if len(class_name_config) == 1:
            return np.zeros(len(gt_names), dtype=np.int32)
        else:
            return np.array([class_name_config.index(name) for name in gt_names], dtype=np.int32)

    def convert_box9d_to_heatmap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_heatmap, config=config)
        
        center_rad = self.dataset_cfg.CENTER_RAD
        offset = config.OFFSET
        key_point_encoder = all_object_encoders[config.ENCODER]
        encode_corner = np.array(config.ENCODER_CORNER)

        # 处理RGB和IR两个分支的数据
        intrinsic = data_dict['rgb_intrinsic']  # 使用RGB内参作为主要参考
        extrinsic = data_dict['rgb_extrinsic']  # 使用RGB外参作为主要参考
        distortion = data_dict['distortion']
        raw_w, raw_h = data_dict['raw_im_size']
        new_w, new_h = data_dict['new_im_size']
        stride = data_dict['stride']
        box9d = data_dict['gt_boxes']  # (N, 9, 3)

        # 转换9点框为9参数（无local_prototype）
        box9d_params, _ = self.convert_9points_to_9params(box9d)
        centers_world = box9d_params[:, :3]
        extrinsic_inv = np.linalg.inv(extrinsic)
        centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)
        box9d_params[:, :3] = centers_cv

        # 编码关键点
        pts3d, pts2d = key_point_encoder(
            box9d_params,
            encode_corner=encode_corner,
            intrinsic_mat=np.array([intrinsic]),  # 转换为数组形式
            extrinsic_mat=np.array([extrinsic]),  # 转换为数组形式
            distortion_matrix=np.array([distortion]),  # 转换为数组形式
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

    # 修复梯度计算问题的自定义编码器函数
    def safe_center_point_encoder(self, gt_box9d_with_cls, intrinsic_mat, extrinsic_mat, distortion_matrix,
                                new_im_width, new_im_hight, raw_im_width, raw_im_hight, stride,
                                im_num, class_name_config, center_rad):
        cls_num = len(class_name_config)
        gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []
        
        # 调试统计信息
        # total_objects = len(gt_box9d_with_cls)
        # print(f"Total objects: {total_objects}")
        filtered_objects = 0
        z_filtered = 0
        out_of_bounds = 0

        scale_heatmap_w = 1.0 / stride
        scale_heatmap_h = 1.0 / stride

        # 注意：这里使用的xyz已经是OpenCV相机坐标系下的坐标
        # 在convert_box9d_to_centermap中已通过decode_box_centers_from_world完成了世界坐标到OpenCV相机坐标的转换
        xyz = copy.deepcopy(gt_box9d_with_cls[:, 0:3])

        for im_id in range(im_num):
            hm_height = new_im_hight // stride
            hm_width = new_im_width // stride

            this_heat_map = torch.zeros(cls_num, hm_height, hm_width)
            this_res_map = np.zeros(shape=(2, hm_height, hm_width))
            this_dis_map = np.zeros(shape=(1, hm_height, hm_width))
            this_size_map = np.zeros(shape=(3, hm_height, hm_width))
            this_angle_map = np.zeros(shape=(6, hm_height, hm_width))

            for obj_i, obj in enumerate(gt_box9d_with_cls):
                this_cls = int(obj[-1])
                X, Y, Z = xyz[obj_i]
                
                # 在OpenCV相机坐标系中，Z必须为正（表示在相机前方）
                if Z <= 1e-6:
                    z_filtered += 1
                    filtered_objects += 1
                    continue

                # 使用OpenCV相机坐标直接进行投影计算
                fx, fy = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
                cx, cy = intrinsic_mat[0, 2], intrinsic_mat[1, 2]
                u = (X / Z) * fx + cx
                v = (Y / Z) * fy + cy

                # 缩放坐标到热力图尺寸
                u_heatmap = u * scale_heatmap_w
                v_heatmap = v * scale_heatmap_h
                center_heatmap = [u_heatmap, v_heatmap]
                
                # 检查投影点是否在热力图范围内
                if not (0 <= v_heatmap < hm_height and 0 <= u_heatmap < hm_width):
                    out_of_bounds += 1
                    filtered_objects += 1
                    continue

                # 使用临时副本避免in-place操作
                heatmap_cls_copy = this_heat_map[this_cls].clone()
                heatmap_cls_copy = draw_gaussian_to_heatmap(heatmap_cls_copy, center_heatmap, center_rad)
                this_heat_map[this_cls] = heatmap_cls_copy

                # 残差图也使用副本
                res_x_copy = this_res_map[0].copy()
                res_y_copy = this_res_map[1].copy()
                res_x_copy, res_y_copy = draw_res_to_heatmap(res_x_copy, res_y_copy, center_heatmap)
                this_res_map[0] = res_x_copy
                this_res_map[1] = res_y_copy

                try:
                    h_idx = int(v_heatmap)
                    w_idx = int(u_heatmap)

                    if 0 <= h_idx < hm_height and 0 <= w_idx < hm_width:
                        # 关键修复：这里的Z已经是OpenCV相机坐标系下的深度值（相机到物体的实际距离）
                        # 在OpenCV坐标系中，Z轴指向相机前方，所以这个Z值直接代表了深度信息
                        this_dis_map[0, h_idx, w_idx] = Z
                        l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]
                        # 注意：a1, a2, a3仍然是世界坐标系下的旋转角度（未转换）
                        this_size_map[0, h_idx, w_idx] = l
                        this_size_map[1, h_idx, w_idx] = w
                        this_size_map[2, h_idx, w_idx] = h

                        this_angle_map[0, h_idx, w_idx] = np.cos(a1)
                        this_angle_map[1, h_idx, w_idx] = np.sin(a1)
                        this_angle_map[2, h_idx, w_idx] = np.cos(a2)
                        this_angle_map[3, h_idx, w_idx] = np.sin(a2)
                        this_angle_map[4, h_idx, w_idx] = np.cos(a3)
                        this_angle_map[5, h_idx, w_idx] = np.sin(a3)
                except:
                    continue
            
            gt_hm.append(this_heat_map.cpu().numpy())
            gt_center_res.append(this_res_map)
            gt_center_dis.append(this_dis_map)
            gt_dim.append(this_size_map)
            gt_rot.append(this_angle_map)

        return np.array(gt_hm), np.array(gt_center_res), np.array(gt_center_dis), np.array(gt_dim), np.array(gt_rot)

    def convert_box9d_to_centermap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_centermap, config=config)

        center_rad = self.dataset_cfg.CENTER_RAD
        stride = data_dict['stride']
        raw_im_size = data_dict['raw_im_size']
        new_im_size = data_dict['new_im_size']
        gt_name = data_dict['gt_names']
        gt_box9d = data_dict['gt_boxes']  # (N, 9, 3)
        intrinsic = data_dict['rgb_intrinsic']  # 使用RGB内参
        extrinsic = data_dict['rgb_extrinsic']  # 使用RGB外参
        distortion = data_dict['distortion']
        
        # 获取特征图尺寸
        H, W = new_im_size[1] // stride, new_im_size[0] // stride
        num_classes = len(self.class_name_config)
        
        # 处理空GT情况
        if len(gt_box9d) == 0:
            data_dict['hm'] = np.zeros((num_classes, H, W), dtype=np.float32)
            data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
            data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
            data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
            data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
            return data_dict

        # 转换9点框为9参数
        box9d_params, _ = self.convert_9points_to_9params(gt_box9d)
        
        # 确保box9d_params有效
        if not box9d_params.size:
            data_dict['hm'] = np.zeros((num_classes, H, W), dtype=np.float32)
            data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
            data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
            data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
            data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
            return data_dict

        # 转换坐标系
        extrinsic_inv = np.linalg.inv(extrinsic)
        centers_world = box9d_params[:, :3]
        centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)
        box9d_params[:, :3] = centers_cv

        # 拼接类别索引
        try:
            cls_index = self.map_name_to_index(gt_name, self.class_name_config).reshape(-1, 1)
            
            # 确保类别索引与box参数长度匹配
            min_length = min(len(box9d_params), len(cls_index))
            if min_length > 0:
                box9d_params = box9d_params[:min_length]
                cls_index = cls_index[:min_length]
                
                # 调用自定义的安全编码器生成特征图
                gt_box9d_with_cls = np.concatenate([box9d_params, cls_index], axis=1)
                gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = self.safe_center_point_encoder(
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
                
                # 设置数据字典
                data_dict['hm'] = gt_hm
                data_dict['center_res'] = gt_center_res
                data_dict['center_dis'] = gt_center_dis / self.dataset_cfg.MAX_DIS
                data_dict['dim'] = gt_dim / self.dataset_cfg.MAX_SIZE
                data_dict['rot'] = gt_rot
                
                # =========================================hm visualization===================================================
                # # 调试信息：检查生成的热力图是否全为零值
                # hm_sum = np.sum(gt_hm)
                # print(f"Debug centermap: Generated heatmap sum: {hm_sum}")
                # if hm_sum == 0:
                #     print(f"Debug centermap: Warning! Generated heatmap is empty despite {len(gt_box9d)} input boxes")
                
                # # Debug代码：保存hm和gt_center_dis为图片
                # try:
                #     # 检查gt_hm的维度信息
                #     print(f"Debug centermap: gt_hm shape: {gt_hm.shape}")
                #     print(f"Debug centermap: gt_center_dis shape: {data_dict['center_dis'].shape}")
                    
                #     # 创建debug目录
                #     debug_dir = 'debug_hm_images'
                #     os.makedirs(debug_dir, exist_ok=True)
                    
                #     # 获取当前时间戳作为文件名的一部分
                #     timestamp = time.strftime('%Y%m%d_%H%M%S')
                    
                #     # 添加调试信息到文件名
                #     objects_info = f"{len(gt_box9d)}obj_{int(hm_sum>0)}hm"
                    
                #     # 修复：确保正确访问热力图数据
                #     # 根据safe_center_point_encoder的返回值，gt_hm的结构应该是[图像数量, 类别数量, 高度, 宽度]
                #     if len(gt_hm.shape) == 4:
                #         # 正确的4维结构：[图像数量, 类别数量, 高度, 宽度]
                #         img_idx = 0  # 第一张图像
                #         num_classes = gt_hm.shape[1]
                        
                #         # 只保存所有类别的叠加图，不再分类别保存
                #         hm_sum_all = np.sum(gt_hm[img_idx], axis=0)
                #         if hm_sum_all.max() > 0:
                #             hm_sum_all = (hm_sum_all / hm_sum_all.max() * 255).astype(np.uint8)
                #             print(f"Debug centermap: Sum heatmap max value: {hm_sum_all.max()}")
                #         else:
                #             hm_sum_all = (hm_sum_all * 255).astype(np.uint8)
                #         hm_sum_colored = cv2.applyColorMap(hm_sum_all, cv2.COLORMAP_JET)
                #         filename_sum = os.path.join(debug_dir, f'hm_all_classes_{objects_info}_{timestamp}.png')
                #         cv2.imwrite(filename_sum, hm_sum_colored)
                #         print(f'Debug: Saved combined heatmap to {filename_sum}')
                #     else:
                #         # 如果维度不是预期的，打印警告并尝试保存原始数据
                #         print(f"Debug centermap: Warning! gt_hm has unexpected shape: {gt_hm.shape}")
                #         # 尝试直接保存原始数据
                #         try:
                #             if gt_hm.max() > 0:
                #                 hm_image = (gt_hm / gt_hm.max() * 255).astype(np.uint8)
                #             else:
                #                 hm_image = (gt_hm * 255).astype(np.uint8)
                            
                #             if len(hm_image.shape) > 2:
                #                 hm_image = np.sum(hm_image, axis=0)
                                
                #             hm_colored = cv2.applyColorMap(hm_image, cv2.COLORMAP_JET)
                #             filename = os.path.join(debug_dir, f'hm_unexpected_shape_{timestamp}.png')
                #             cv2.imwrite(filename, hm_colored)
                #             print(f'Debug: Saved unexpected shape heatmap to {filename}')
                #         except Exception as inner_e:
                #             print(f"Debug centermap: Failed to save unexpected shape heatmap: {inner_e}")
                    
                #     # 可视化gt_center_dis
                #     try:
                #         center_dis = data_dict['center_dis']
                #         print(f"Debug centermap: center_dis max value: {center_dis.max()}")
                #         print(f"Debug centermap: center_dis min value: {center_dis.min()}")
                #         print(f"Debug centermap: center_dis dtype: {center_dis.dtype}")
                        
                #         # 辅助函数：确保数据是正确的uint8类型并应用颜色映射
                #         def apply_color_map_and_save(data, colormap, filename):
                #             # 确保数据是2D
                #             if len(data.shape) > 2:
                #                 data = np.sum(data, axis=0)
                #                 print(f"Debug centermap: Reduced data to 2D shape: {data.shape}")
                            
                #             # 归一化到0-255范围
                #             if data.max() > data.min():
                #                 norm_data = ((data - data.min()) / (data.max() - data.min()) * 255)
                #             else:
                #                 norm_data = data * 255
                            
                #             # 确保是uint8类型
                #             norm_data = np.clip(norm_data, 0, 255).astype(np.uint8)
                #             print(f"Debug centermap: After normalization, dtype: {norm_data.dtype}, shape: {norm_data.shape}")
                            
                #             # 应用颜色映射
                #             colored_data = cv2.applyColorMap(norm_data, colormap)
                            
                #             # 保存图片
                #             cv2.imwrite(filename, colored_data)
                #             print(f'Debug: Saved colored map to {filename}')
                        
                #         # 处理center_dis维度
                #         if len(center_dis.shape) == 3:
                #             # 假设结构为[2, H, W]，其中2可能代表x和y方向的距离
                #             # 分别可视化x和y方向的距离
                #             for i in range(center_dis.shape[0]):
                #                 try:
                #                     dis_map = center_dis[i]
                #                     filename = os.path.join(debug_dir, f'center_dis_dir{i}_{objects_info}_{timestamp}.png')
                #                     apply_color_map_and_save(dis_map, cv2.COLORMAP_VIRIDIS, filename)
                #                 except Exception as inner_e:
                #                     print(f'Debug center_dis direction {i} error: {inner_e}')
                            
                #             # 保存距离的绝对值或欧几里得距离
                #             if center_dis.shape[0] == 2:
                #                 try:
                #                     # 计算欧几里得距离
                #                     euclidean_dis = np.sqrt(center_dis[0]**2 + center_dis[1]**2)
                #                     filename = os.path.join(debug_dir, f'center_dis_euclidean_{objects_info}_{timestamp}.png')
                #                     apply_color_map_and_save(euclidean_dis, cv2.COLORMAP_MAGMA, filename)
                #                 except Exception as inner_e:
                #                     print(f'Debug center_dis euclidean error: {inner_e}')
                #         else:
                #             # 尝试处理其他维度结构
                #             print(f"Debug centermap: center_dis has unexpected shape: {center_dis.shape}")
                #             try:
                #                 # 尝试将多维数据压缩为2D
                #                 if len(center_dis.shape) > 2:
                #                     dis_map = np.sum(center_dis, axis=0)
                #                 else:
                #                     dis_map = center_dis
                                
                #                 filename = os.path.join(debug_dir, f'center_dis_general_{objects_info}_{timestamp}.png')
                #                 apply_color_map_and_save(dis_map, cv2.COLORMAP_INFERNO, filename)
                #             except Exception as inner_e:
                #                 print(f'Debug center_dis general error: {inner_e}')
                #     except Exception as dis_e:
                #         print(f'Debug center_dis visualization error: {dis_e}')
                # except Exception as e:
                #     print(f'Debug save error: {e}')
                # ===================================================================================================
                    
                return data_dict
        except:
            pass
        
        # 任何情况下失败都返回空特征图
        data_dict['hm'] = np.zeros((num_classes, H, W), dtype=np.float32)
        data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
        data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
        data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
        data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
        return data_dict

    def filter_box_outside(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_box_outside, config=config)

        # 使用new_im_size而不是raw_im_size，因为我们要检查的是处理后的图像尺寸
        new_im_width, new_im_height = data_dict['new_im_size']
        # 使用rgb_intrinsic和rgb_extrinsic以保持与convert_box9d_to_centermap一致
        intrinsic = data_dict['rgb_intrinsic']
        extrinsic = data_dict['rgb_extrinsic']
        distortion = data_dict['distortion']
        boxes = data_dict['gt_boxes']
        names = data_dict['gt_names']

        if len(boxes) > 0:
            centers = boxes[:, 0]
            is_valid = ~np.isnan(centers).any(axis=1) & (np.abs(centers) < 1e9).all(axis=1)
        else:
            is_valid = np.array([], dtype=bool)

        if len(boxes) > 0 and np.any(is_valid):
            valid_boxes = boxes[is_valid]

            centers_world = valid_boxes[:, 0]
            extrinsic_inv = np.linalg.inv(extrinsic)
            centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)

            # 投影到图像验证是否在视野内
            rvec, _ = cv2.Rodrigues(np.eye(3))
            tvec = np.zeros((3, 1), dtype=np.float32)
            corners_2d, _ = cv2.projectPoints(centers_cv, rvec, tvec, intrinsic, distortion)
            corners_2d = corners_2d.reshape(-1, 2).astype(int)

            # 使用new_im_size的尺寸进行边界检查
            mask_x = (corners_2d[:, 0] > 0) & (corners_2d[:, 0] < new_im_width)
            mask_y = (corners_2d[:, 1] > 0) & (corners_2d[:, 1] < new_im_height)
            mask = mask_x & mask_y

            final_mask = np.zeros(len(boxes), dtype=bool)
            final_mask[is_valid] = mask
        else:
            final_mask = np.zeros(len(boxes), dtype=bool)

        data_dict['gt_boxes'] = boxes[final_mask]
        data_dict['gt_names'] = names[final_mask] if len(names) > 0 else []
        return data_dict

    def decode_box_centers_from_world(self, centers_world, extrinsic_inv):
        carla_to_opencv = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        centers_world = np.asarray(centers_world).reshape(-1, 3)
        centers_hom = np.hstack([centers_world, np.ones((centers_world.shape[0], 1))])  # (N, 4)
        centers_cam = (extrinsic_inv @ centers_hom.T).T  # 世界→相机坐标系 (N, 4)
        centers_cv = (carla_to_opencv @ centers_cam.T).T  # Carla→OpenCV坐标系 (N, 4)
        return centers_cv[:, :3].astype(np.float32)  # 取前三维

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

            # 4. 计算尺寸（将角点投影到物体坐标系后测量）
            corners_self = (orthogonal_rot.T @ corners_local.T).T  # 转换到物体坐标系
            l = np.max(corners_self[:, 0]) - np.min(corners_self[:, 0])  # x轴长度
            w = np.max(corners_self[:, 1]) - np.min(corners_self[:, 1])  # y轴长度
            h = np.max(corners_self[:, 2]) - np.min(corners_self[:, 2])  # z轴长度

            # 5. 保存归一化的局部原型（用于解码时一致的缩放）
            normalized_prototype = corners_self / np.array([l, w, h]) if (l * w * h) != 0 else self.standard_prototype
            local_prototypes.append(normalized_prototype)

            # 6. 组合为9参数
            box_param = np.array([x, y, z, l, w, h, angle1, angle2, angle3], dtype=np.float32)
            all_box_params.append(box_param)

        return np.array(all_box_params) if len(all_box_params) > 0 else np.array([]), local_prototypes

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

    def image_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalization, config=config)

        # 处理图像归一化
        if 'image' in data_dict and data_dict['image'] is not None:
            # 确保数据是numpy数组
            image = data_dict['image']
            if isinstance(image, np.ndarray):
                # 创建副本以避免修改原数据
                image_copy = np.copy(image)
                
                # 只处理前2种模态(RGB和IR)
                num_modalities = min(2, image_copy.shape[0])
                
                # 使用numpy向量化操作替代for循环，提高效率
                # 1. 处理NaN和Inf值
                # 将NaN替换为0，将Inf替换为有效数据中的最大值
                mask_nan = np.isnan(image_copy)
                mask_inf = np.isinf(image_copy)
                
                if np.any(mask_nan) or np.any(mask_inf):
                    # 计算有效数据的最大值
                    valid_data = image_copy[np.isfinite(image_copy)]
                    max_valid = np.max(valid_data) if valid_data.size > 0 else 0.0
                    # 替换NaN和Inf
                    image_copy[mask_nan] = 0.0
                    image_copy[mask_inf & (image_copy > 0)] = max_valid
                    image_copy[mask_inf & (image_copy < 0)] = 0.0
                
                # 2. 对RGB和IR模态进行归一化(除以255.0)
                # 只处理前2种模态
                image_copy[:num_modalities] = image_copy[:num_modalities] / 255.0
                
                # 如果有原图融合的数据，也进行归一化处理
                if 'fused_image' in data_dict and data_dict['fused_image'] is not None:
                    # 前3通道是RGB，中间3通道是IR，后6通道是LiDAR和Radar投影数据
                    data_dict['fused_image'][:6] = data_dict['fused_image'][:6] / 255.0             
                # 转换为float32类型
                data_dict['image'] = image_copy.astype(np.float32)
        
        return data_dict

    def __call__(self, data_dict):
        # 处理相机参数格式，确保它们是正确的数组形式
        if 'rgb_intrinsic' in data_dict:
            data_dict['rgb_intrinsic'] = np.array(data_dict['rgb_intrinsic'], dtype=np.float32)
        if 'rgb_extrinsic' in data_dict:
            data_dict['rgb_extrinsic'] = np.array(data_dict['rgb_extrinsic'], dtype=np.float32)
        if 'ir_intrinsic' in data_dict:
            data_dict['ir_intrinsic'] = np.array(data_dict['ir_intrinsic'], dtype=np.float32)
        if 'ir_extrinsic' in data_dict:
            data_dict['ir_extrinsic'] = np.array(data_dict['ir_extrinsic'], dtype=np.float32)
        if 'distortion' in data_dict:
            data_dict['distortion'] = np.array(data_dict['distortion'], dtype=np.float32)

        # 应用数据处理器队列中的所有处理器
        for func in self.data_processor_queue:
            data_dict = func(data_dict)
        
        return data_dict