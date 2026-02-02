from collections import defaultdict
from uavdet3d.datasets import DatasetTemplate
import numpy as np
import os
import torch
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import random
from glob import glob
from PIL import Image
import time
from .ads_metric_laam6d import LAA3D_ADS_Metric
from .dataset_utils import convert_9params_to_9points, convert_9points_to_9params, convert_box9d_to_box_param,\
    convert_box_opencv_to_world, radar_to_velocity_heatmap, draw_box9d_on_image, draw_box9d_on_image_gt

try:
    # 尝试用 tkAGG（有界面环境可用）
    matplotlib.use('tkAGG')
except ImportError:
    # 无界面环境 fallback 到 Agg
    matplotlib.use('Agg')


class LidarBasedFusionDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, training, root_path, logger)
        
        # 初始化参数
        self.IM_NUM = dataset_cfg.get('IM_NUM', 2)
        self.LIDAR_OFFSET = dataset_cfg.get('LIDAR_OFFSET', 0)
        self.RADAR_OFFSET = dataset_cfg.get('RADAR_OFFSET', 0)
        
        # 模态设置
        self.modalities = ['rgb', 'ir', 'dvs']
        self.im_path_names = {
            'rgb': 'images_rgb',
            'ir': 'images_ir',
            'dvs': 'images_dvs'
        }
        
        # 其他参数
        self.im_resize = dataset_cfg.get('IM_RESIZE')
        self.max_dis = dataset_cfg.get('MAX_DIS')
        self.max_size = dataset_cfg.get('MAX_SIZE')
        self.obj_size = np.array(dataset_cfg.OB_SIZE)
        self.stride = dataset_cfg.get('STRIDE')
        
        # 使用父类的mode属性或通过training参数直接判断
        self.sorted_namelist = dataset_cfg.get('CLASS_NAMES')
        
        # 默认畸变参数
        self.default_distortion = np.zeros(5, dtype=np.float32)
        
        # 加载数据集
        self.set_split(0.9)
        self.logger.info("Total samples: {}".format(len(self.infos)))
        
        # 设置图像尺寸
        self.raw_im_width, self.raw_im_hight = self.dataset_cfg.get('IM_SIZE')
        self.new_im_width, self.new_im_hight = self.im_resize
        self.scale_x = self.new_im_width / self.raw_im_width
        self.scale_y = self.new_im_hight / self.raw_im_hight
        
        # 添加对应点缓存功能
        self.precompute_correspondences = dataset_cfg.get('PRECOMPUTE_CORRESPONDENCES', True)  # 是否在数据加载阶段预处理对应点
        self.correspondence_cache = {}  # 缓存对应点结果
        self.cache_size_limit = 1000  # 缓存大小限制，防止内存溢出
    
    def __len__(self):

        return len(self.infos)
    
    def __getitem__(self, index):
        each_info = self.infos[index]

        seq_id = each_info['seq_id']
        frame_id = each_info['frame_id']
        
        # 加载图像数据
        rgb_image = cv2.imread(each_info['im_paths']['rgb'], cv2.IMREAD_COLOR)
        ir_img = cv2.imread(each_info['im_paths']['ir'], cv2.IMREAD_COLOR)
        dvs_img = cv2.imread(each_info['im_paths']['dvs'], cv2.IMREAD_COLOR)

        # 调整图像大小
        rgb_image = cv2.resize(rgb_image, (self.new_im_width, self.new_im_hight))
        ir_img = cv2.resize(ir_img, (self.new_im_width, self.new_im_hight))
        dvs_img = cv2.resize(dvs_img, (self.new_im_width, self.new_im_hight))

        # 堆叠图像数据
        image_modal_stack = []
        for mode in self.modalities:
            if mode == 'rgb':
                img = rgb_image
            elif mode == 'ir':
                img = ir_img
            elif mode == 'dvs':
                img = dvs_img
            else:
                img = cv2.imread(each_info['im_paths'][mode], cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.new_im_width, self.new_im_hight))

            img = img.astype(np.float32)
            if mode == 'ir':
                if len(img.shape) == 2:
                    img = img[np.newaxis, :, :]  # (1, H, W)
                    img = np.repeat(img, 3, axis=0)  # (3, H, W)
                else:
                    img = img.transpose(2, 0, 1)
            else:
                img = img.transpose(2, 0, 1)
            image_modal_stack.append(img)
        image = np.stack(image_modal_stack, axis=0)  # (3, C, H, W)

        # 加载LiDAR点云和雷达数据
        try:
            lidar_data = np.load(each_info['lidar_path'])

            if '00004' in each_info['lidar_path']:  
                print(each_info['lidar_path'])
                self._save_lidar_as_ply(lidar_data, seq_id, frame_id)
            # max_values = np.max(lidar_data, axis=0)  # axis=0 表示按列计算最大值
            # print(f"每个维度的最大值: {max_values}")
            # print(f"维度含义假设: [x, y, z, intensity, flag]")
            # print(f"  x最大: {max_values[0]:.4f}")
            # print(f"  y最大: {max_values[1]:.4f}")
            # print(f"  z最大: {max_values[2]:.4f}")
            # print(f"  intensity最大: {max_values[3]:.4f}")
            # print(f"  flag最大: {max_values[4]:.0f}")
            # min_values = np.min(lidar_data, axis=0)  # axis=0 表示按列计算最大值
            # print(f"每个维度的最小值: {min_values}")
            # print(f"维度含义假设: [x, y, z, intensity, flag]")
            # print(f"  x最小: {min_values[0]:.4f}")
            # print(f"  y最小: {min_values[1]:.4f}")
            # print(f"  z最小: {min_values[2]:.4f}")
            # print(f"  intensity最小: {min_values[3]:.4f}")
            # print(f"  flag最小: {min_values[4]:.0f}")
            
            assert lidar_data.ndim == 2 and lidar_data.shape[1] == 5, "LiDAR format error"
            radar_data = np.load(each_info['radar_path'])
            assert radar_data.ndim == 2 and radar_data.shape[1] == 5, "Radar format error"
        except Exception as e:
            self.logger.error(f"Sample {seq_id}_{frame_id} loading failed: {e}, recommended to clean up this sample")
            return self.__getitem__((index + 1) % len(self))
        
        # 加载相机参数
        with open(each_info['im_info_path'], 'rb') as f:
            camera_info = pickle.load(f)
        
        # 分别获取RGB和IR的内外参
        intrinsic_rgb = np.array(camera_info['rgb']['intrinsic'])
        extrinsic_rgb = np.array(camera_info['rgb']['extrinsic'])
        intrinsic_ir = np.array(camera_info['ir']['intrinsic'])
        extrinsic_ir = np.array(camera_info['ir']['extrinsic'])
        intrinsic_dvs = np.array(camera_info['dvs']['intrinsic'])
        extrinsic_dvs = np.array(camera_info['dvs']['extrinsic'])
        
        # 调整三种图像模态的内参
        resized_intrinsics_rgb = intrinsic_rgb.copy()
        resized_intrinsics_rgb[0, 0] *= self.scale_x
        resized_intrinsics_rgb[1, 1] *= self.scale_y
        resized_intrinsics_rgb[0, 2] *= self.scale_x
        resized_intrinsics_rgb[1, 2] *= self.scale_y
        
        resized_intrinsics_ir = intrinsic_ir.copy()
        resized_intrinsics_ir[0, 0] *= self.scale_x
        resized_intrinsics_ir[1, 1] *= self.scale_y
        resized_intrinsics_ir[0, 2] *= self.scale_x
        resized_intrinsics_ir[1, 2] *= self.scale_y

        resized_intrinsics_dvs = intrinsic_dvs.copy()
        resized_intrinsics_dvs[0, 0] *= self.scale_x
        resized_intrinsics_dvs[1, 1] *= self.scale_y
        resized_intrinsics_dvs[0, 2] *= self.scale_x
        resized_intrinsics_dvs[1, 2] *= self.scale_y

        # 获取Lidar/雷达外参
        lidar_extrinsic = each_info['lidar_extrinsic']
        radar_extrinsic = each_info['radar_extrinsic']

        # 获取畸变参数（所有模态使用默认值）
        distortion = self.default_distortion.copy()
        
        # 加载RGB和IR的标签
        rgb_label_path = each_info['label_paths']['rgb']
        ir_label_path = each_info['label_paths']['ir']
        dvs_label_path = each_info['label_paths']['dvs']
        
        # 处理RGB标签
        rgb_gt_boxes, rgb_gt_names = [], []
        if os.path.exists(rgb_label_path):
            with open(rgb_label_path, 'rb') as f:
                raw_data = pickle.load(f)

            for row in raw_data:
                if isinstance(row[0], str):
                    raw_name = row[0]
                    pts = np.array(row[1:], dtype=np.float32).reshape(8, 3)
                else:
                    raw_name = "Unknown"
                    pts = np.array(row, dtype=np.float32).reshape(8, 3)

                pts_world = convert_box_opencv_to_world(pts, extrinsic_rgb)

                center_world = ((pts_world[0] + pts_world[6]) +
                                (pts_world[1] + pts_world[7]) +
                                (pts_world[2] + pts_world[4]) +
                                (pts_world[3] + pts_world[5])
                                ) / 8.0
                center_world = center_world.reshape(1, 3)
                box_9pts_world = np.concatenate([center_world, pts_world], axis=0)
                rgb_gt_boxes.append(box_9pts_world)

                matched = "Unknown"
                for cls in each_info['cls_name']:
                    if cls in raw_name:
                        matched = cls
                        break
                rgb_gt_names.append(matched)
            assert len(rgb_gt_boxes) == len(rgb_gt_names), f"Boxes/names mismatch: {len(rgb_gt_boxes)} vs {len(rgb_gt_names)}"
            rgb_gt_boxes = np.array(rgb_gt_boxes, dtype=np.float32)  # (N, 9, 3)
            rgb_gt_names = np.array(rgb_gt_names)
            assert len(rgb_gt_boxes) == len(rgb_gt_names), f"Boxes/names mismatch: {len(rgb_gt_boxes)} vs {len(rgb_gt_names)}"
        else:
            self.logger.warning(f"No label file for RGB: {rgb_label_path}")
            rgb_gt_boxes = np.empty((0, 9, 3), dtype=np.float32)
            rgb_gt_names = np.empty((0,), dtype='<U32')
        
        # 处理IR标签
        ir_gt_boxes, ir_gt_names = [], []
        if os.path.exists(ir_label_path):
            with open(ir_label_path, 'rb') as f:
                raw_data = pickle.load(f)

            for row in raw_data:
                if isinstance(row[0], str):
                    raw_name = row[0]
                    pts = np.array(row[1:], dtype=np.float32).reshape(8, 3)
                else:
                    raw_name = "Unknown"
                    pts = np.array(row, dtype=np.float32).reshape(8, 3)

                pts_world = convert_box_opencv_to_world(pts, extrinsic_ir)

                center_world = ((pts_world[0] + pts_world[6]) +
                                (pts_world[1] + pts_world[7]) +
                                (pts_world[2] + pts_world[4]) +
                                (pts_world[3] + pts_world[5])
                                ) / 8.0
                center_world = center_world.reshape(1, 3)
                box_9pts_world = np.concatenate([center_world, pts_world], axis=0)
                ir_gt_boxes.append(box_9pts_world)

                matched = "Unknown"
                for cls in each_info['cls_name']:
                    if cls in raw_name:
                        matched = cls
                        break
                ir_gt_names.append(matched)
            assert len(ir_gt_boxes) == len(ir_gt_names), f"Boxes/names mismatch: {len(ir_gt_boxes)} vs {len(ir_gt_names)}"
            ir_gt_boxes = np.array(ir_gt_boxes, dtype=np.float32)  # (N, 9, 3)
            ir_gt_names = np.array(ir_gt_names)
            assert len(ir_gt_boxes) == len(ir_gt_names), f"Boxes/names mismatch: {len(ir_gt_boxes)} vs {len(ir_gt_names)}"
        else:
            self.logger.warning(f"No label file for IR: {ir_label_path}")
            ir_gt_boxes = np.empty((0, 9, 3), dtype=np.float32)
            ir_gt_names = np.empty((0,), dtype='<U32')
        
        # 处理DVS标签
        dvs_gt_boxes, dvs_gt_names = [], []
        if os.path.exists(dvs_label_path):
            with open(dvs_label_path, 'rb') as f:
                raw_data = pickle.load(f)

            for row in raw_data:
                if isinstance(row[0], str):
                    raw_name = row[0]
                    pts = np.array(row[1:], dtype=np.float32).reshape(8, 3)
                else:
                    raw_name = "Unknown"
                    pts = np.array(row, dtype=np.float32).reshape(8, 3)

                pts_world = convert_box_opencv_to_world(pts, extrinsic_dvs)

                center_world = ((pts_world[0] + pts_world[6]) +
                                (pts_world[1] + pts_world[7]) +
                                (pts_world[2] + pts_world[4]) +
                                (pts_world[3] + pts_world[5])
                                ) / 8.0
                center_world = center_world.reshape(1, 3)
                box_9pts_world = np.concatenate([center_world, pts_world], axis=0)
                dvs_gt_boxes.append(box_9pts_world)

                matched = "Unknown"
                for cls in each_info['cls_name']:
                    if cls in raw_name:
                        matched = cls
                        break
                dvs_gt_names.append(matched)
            assert len(dvs_gt_boxes) == len(dvs_gt_names), f"Boxes/names mismatch: {len(dvs_gt_boxes)} vs {len(dvs_gt_names)}"
            dvs_gt_boxes = np.array(dvs_gt_boxes, dtype=np.float32)  # (N, 9, 3)
            dvs_gt_names = np.array(dvs_gt_names)
            assert len(dvs_gt_boxes) == len(dvs_gt_names), f"Boxes/names mismatch: {len(dvs_gt_boxes)} vs {len(dvs_gt_names)}"
        else:
            self.logger.warning(f"No label file for DVS: {dvs_label_path}")
            dvs_gt_boxes = np.empty((0, 9, 3), dtype=np.float32)
            dvs_gt_names = np.empty((0,), dtype='<U32')

        # 构建两个分支：RGB分支和IR分支
        # RGB分支: rgb + radar(rgb投影) + lidar(rgb投影)
        # IR分支: ir + radar(ir投影) + lidar(ir投影)

        # 1. RGB分支处理
        # 将雷达数据投影到RGB图像上
        if radar_data is not None:
            rgb_radar_projection = self.project_radar_to_image(radar_data, resized_intrinsics_rgb, extrinsic_rgb, radar_extrinsic)
        # 将LiDAR数据投影到RGB图像上
        if len(lidar_data) > 0:
            rgb_lidar_projection = self.project_lidar_to_image(lidar_data, resized_intrinsics_rgb, extrinsic_rgb, lidar_extrinsic)
        
        # 2. IR分支处理
        # 将雷达数据投影到IR图像上
        if radar_data is not None:
            ir_radar_projection = self.project_radar_to_image(radar_data, resized_intrinsics_ir, extrinsic_ir, radar_extrinsic)  
        # 将LiDAR数据投影到IR图像上
        if len(lidar_data) > 0:
            ir_lidar_projection = self.project_lidar_to_image(lidar_data, resized_intrinsics_ir, extrinsic_ir, lidar_extrinsic)
            
        if radar_data is None or len(lidar_data) == 0:
            return self.__getitem__((index + 1) % len(self))

        # 互换 RGB 与 IR 的图像数据
        if self.dataset_cfg.get('SWAP_RGB_IR', True):
            # 交换 image 中 RGB 与 IR 的通道顺序
            image[0], image[1] = image[1].copy(), image[0].copy()
            rgb_image, ir_img = ir_img, rgb_image
            # 互换 RGB 与 IR 的内外参
            resized_intrinsics_rgb, resized_intrinsics_ir = resized_intrinsics_ir, resized_intrinsics_rgb
            extrinsic_rgb, extrinsic_ir = extrinsic_ir, extrinsic_rgb
            # 互换 RGB 与 IR 的 GT 框与类别
            rgb_gt_boxes, ir_gt_boxes = ir_gt_boxes, rgb_gt_boxes
            rgb_gt_names, ir_gt_names = ir_gt_names, rgb_gt_names
            # 互换 RGB 与 IR 的投影结果
            rgb_radar_projection, ir_radar_projection = ir_radar_projection, rgb_radar_projection
            rgb_lidar_projection, ir_lidar_projection = ir_lidar_projection, rgb_lidar_projection
        
        correspondences = None  
        # 检查是否可以进行关键点匹配
        if (self.precompute_correspondences and 
            rgb_lidar_projection is not None and 
            ir_lidar_projection is not None and
            len(rgb_lidar_projection.get('image_points', [])) > 0 and 
            len(ir_lidar_projection.get('image_points', [])) > 0):
            # 创建缓存键
            cache_key = f"{seq_id}_{frame_id}"
            # 检查缓存中是否已有结果
            if cache_key in self.correspondence_cache:
                correspondences = self.correspondence_cache[cache_key]
            else:
                # 构建RGB和IR投影信息
                rgb_proj_info = [
                    (u, v, x, y, z)
                    for (u, v), (x, y, z, *_) in zip(
                        rgb_lidar_projection['image_points'],
                        rgb_lidar_projection['world_coords_data']  # 使用世界坐标系下的点坐标
                    )
                ]
                ir_proj_info = [
                    (u, v, x, y, z)
                    for (u, v), (x, y, z, *_) in zip(
                        ir_lidar_projection['image_points'],
                        ir_lidar_projection['world_coords_data']  # 使用世界坐标系下的点坐标
                    )
                ]
                
                # 调用优化后的方法计算对应点
                visualize = False  # Disable visualization to save memory
                save_path = None
                if visualize:
                    os.makedirs('corresponding_points_vis', exist_ok=True)
                    # 清理seq_id和frame_id中的特殊字符，避免Windows路径问题
                    safe_seq_id = str(seq_id).replace('\\', '_').replace('/', '_').replace(':', '-')
                    safe_frame_id = str(frame_id).replace('\\', '_').replace('/', '_').replace(':', '-')
                    # 移除可能的.png扩展名，避免重复
                    if safe_frame_id.endswith('.png'):
                        safe_frame_id = safe_frame_id[:-4]
                    # 正确生成保存路径
                    save_path = os.path.join('corresponding_points_vis', f'correspondences_{safe_seq_id}_{safe_frame_id}.jpg')

                    ir_img = draw_box9d_on_image_gt(
                        ir_gt_boxes,
                        ir_img.copy(),
                        img_width=ir_img.shape[1],
                        img_height=ir_img.shape[0],
                        intrinsic_mat=resized_intrinsics_ir,
                        extrinsic_mat=extrinsic_ir,
                        distortion_matrix=distortion
                    )
                
                correspondences = self._find_corresponding_points(
                    rgb_proj_info, 
                    ir_proj_info, 
                    gt_boxes=rgb_gt_boxes,  # 传递世界坐标系下的无人机边界框
                    rgb_image=rgb_image,  # 传递RGB图像
                    ir_image=ir_img,      # 传递IR图像
                    visualize=visualize,  # 启用可视化
                    save_path=save_path,  # 保存路径
                    ir_gt_boxes=ir_gt_boxes  # 传递IR图像上的边界框用于可视化
                )
                correspondences = correspondences.cpu().numpy().tolist()
                
                # 缓存结果
                self.correspondence_cache[cache_key] = correspondences
                
                # 当缓存大小超过限制时，清除部分缓存
                if len(self.correspondence_cache) > self.cache_size_limit:
                    # 移除最早添加的10%缓存项
                    for key in list(self.correspondence_cache.keys())[:int(self.cache_size_limit * 0.1)]:
                        del self.correspondence_cache[key]
        
        # 构建数据字典
        data_dict = {
            'image': image,
            'gt_boxes': rgb_gt_boxes,  # 使用RGB分支的GT框作为主要参考
            'gt_names': rgb_gt_names,
            'rgb_gt_boxes': rgb_gt_boxes,
            'ir_gt_boxes': ir_gt_boxes,
            'rgb_intrinsic': resized_intrinsics_rgb,
            'ir_intrinsic': resized_intrinsics_ir,
            'rgb_extrinsic': extrinsic_rgb,
            'ir_extrinsic': extrinsic_ir,
            'intrinsic': resized_intrinsics_rgb,
            'extrinsic': extrinsic_rgb,
            'distortion': distortion,  # (5,)
            'raw_im_size': np.array([self.raw_im_width, self.raw_im_hight]),
            'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': self.obj_size,
            'seq_id': seq_id,
            'frame_id': frame_id,
            'stride': self.stride,
            'sorted_namelist': self.sorted_namelist,
            'lidar_points': lidar_data,
            'lidar_extrinsic': lidar_extrinsic,
            'radar_data': radar_data,
            'radar_extrinsic': radar_extrinsic,
            'rgb_radar_projection': rgb_radar_projection,
            'rgb_lidar_projection': rgb_lidar_projection,
            'ir_radar_projection': ir_radar_projection,
            'ir_lidar_projection': ir_lidar_projection,
            'precomputed_correspondences': correspondences  # 添加预处理的对应点
        }
        
        # 实现原图层面的融合
        if 'precomputed_correspondences' in data_dict and data_dict['precomputed_correspondences'] is not None:
            # 获取预处理后的RGB和IR图像
            # 注意：这里我们需要使用原始图像尺寸进行融合
            # 先重新加载原始尺寸的图像
            rgb_image_original = cv2.imread(each_info['im_paths']['rgb'], cv2.IMREAD_COLOR)
            ir_image_original = cv2.imread(each_info['im_paths']['ir'], cv2.IMREAD_COLOR)
            
            # 调整到处理后的尺寸
            rgb_image_processed = cv2.resize(rgb_image_original, (self.new_im_width, self.new_im_hight))
            ir_image_processed = cv2.resize(ir_image_original, (self.new_im_width, self.new_im_hight))
            
            # 在原图层面进行融合
            fused_image = self._fuse_images_at_original(rgb_image_processed, ir_image_processed, data_dict['precomputed_correspondences'])
            
            # 将融合后的图像添加到数据字典中
            data_dict['fused_image_visual'] = fused_image
            
            # 转换为合适的格式用于网络输入
            # 从(H, W, 6)转换为(6, H, W)
            fused_image_transposed = fused_image.transpose(2, 0, 1)
            data_dict['fused_image'] = fused_image_transposed.astype(np.float32)
        
        # 数据预处理
        data_dict = self.data_pre_processor(data_dict)

        # ====================================multimodal_visualizations_with_gt=========================================
        # def normalize_world_coords(world_coords, colormap_type='BGR'):
        #     # 创建输出图像
        #     h, w = world_coords.shape[:2]
        #     normalized_image = np.zeros((h, w, 3), dtype=np.uint8)
            
        #     # 对每个坐标通道(X, Y, Z)分别进行归一化处理
        #     for i in range(3):
        #         # 获取当前通道数据
        #         coords = world_coords[..., i]
        #         # 创建有效掩码（非零值）
        #         valid_mask = coords != 0
        #         channel_normalized = np.zeros_like(coords, dtype=np.uint8)
                
        #         # 检查是否有有效值
        #         if np.any(valid_mask):
        #             coords_valid = coords[valid_mask]
                    
        #             # 使用百分位数进行异常值处理，保留更多数据点
        #             min_val = np.percentile(coords_valid, 5)
        #             max_val = np.percentile(coords_valid, 95)
                    
        #             # 确保至少有一定的范围
        #             if max_val <= min_val:
        #                 max_val = min_val + 1e-6  # 添加一个小值避免除零
                    
        #             # 对有效区域进行归一化
        #             normalized = (coords_valid - min_val) / (max_val - min_val)
        #             normalized = np.clip(normalized, 0, 1)  # 确保值在0-1范围内
                    
        #             # 转换到0-255范围
        #             channel_normalized[valid_mask] = (normalized * 255).astype(np.uint8)
                
        #         # 将归一化后的通道保存到输出图像中
        #         normalized_image[..., i] = channel_normalized

        #     if colormap_type == 'BGR':
        #         normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)
        #     elif colormap_type == 'JET':
        #         normalized_image = cv2.applyColorMap(normalized_image[:,:,0], cv2.COLORMAP_JET)
        #     else:
        #         normalized_image = cv2.applyColorMap(normalized_image[:,:,0], cv2.COLORMAP_VIRIDIS)
            
        #     return normalized_image

        # def process_radar_velocity_heatmap(velocity_heatmap, fixed_min=-10.0, fixed_max=10.0, velocity_threshold=0.1, colormap_type='BGR'):
        #     # 判断雷达速度是否存在有效的速度变化
        #     if velocity_heatmap.max() != velocity_heatmap.min():
        #         # 创建掩码，只保留速度绝对值大于阈值的区域
        #         mask = np.abs(velocity_heatmap) > velocity_threshold
        #         # 使用固定范围归一化处理
        #         normalized = (velocity_heatmap - fixed_min) / (fixed_max - fixed_min)
        #         normalized = np.clip(normalized, 0, 1)  # 确保值在0-1范围内
        #         # 应用掩码，将背景设为黑色（0）
        #         velocity_img = (normalized * 255 * mask).astype(np.uint8)
        #     else:
        #         velocity_img = np.zeros_like(velocity_heatmap).astype(np.uint8)
            
        #     # 根据指定的色彩映射类型进行处理
        #     if colormap_type == 'BGR':
        #         velocity_img = np.mean(velocity_img, axis=2).astype(np.uint8)
        #         velocity_img = cv2.cvtColor(velocity_img, cv2.COLOR_GRAY2BGR)
        #     elif colormap_type == 'JET':
        #         velocity_img = np.mean(velocity_img, axis=2).astype(np.uint8)
        #         velocity_img = cv2.applyColorMap(velocity_img, cv2.COLORMAP_JET)
        #     # 如果是'GRAY'或其他值，则保持灰度图不变
        #     elif colormap_type == 'GRAY':
        #         velocity_img = cv2.cvtColor(velocity_img, cv2.COLOR_BGR2GRAY)
            
        #     return velocity_img
        
        # def visualize_multi_modal_align(
        #         rgb_image,
        #         ir_image,
        #         dvs_image,
        #         velocity_image,
        #         world_coords_map,
        #         world_coords_map_original,
        #         save_path):
        #     """
        #     Visualize multimodal images, force specify Noto font via font file path to solve Chinese display issues
        #     """
        #     # 确保所有图像尺寸一致
        #     h, w = rgb_image.shape[:2]
        #     assert ir_image.shape[:2] == (h, w), f"IR size mismatch: {ir_image.shape[:2]} vs {(h, w)}"
        #     assert dvs_image.shape[:2] == (h, w), f"DVS size mismatch: {dvs_image.shape[:2]} vs {(h, w)}"
        #     assert velocity_image.shape[:2] == (h, w), f"Velocity size mismatch: {velocity_image.shape[:2]} vs {(h, w)}"
        #     assert world_coords_map.shape[:2] == (h, w), f"World coordinate map size mismatch: {world_coords_map.shape[:2]} vs {(h, w)}"

        #     velocity_image = process_radar_velocity_heatmap(velocity_image, colormap_type='BGR')

        #     # 预处理单通道图像为3通道
        #     if len(ir_image.shape) == 2:
        #         ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            
        #     # 应用增强的归一化函数，考虑训练模式
        #     world_vis = normalize_world_coords(world_coords_map)
        #     world_vis_origin = normalize_world_coords(world_coords_map_original)

        #     # 核心布局参数
        #     plt.figure(figsize=(20, 12))
        #     gs = gridspec.GridSpec(
        #         2, 6,
        #         figure=plt.gcf(),
        #         hspace=0.02,
        #         height_ratios=[5, 3]
        #     )

        #     # 五种模态（包含中文标题）
        #     modalities = [
        #         ("RGB", cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)),
        #         ("IR", cv2.cvtColor(ir_image, cv2.COLOR_BGR2RGB)),
        #         ("DVS", cv2.cvtColor(dvs_image, cv2.COLOR_BGR2RGB)),
        #         ("Radar HM", cv2.cvtColor(velocity_image, cv2.COLOR_BGR2RGB)),
        #         ("Lidar", cv2.cvtColor(world_vis, cv2.COLOR_BGR2RGB))
        #     ]

        #     # 第一行大图
        #     ax1 = plt.subplot(gs[0, 0:3])
        #     ax1.set_title(modalities[0][0], fontsize=16, pad=5)
        #     ax1.imshow(modalities[0][1])
        #     ax1.set_xticks([])
        #     ax1.set_yticks([])

        #     ax2 = plt.subplot(gs[0, 3:6])
        #     ax2.set_title(modalities[1][0], fontsize=16, pad=5)
        #     ax2.imshow(modalities[1][1])
        #     ax2.set_xticks([])
        #     ax2.set_yticks([])

        #     # 第二行小图
        #     ax3 = plt.subplot(gs[1, 0:2])
        #     ax3.set_title(modalities[2][0], fontsize=14, pad=3)
        #     ax3.imshow(modalities[2][1])
        #     ax3.set_xticks([])
        #     ax3.set_yticks([])

        #     ax4 = plt.subplot(gs[1, 2:4])
        #     ax4.set_title(modalities[3][0], fontsize=14, pad=3)
        #     ax4.imshow(modalities[3][1])
        #     ax4.set_xticks([])
        #     ax4.set_yticks([])

        #     ax5 = plt.subplot(gs[1, 4:6])
        #     ax5.set_title(modalities[4][0], fontsize=14, pad=3)
        #     ax5.imshow(modalities[4][1])
        #     ax5.set_xticks([])
        #     ax5.set_yticks([])

        #     # 确保保存目录存在
        #     save_dir = os.path.dirname(save_path)
        #     if save_dir and not os.path.exists(save_dir):
        #         try:
        #             os.makedirs(save_dir, exist_ok=True)
        #             print(f"Directory created: {save_dir}")
        #         except Exception as e:
        #             print(f"Error creating directory: {e}")
            
        #     plt.savefig(
        #         save_path,
        #         dpi=350,
        #         bbox_inches='tight',
        #         pad_inches=0.1
        #     )
        #     plt.close()
        #     print(f"Saved multimodal alignment visualization to: {save_path}")

        # # 直接使用project_radar_to_image方法已经生成的radar_velocity_heatmap, 转换维度顺序 (3, H, W) -> (H, W, 3)
        # velocity_heatmap = np.transpose(rgb_radar_projection['radar_velocity_heatmap'], (1, 2, 0))
        
        # # 使用project_lidar_to_image函数已经生成的world_coords_map, 转换维度顺序 (3, H, W) -> (H, W, 3)
        # world_coords_map = np.transpose(rgb_lidar_projection['world_coords_map'], (1, 2, 0))
        
        # rgb_with_gt = draw_box9d_on_image_gt(
        #     rgb_gt_boxes,
        #     rgb_image.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=resized_intrinsics_rgb,
        #     extrinsic_mat=extrinsic_rgb,
        #     distortion_matrix=distortion
        # )
        # ir_with_gt = draw_box9d_on_image_gt(
        #     ir_gt_boxes,
        #     ir_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=resized_intrinsics_ir,
        #     extrinsic_mat=extrinsic_ir,
        #     distortion_matrix=distortion
        # )
        # dvs_with_gt = draw_box9d_on_image_gt(
        #     dvs_gt_boxes,
        #     dvs_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=resized_intrinsics_dvs,
        #     extrinsic_mat=extrinsic_dvs,
        #     distortion_matrix=distortion
        # )
        # velocity_with_gt = draw_box9d_on_image_gt(
        #     rgb_gt_boxes,
        #     velocity_heatmap.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=resized_intrinsics_rgb,
        #     extrinsic_mat=extrinsic_rgb,
        #     distortion_matrix=distortion
        # )
        # world_coords_map_with_gt = draw_box9d_on_image_gt(
        #     rgb_gt_boxes,
        #     world_coords_map.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=resized_intrinsics_rgb,
        #     extrinsic_mat=extrinsic_rgb,
        #     distortion_matrix=distortion
        # )
        # self.vis_save_dir = os.path.join(os.getcwd(), 'multimodal_visualizations_with_gt')
        # os.makedirs(self.vis_save_dir, exist_ok=True)
        # # 修复f-string中不能直接使用反斜杠的问题
        # seq_id_fixed = seq_id.replace('\\', '_')
        # frame_id_fixed = frame_id.replace('.png', '.jpg')
        # save_filename = f"{seq_id_fixed}_{frame_id_fixed}"
        # save_path = os.path.join(self.vis_save_dir, save_filename)
        
        # visualize_multi_modal_align(
        #     rgb_image=rgb_with_gt,
        #     ir_image=ir_with_gt,
        #     dvs_image=dvs_with_gt,
        #     velocity_image=velocity_with_gt,
        #     world_coords_map=world_coords_map_with_gt,
        #     world_coords_map_original=world_coords_map,
        #     save_path=save_path
        # )
        # ==============================================================================================================
        
        return data_dict
    
    def include_CARLA_data(self, mode):
        self.logger.info('Loading CARLA dataset')
        CARLA_infos = []
        
        seq_groups = defaultdict(list)  # {seq_name: [(frame_name1), (frame_name2), ...], ...}
        for seq_name, frame_name in self.sample_scene_list:
            seq_groups[seq_name].append(frame_name)
        
        for seq_name, frame_list in seq_groups.items():
            total_frames_in_seq = len(frame_list)
            
            valid_end = total_frames_in_seq - max(self.LIDAR_OFFSET, self.RADAR_OFFSET)
            if valid_end <= 0:
                self.logger.warning(
                    f"Sequence {seq_name} has insufficient frames (total {total_frames_in_seq}, offset {max(self.LIDAR_OFFSET, self.RADAR_OFFSET)}), skipping this sequence")
                continue
            
            base_path = os.path.join(self.root_path, seq_name)
            # Use mode parameter to determine sampling interval
            for i in range(0, valid_end, self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                curr_frame = frame_list[i]
                
                lidar_frame = frame_list[i + self.LIDAR_OFFSET]
                radar_frame = frame_list[i + self.RADAR_OFFSET]
                
                im_paths = {}
                for mode_name in self.modalities:
                    image_dir = self.im_path_names[mode_name]
                    image_path = os.path.join(base_path, image_dir, curr_frame)
                    if not os.path.exists(image_path):
                        self.logger.warning(f"Sequence {seq_name} image not found: {image_path}")
                    im_paths[mode_name] = image_path
                
                label_paths = {}
                for mode_name in self.modalities:
                    label_dir = f'boxes_{mode_name}'
                    label_path = os.path.join(base_path, label_dir, curr_frame.replace('.png', '.pkl'))
                    if not os.path.exists(label_path):
                        self.logger.warning(f"Sequence {seq_name} label not found: {label_path}")
                    label_paths[mode_name] = label_path
                
                lidar_path = os.path.join(base_path, 'lidar_1', lidar_frame.replace('.png', '.npy'))
                radar_path = os.path.join(base_path, 'radar_1', radar_frame.replace('.png', '.npy'))
                
                im_info_path = os.path.join(base_path, 'im_info.pkl')
                lidar_radar_info_path = os.path.join(base_path, 'lidar_radar_info.pkl')
                if not os.path.exists(lidar_radar_info_path):
                    self.logger.warning(f"Sequence {seq_name} extrinsic file not found: {lidar_radar_info_path}")
                    lidar_extrinsic = np.eye(4)
                    radar_extrinsic = np.eye(4)
                else:
                    with open(lidar_radar_info_path, 'rb') as f:
                        lidar_radar_info = pickle.load(f)
                    lidar_extrinsic = np.array(lidar_radar_info['lidars'][0]['extrinsic'], dtype=np.float32)
                    radar_extrinsic = np.array(lidar_radar_info['radars'][0]['extrinsic'], dtype=np.float32)
                
                data_info = {
                    'im_paths': im_paths,
                    'label_paths': label_paths,
                    'lidar_path': lidar_path,
                    'radar_path': radar_path,
                    'im_info_path': im_info_path,
                    'lidar_extrinsic': lidar_extrinsic,
                    'radar_extrinsic': radar_extrinsic,
                    'seq_id': seq_name,
                    'frame_id': curr_frame,
                    'cls_name': self.sorted_namelist
                }
                
                CARLA_infos.append(data_info)
        
        self.infos = CARLA_infos
    
    def _build_seq_list_all_maps(self, split_ratio=0.8):
        all_samples = set()
        map_dirs = sorted([
            d for d in os.listdir(self.root_path)
            if os.path.isdir(os.path.join(self.root_path, d))
        ])
        
        for map_name in map_dirs:
            carla_data_root = os.path.join(self.root_path, map_name, "carla_data")
            if not os.path.exists(carla_data_root):
                continue
            seq_dirs = sorted(glob(os.path.join(carla_data_root, "0000*")))
            for seq_path in seq_dirs:
                seq_id = os.path.basename(seq_path)
                if seq_id not in self.dataset_cfg.get('SEQ_IDS'):
                    continue
                weather_dirs = [
                    w for w in os.listdir(seq_path)
                    if os.path.isdir(os.path.join(seq_path, w))
                ]
                for weather_name in weather_dirs:
                    if weather_name not in self.dataset_cfg.get('WEATHER_NAMES'):
                        continue
                    weather_path = os.path.join(seq_path, weather_name)
                    drone_dirs = [
                        d for d in os.listdir(weather_path)
                        if os.path.isdir(os.path.join(weather_path, d))
                    ]
                    for drone_name in drone_dirs:
                        full_path = os.path.join(map_name, 'carla_data', seq_id, weather_name, drone_name)
                        all_samples.add(full_path)
        
        all_samples = sorted(list(all_samples))
        self.logger.info(f"Total sequences: {len(all_samples)}")
        
        split_idx = int(len(all_samples) * split_ratio)
        
        train_list = all_samples[:split_idx]
        val_list = all_samples[split_idx:]
        random.shuffle(train_list)
        random.shuffle(val_list)
        
        if self.training:
            self.logger.info(f"Train Sequences: {len(train_list)}")
            return train_list
        else:
            self.logger.info(f"Test Sequences: {len(val_list)}")
            return val_list

    def set_split(self, split_ratio):
        super(LidarBasedFusionDataset, self).__init__(
            dataset_cfg=self.dataset_cfg,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger
        )

        self.sample_scene_list = []

        self.seq_list = self._build_seq_list_all_maps(split_ratio=split_ratio)

        for seq_name in self.seq_list:
            self.root_path = os.path.normpath(str(self.root_path))
            seq_path = os.path.join(str(self.root_path), seq_name, 'images_rgb')
            if not os.path.exists(seq_path):
                self.logger.warning(f"No image path: {seq_path}")
                continue

            all_frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame in all_frames:
                self.sample_scene_list.append([seq_name, frame])

        self.infos = []
        self.include_CARLA_data(self.mode)

    def collate_batch(self, batch_list, _unused=False):
        # 过滤掉有问题的样本（precomputed_correspondences为None或无效的样本）
        valid_samples = []
        for sample in batch_list:
            corr = sample.get('precomputed_correspondences')
            if corr is not None:
                try:
                    corr_array = np.array(corr, dtype=np.float32)
                    # 检查是否为空数组或维度不匹配
                    if corr_array.size > 0 and corr_array.ndim == 2 and corr_array.shape[1] == 4:
                        valid_samples.append(sample)
                except:
                    # 如果转换为数组失败，也跳过该样本
                    continue
        
        # 如果过滤后没有有效样本，返回一个空的batch或抛出异常
        if len(valid_samples) == 0:
            print("Warning: No valid samples in batch, returning empty batch")
            # 创建一个最小的有效batch
            ret = {'batch_size': 0, 'sorted_namelist': self.dataset_cfg.CLASS_NAMES}
            return ret
        
        # 对有效样本进行batch处理
        data_dict = defaultdict(list)
        for i, cur_sample in enumerate(valid_samples):
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        
        batch_size = len(valid_samples)
        ret = {}
        
        for key, val in data_dict.items():
            try:
                if key == 'image':
                    merged = np.stack(val, axis=0)  # (B, 2, 3, H, W)
                    ret[key] = merged
                elif key == 'gt_boxes':
                    max_gt = max([v.shape[0] for v in val])
                    merged = np.full((batch_size, max_gt, 9, 3), np.nan, dtype=np.float32)
                    for i in range(batch_size):
                        cur_len = val[i].shape[0]
                        merged[i, :cur_len] = val[i]
                    ret[key] = merged
                elif key == 'gt_names':
                    ret[key] = val
                elif key in ['rgb_intrinsic', 'ir_intrinsic', 'rgb_extrinsic', 'ir_extrinsic']:
                    stacked = np.stack(val, axis=0)
                    ret[key] = stacked
                elif key == 'distortion':
                    ret[key] = val
                elif key == 'new_im_size':
                    # 堆叠图像尺寸参数
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'raw_im_size':
                    # 堆叠原始图像尺寸参数
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'obj_size':
                    # 堆叠目标尺寸参数
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'lidar_extrinsic':
                    # 堆叠LiDAR外参
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'radar_extrinsic':
                    # 堆叠Radar外参
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'seq_id':
                    # 保持序列ID原样
                    ret[key] = val
                elif key == 'frame_id':
                    # 保持帧ID原样
                    ret[key] = val
                elif key == 'stride':
                    # 堆叠步长参数
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'intrinsic':
                    # 堆叠内参参数
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'extrinsic':
                    # 堆叠外参参数
                    if len(val) > 0:
                        ret[key] = np.stack(val, axis=0)
                elif key == 'lidar_points':
                    # 通过0点补齐的方式让一个batch内的lidar点云样本具有相同大小
                    if len(val) > 0 and all(isinstance(v, np.ndarray) for v in val):
                        # 找到batch中点云数量的最大值
                        max_points = max([v.shape[0] for v in val])
                        batch_size = len(val)
                        # 只保留点云数据的x, y, z三个维度
                        point_dims = 3
                        # 创建一个形状为(B, N, 3)的数组，初始化为0
                        merged = np.zeros((batch_size, max_points, point_dims), dtype=np.float32)
                        # 将每个样本的点云数据填充到merged数组中，只保留前三个维度
                        for i in range(batch_size):
                            num_points = val[i].shape[0]
                            if num_points > 0:
                                # 只取前三个维度（x, y, z）
                                merged[i, :num_points] = val[i][:, :3]
                        ret[key] = merged
                    else:
                        print(f"Warning: Invalid lidar_points format. Expected list of numpy arrays, got {type(val[0])}")
                        ret[key] = val
                elif key in ['rgb_radar_projection', 'rgb_lidar_projection', 'ir_radar_projection', 'ir_lidar_projection']:
                    # 处理投影数据
                    if len(val) > 0 and all(isinstance(v, dict) or v is None for v in val):
                        # 对于每个投影类型，收集所有有效样本的各个字段
                        batch_projections = []
                        for proj in val:
                            if proj is None:
                                # 如果投影数据为空，创建空字典
                                batch_projections.append(None)
                            else:
                                # 保留原始字典结构
                                batch_projections.append(proj)
                        ret[key] = batch_projections
                    else:
                        ret[key] = val
                elif key == 'precomputed_correspondences':
                    # 对precomputed_correspondences进行填充处理
                    max_corr_length = 0
                    # 找出批次中最大的对应点数量
                    for corr in val:
                        corr_length = len(corr)
                        if corr_length > max_corr_length:
                            max_corr_length = corr_length
                    
                    # 对每个样本的precomputed_correspondences进行填充
                    padded_correspondences = []
                    for corr in val:
                        corr_array = np.array(corr, dtype=np.float32)
                        # 由于前面已经过滤了无效样本，这里可以直接处理
                        
                        # 如果数量不足，用0填充到最大长度
                        if len(corr_array) < max_corr_length:
                            padding = np.zeros((max_corr_length - len(corr_array), 4), dtype=np.float32)
                            padded = np.vstack((corr_array, padding))
                        else:
                            padded = corr_array[:max_corr_length]  # 如果超过最大长度，截断
                        padded_correspondences.append(padded)
                    
                    # 堆叠所有填充后的对应点
                    ret[key] = np.stack(padded_correspondences, axis=0)
                elif key in ['hm', 'center_res', 'center_dis', 'dim', 'rot']:
                    # 处理热图和其他预测相关的键
                    try:
                        if key == 'hm':
                            val = [v.squeeze(0) if v.ndim == 4 else v for v in val]
                            max_cls = max([v.shape[0] for v in val])
                            padded_val = []
                            for v in val:
                                if v.shape[0] < max_cls:
                                    pad_width = ((0, max_cls - v.shape[0]), (0, 0), (0, 0))
                                    v = np.pad(v, pad_width, mode='constant')
                                padded_val.append(v)
                            stacked = np.stack(padded_val, axis=0)
                        else:
                            val = [v.squeeze(0) if v.ndim == 4 else v for v in val]
                            stacked = np.stack(val, axis=0)
                        
                        ret[key] = stacked
                    except Exception as e:
                        raise e
            except Exception as e:
                print(f"Error in collate_batch for key={key}: {e}")
                raise e
        
        # 更新返回的batch_size为有效样本数量
        ret['batch_size'] = batch_size
        ret['sorted_namelist'] = self.dataset_cfg.CLASS_NAMES
        
        return ret

    def name_from_code(self, name_indices):
        if len(self.dataset_cfg.CLASS_NAMES) == 1:
            return np.array([self.dataset_cfg.CLASS_NAMES[0]] * len(name_indices))
        else:
            names = [self.dataset_cfg.CLASS_NAMES[int(x)] for x in name_indices]
            return np.array(names)

    def generate_prediction_dicts(self, batch_dict, output_path):
        # 直接使用LAAM6D_Det_Dataset的方法，保持一致性
        batch_size = batch_dict['batch_size']
        annos = []

        base_debug_dir = os.path.join(os.getcwd(), 'debug_images_fusion')
        os.makedirs(base_debug_dir, exist_ok=True)
        
        # 仅在本地调试时启用显示（通过环境变量控制，默认关闭:export ENABLE_IMSHOW=true）
        enable_imshow = os.environ.get('ENABLE_IMSHOW', 'False').lower() == 'true'
        
        for batch_id in range(batch_size):
            seq_id = batch_dict['seq_id'][batch_id]
            frame_id = batch_dict['frame_id'][batch_id]
            
            raw_im_size = batch_dict['raw_im_size'][batch_id]
            obj_size = batch_dict['obj_size'][batch_id]
            intrinsic = batch_dict['intrinsic'][batch_id]
            extrinsic = batch_dict['extrinsic'][batch_id]
            distortion = batch_dict['distortion'][batch_id]

            gt_boxes = batch_dict['gt_boxes'][batch_id]
            gt_names = batch_dict['gt_names'][batch_id]

            # 打印尺寸信息，查看是否一致
            # gt_boxes = convert_9points_to_9params(np.array(gt_boxes.cpu()))
            # print(f"gt_size: {gt_boxes9d[:, 3:6]}, pred_size: {batch_dict['pred_boxes9d'][batch_id][:, 3:6]}")
            
            # [center_x, center_y, center_z, l, w, h, a1, a2, a3, class_id]
            pred_boxes9d = batch_dict['pred_boxes9d'][batch_id]
            pred_class_ids = pred_boxes9d[:, -1].astype(int)
            pred_names = self.name_from_code(pred_class_ids)
            pred_boxes9d = convert_9params_to_9points(pred_boxes9d[:, :-1])
            
            confidence = batch_dict['confidence'][batch_id]
            sorted_namelist = batch_dict['sorted_namelist']
            frame_dict = {
                'seq_id': seq_id,
                'frame_id': frame_id,
                'obj_size': obj_size,
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'pred_boxes': pred_boxes9d,
                'pred_names': pred_names,
                'confidence': confidence,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'distortion': distortion,
            }
            
            annos.append(frame_dict)
            
            # ====================================prediction_visualizations=============================================
            # # 确保图像路径包含扩展名（假设为png，可根据实际情况调整）
            # if not any(frame_id.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            #     frame_id_with_ext = f"{frame_id}.png"
            # else:
            #     frame_id_with_ext = frame_id
            # im_path = os.path.join(self.root_path, seq_id, 'images_rgb', frame_id_with_ext)
            # frame_dict['im_path'] = im_path
            
            # image = cv2.imread(im_path)
            # if image is None:
            #     print(f"警告：无法读取图像 {im_path}，跳过该帧")
            #     continue
            
            # # 原始(网络输入/训练时使用)的目标大小
            # orig_w, orig_h = self.new_im_width, self.new_im_hight
            
            # # 实际要画框的读取图像大小 (h, w)
            # tgt_h, tgt_w = image.shape[:2]
            
            # # 跳过空图像（异常处理）
            # if tgt_h == 0 or tgt_w == 0:
            #     print(f"警告：图像 {im_path} 尺寸异常，跳过该帧")
            #     continue
            
            # # 计算缩放因子（确保不为0，避免除零错误）
            # scale_x = tgt_w / float(orig_w) if orig_w != 0 else 1.0
            # scale_y = tgt_h / float(orig_h) if orig_h != 0 else 1.0
            
            # # 调整内参以匹配实际图像尺寸
            # resized_intrinsics_rgb = intrinsic.copy()
            # resized_intrinsics_rgb[0, 0] *= scale_x  # fx
            # resized_intrinsics_rgb[1, 1] *= scale_y  # fy
            # resized_intrinsics_rgb[0, 2] *= scale_x  # cx
            # resized_intrinsics_rgb[1, 2] *= scale_y  # cy
            
            # # 无预测框且无GT框时，可跳过绘制（可选优化）
            # has_pred = len(pred_boxes9d) > 0
            # has_gt = 'gt_boxes' in batch_dict and len(batch_dict['gt_boxes'][batch_id]) > 0
            # if not has_pred and not has_gt:
            #     print(f"帧 {frame_id} 无预测框和GT框，跳过绘制")
            #     annos.append(frame_dict)
            #     continue
            
            # # 绘制预测框
            # image = draw_box9d_on_image(
            #     pred_boxes9d, 
            #     image,
            #     img_width=tgt_w,  # 使用实际图像宽度
            #     img_height=tgt_h,  # 使用实际图像高度
            #     color=(255, 0, 0),
            #     intrinsic_mat=resized_intrinsics_rgb,
            #     extrinsic_mat=extrinsic,
            #     distortion_matrix=distortion
            # )
            
            # # 绘制GT框（如果存在）
            # if 'gt_boxes' in batch_dict:
            #     gt_box9d = batch_dict['gt_boxes'][batch_id]  # (N, 9, 3)
            #     gt_names = batch_dict['gt_names'][batch_id]  # (N,)
            #     frame_dict['gt_boxes'] = gt_box9d
            #     frame_dict['gt_names'] = gt_names
            
            #     image = draw_box9d_on_image_gt(
            #         gt_box9d, 
            #         image,
            #         img_width=tgt_w,  # 使用实际图像宽度
            #         img_height=tgt_h,  # 使用实际图像高度
            #         color=(0, 0, 255),
            #         intrinsic_mat=resized_intrinsics_rgb,
            #         extrinsic_mat=extrinsic,
            #         distortion_matrix=distortion
            #     )
            
            # # 处理安全的文件名和路径（替换特殊字符）
            # safe_frame_id = frame_id_with_ext.replace('.', '_').replace('=', '_')
            # safe_seq_id = seq_id.replace('/', '_').replace('\\', '_')  # 移除路径分隔符
            # save_filename = f"batch_{batch_id}_frame_{safe_frame_id}.jpg"
            # save_rel_path = os.path.join(safe_seq_id, save_filename)
            # save_path = os.path.join(base_debug_dir, save_rel_path)
            
            # # 确保保存目录存在
            # save_dir = os.path.dirname(save_path)
            # os.makedirs(save_dir, exist_ok=True)
            
            # # 保存图像
            # try:
            #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     pil_image = Image.fromarray(image_rgb)
            #     pil_image.save(save_path)
            #     print(f"已保存调试图像: {save_path}")
            # except Exception as e:
            #     print(f"保存图像失败: {e}，路径: {save_path}")
            
            # # 可选：本地调试时显示图像（服务器环境自动禁用）
            # if enable_imshow:
            #     cv2.imshow('rgb_image', image)
            #     cv2.waitKey(1)
            # ==========================================================================================================

        
        return annos
    
    def evaluation(self, annos, metric_root_path):
        print('evaluating !!!')
        
        i_str = self.evaluation_all(annos, metric_root_path)
        
        return i_str
    
    def evaluation_all(self, annos, metric_root_path):
        
        laa_ads_fun = LAA3D_ADS_Metric(eval_config=self.dataset_cfg.LAA3D_ADS_METRIC,
                                       classes=self.sorted_namelist,
                                       metric_save_path=metric_root_path)
        
        final_str = laa_ads_fun.eval(annos)
        
        return final_str
        
    def project_radar_to_image(self, radar_data, intrinsic, extrinsic, radar_extrinsic=None):
        # 将雷达数据投影到图像上
        # radar_data: (N, 5) [velocity, azim, alt, depth, is_drone] 极坐标格式
        # intrinsic: 相机内参 (3, 3)
        # extrinsic: 相机外参 (4, 4)
        # radar_extrinsic: 雷达外参 (4, 4)

        radar_velocity_heatmap = np.zeros((3, self.new_im_hight, self.new_im_width), dtype=np.float32)
        
        if radar_data is None or len(radar_data) == 0:
            return {
                'image_points': np.zeros((0, 2), dtype=np.float32),
                'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
                'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
                'velocity': np.zeros((0,), dtype=np.float32),
                'radar_xyz': np.zeros((0, 3), dtype=np.float32),
                'radar_velocity_heatmap': radar_velocity_heatmap
            }
        
        # 解析雷达数据（极坐标格式）
        velocity = radar_data[:, 0]
        azim = radar_data[:, 1]
        alt = radar_data[:, 2]
        depth = radar_data[:, 3]
        
        # 极坐标 → 笛卡尔坐标（雷达坐标系）
        x = depth * np.cos(alt) * np.cos(azim)
        y = depth * np.cos(alt) * np.sin(azim)
        z = depth * np.sin(alt)
        radar_xyz = np.stack([x, y, z], axis=1)
        
        # 雷达坐标系 → 世界坐标系
        radar_homo = np.hstack([radar_xyz, np.ones((radar_xyz.shape[0], 1))])
        # 如果没有提供雷达外参，使用单位矩阵
        if radar_extrinsic is None:
            radar_extrinsic = np.eye(4)
        pts_world = (radar_extrinsic @ radar_homo.T).T[:, :3]
        
        # 世界坐标系 → 相机坐标系（CARLA系）
        world_to_camera = np.linalg.inv(extrinsic) if extrinsic.shape == (4, 4) else np.eye(4)
        world_homo = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
        camera_points_carla = (world_to_camera @ world_homo.T).T[:, :3]  # 只取前3列
        
        # CARLA坐标系到OpenCV坐标系的转换矩阵
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 转换到OpenCV坐标系
        camera_points_carla_hom = np.hstack([camera_points_carla, np.ones((len(camera_points_carla), 1))])
        camera_points_opencv = (carla_to_opencv @ camera_points_carla_hom.T).T[:, :3]
        
        # 过滤OpenCV系前方的点（Z>0，确保投影有效）
        valid_z_indices = camera_points_opencv[:, 2] > 0
        if not np.any(valid_z_indices):
            return {
                'image_points': np.zeros((0, 2), dtype=np.float32),
                'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
                'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
                'velocity': np.zeros((0,), dtype=np.float32),
                'radar_xyz': np.zeros((0, 3), dtype=np.float32),
                'radar_velocity_heatmap': radar_velocity_heatmap
            }
        
        # 只对有效点进行后续处理
        valid_camera_points_opencv = camera_points_opencv[valid_z_indices]
        valid_velocity = velocity[valid_z_indices]
        valid_camera_points_carla = camera_points_carla[valid_z_indices]
        valid_radar_xyz = radar_xyz[valid_z_indices]
        
        # 使用cv2.projectPoints进行投影，处理畸变
        # 假设没有畸变参数，传入None
        image_points, _ = cv2.projectPoints(
            valid_camera_points_opencv,
            np.zeros(3),  # 旋转向量
            np.zeros(3),  # 平移向量
            intrinsic,    # 内参矩阵
            None          # 畸变参数
        )
        
        # 重塑投影结果并确保是二维数组
        image_points = image_points.squeeze().astype(np.float32)
        # 处理只有一个点的情况，确保数组是二维的
        if len(image_points.shape) == 1:
            image_points = np.expand_dims(image_points, axis=0)
        
        # 只保留在图像范围内的点
        valid_indices = (image_points[:, 0] >= 0) & (image_points[:, 0] < self.new_im_width) & \
                       (image_points[:, 1] >= 0) & (image_points[:, 1] < self.new_im_hight)
        
        # 检查是否有在图像范围内的点
        if not np.any(valid_indices):
            return {
                'image_points': np.zeros((0, 2), dtype=np.float32),
                'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
                'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
                'velocity': np.zeros((0,), dtype=np.float32),
                'radar_xyz': np.zeros((0, 3), dtype=np.float32),
                'radar_velocity_heatmap': radar_velocity_heatmap
            }
        
        # 向量化处理：批量填充速度热力图
        if np.any(valid_indices):
            # 获取有效的图像点和速度值
            valid_img_pts = image_points[valid_indices]
            valid_vels = valid_velocity[valid_indices]
            
            # 向量化转换为整数坐标（四舍五入）
            y_coords = np.round(valid_img_pts[:, 1]).astype(int)
            x_coords = np.round(valid_img_pts[:, 0]).astype(int)
            
            # 向量化过滤出在图像范围内的坐标
            in_bounds = (x_coords >= 0) & (x_coords < self.new_im_width) & \
                       (y_coords >= 0) & (y_coords < self.new_im_hight)
            
            if np.any(in_bounds):
                # 获取有效的索引和坐标
                valid_idx = np.where(in_bounds)[0]
                valid_y = y_coords[valid_idx]
                valid_x = x_coords[valid_idx]
                valid_vel_values = valid_vels[valid_idx].astype(float)
                
                # 向量化填充radar_velocity_heatmap的三个通道
                radar_velocity_heatmap[:, valid_y, valid_x] = valid_vel_values
        
        return {
            'image_points': image_points[valid_indices],
            'camera_points_opencv': valid_camera_points_opencv[valid_indices],
            'camera_points_carla': valid_camera_points_carla[valid_indices],
            'velocity': valid_velocity[valid_indices],
            'radar_xyz': valid_radar_xyz[valid_indices],
            'radar_velocity_heatmap': radar_velocity_heatmap
        }
        
    # def project_lidar_to_image(self, lidar_data, intrinsic, extrinsic, lidar_extrinsic=None):
    #     # 将LiDAR数据投影到图像上
    #     # lidar_data: (N, 4) [x, y, z, intensity]
    #     # intrinsic: 相机内参 (3, 3)
    #     # extrinsic: 相机外参 (4, 4)
    #     # lidar_extrinsic: LiDAR外参 (4, 4)，将LiDAR坐标系转换到世界坐标系
        
    #     # 创建原始的世界坐标映射（保留作为备选）
    #     world_coords_map = np.zeros((3, self.new_im_hight, self.new_im_width), dtype=np.float32)
    #     # 创建三通道depth图像，每层通道都包含相同的深度值
    #     depth_map = np.zeros((3, self.new_im_hight, self.new_im_width), dtype=np.float32)

    #     if lidar_data is None or len(lidar_data) == 0:
    #         return {
    #             'image_points': np.zeros((0, 2), dtype=np.float32),
    #             'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
    #             'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
    #             'lidar_data': np.zeros((0, 4), dtype=np.float32),
    #             'world_coords_data': np.zeros((0, 4), dtype=np.float32),
    #             'world_coords_map': world_coords_map,  # 保持原始输出
    #             'depth_map': depth_map  # 新增depth_map输出
    #         }
        
    #     # 提取LiDAR点的3D坐标
    #     lidar_points = lidar_data[:, :3]
        
    #     # CARLA坐标系到OpenCV坐标系的转换矩阵
    #     carla_to_opencv = np.array([
    #         [0, 1, 0, 0],
    #         [0, 0, -1, 0],
    #         [1, 0, 0, 0],
    #         [0, 0, 0, 1]
    #     ], dtype=np.float32)
        
    #     # 创建4xN的齐次坐标
    #     ones = np.ones((lidar_points.shape[0], 1))
    #     lidar_points_hom = np.hstack((lidar_points, ones))
        
    #     # 如果提供了lidar_extrinsic，则将点从lidar坐标系转换到世界坐标系
    #     if lidar_extrinsic is not None:
    #         world_points_hom = lidar_extrinsic @ lidar_points_hom.T
    #         world_points_hom = world_points_hom.T
    #     else:
    #         # 如果没有提供lidar_extrinsic，假设点已经在世界坐标系中
    #         world_points_hom = lidar_points_hom
        
    #     # 相机到世界的变换矩阵的逆
    #     world_to_camera = np.linalg.inv(extrinsic) if extrinsic.shape == (4, 4) else np.eye(4)
        
    #     # 转换到相机坐标系（CARLA系）
    #     camera_points_carla = world_to_camera @ world_points_hom.T
    #     camera_points_carla = camera_points_carla.T[:, :3]  # 只取前3列
        
    #     # 转换到OpenCV坐标系
    #     camera_points_carla_hom = np.hstack([camera_points_carla, np.ones((len(camera_points_carla), 1))])
    #     camera_points_opencv = (carla_to_opencv @ camera_points_carla_hom.T).T[:, :3]
        
    #     # 过滤OpenCV系前方的点（Z>0，确保投影有效）
    #     valid_z_indices = camera_points_opencv[:, 2] > 0
    #     if not np.any(valid_z_indices):
    #         return {
    #             'image_points': np.zeros((0, 2), dtype=np.float32),
    #             'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
    #             'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
    #             'lidar_data': np.zeros((0, 4), dtype=np.float32),
    #             'world_coords_data': np.zeros((0, 4), dtype=np.float32),
    #             'world_coords_map': world_coords_map,
    #             'depth_map': depth_map
    #         }
        
    #     # 只对有效点进行后续处理
    #     valid_camera_points_opencv = camera_points_opencv[valid_z_indices]
    #     valid_lidar_data = lidar_data[valid_z_indices]
    #     valid_camera_points_carla = camera_points_carla[valid_z_indices]
        
    #     # 使用cv2.projectPoints进行投影，处理畸变
    #     # 假设没有畸变参数，传入None
    #     image_points, _ = cv2.projectPoints(
    #         valid_camera_points_opencv,
    #         np.zeros(3),  # 旋转向量
    #         np.zeros(3),  # 平移向量
    #         intrinsic,    # 内参矩阵
    #         None          # 畸变参数
    #     )
        
    #     # 重塑投影结果
    #     image_points = image_points.squeeze().astype(np.float32)
    #     # 处理只有一个点的情况，确保数组是二维的
    #     if len(image_points.shape) == 1:
    #         image_points = np.expand_dims(image_points, axis=0)
        
    #     # 只保留在图像范围内的点
    #     valid_indices = (image_points[:, 0] >= 0) & (image_points[:, 0] < self.new_im_width) & \
    #                    (image_points[:, 1] >= 0) & (image_points[:, 1] < self.new_im_hight)
        
    #     # 检查是否有在图像范围内的点
    #     if not np.any(valid_indices):
    #         return {
    #             'image_points': np.zeros((0, 2), dtype=np.float32),
    #             'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
    #             'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
    #             'lidar_data': np.zeros((0, 4), dtype=np.float32),
    #             'world_coords_data': np.zeros((0, 4), dtype=np.float32),
    #             'world_coords_map': world_coords_map,
    #             'depth_map': depth_map
    #         }
        
    #     # 从world_points_hom中提取世界坐标
    #     valid_world_points = world_points_hom[valid_z_indices][valid_indices][:, :3]
        
    #     # 创建世界坐标系下的点云数据（保持与lidar_data相同的格式，但使用世界坐标）
    #     world_coords_data = np.zeros_like(valid_lidar_data[valid_indices])
    #     world_coords_data[:, :3] = valid_world_points  # 用世界坐标替换原始坐标
    #     world_coords_data[:, 3:] = valid_lidar_data[valid_indices][:, 3:]  # 保留其他属性（如intensity）
        
    #     # 使用向量化操作优化world_coords_map和depth_map的填充
    #     # 获取有效的投影点
    #     valid_img_points = image_points[valid_indices]
    #     # 转换为整数坐标（四舍五入）
    #     y_coords = np.round(valid_img_points[:, 1]).astype(int)
    #     x_coords = np.round(valid_img_points[:, 0]).astype(int)
        
    #     # 过滤出在图像范围内的坐标
    #     in_bounds = (x_coords >= 0) & (x_coords < self.new_im_width) & \
    #                (y_coords >= 0) & (y_coords < self.new_im_hight)
        
    #     if np.any(in_bounds):
    #         # 获取有效的索引和坐标
    #         valid_idx = np.where(in_bounds)[0]
    #         valid_y = y_coords[valid_idx]
    #         valid_x = x_coords[valid_idx]
            
    #         # 获取对应的世界坐标
    #         valid_world = valid_world_points[valid_idx]
            
    #         # 先使用valid_indices筛选，再使用valid_idx进一步筛选
    #         valid_depths = valid_camera_points_opencv[valid_indices][valid_idx, 2]
            
    #         # 向量化填充world_coords_map
    #         world_coords_map[0, valid_y, valid_x] = valid_world[:, 0]
    #         world_coords_map[1, valid_y, valid_x] = valid_world[:, 1]
    #         world_coords_map[2, valid_y, valid_x] = valid_world[:, 2]
            
    #         # 向量化填充depth_map的三个通道
    #         depth_map[:, valid_y, valid_x] = valid_depths 
        
    #     return {
    #         'image_points': image_points[valid_indices],
    #         'camera_points_opencv': valid_camera_points_opencv[valid_indices],
    #         'camera_points_carla': valid_camera_points_carla[valid_indices],
    #         'lidar_data': valid_lidar_data[valid_indices],
    #         'world_coords_data': world_coords_data,
    #         'world_coords_map': world_coords_map,  # 保留原始输出
    #         'depth_map': depth_map  # 新增depth_map输出
    #     }

    def project_lidar_to_image(self, lidar_data, intrinsic, extrinsic, lidar_extrinsic=None):
        # 将LiDAR数据投影到图像上
        # lidar_data: (N, 4) [x, y, z, intensity]
        # intrinsic: 相机内参 (3, 3)
        # extrinsic: 相机外参 (4, 4)
        # lidar_extrinsic: LiDAR外参 (4, 4)，将LiDAR坐标系转换到世界坐标系

        # 创建原始的世界坐标映射（保留作为备选）
        world_coords_map = np.zeros((3, self.new_im_hight, self.new_im_width), dtype=np.float32)

        # ✅ 改：内部用单通道 z-buffer 存最近深度（inf表示无效）
        depth_zbuf = np.full((self.new_im_hight, self.new_im_width), np.inf, dtype=np.float32)
        # 最终仍输出三通道depth图像（与你原接口一致）
        depth_map = np.zeros((3, self.new_im_hight, self.new_im_width), dtype=np.float32)

        if lidar_data is None or len(lidar_data) == 0:
            return {
                'image_points': np.zeros((0, 2), dtype=np.float32),
                'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
                'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
                'lidar_data': np.zeros((0, 4), dtype=np.float32),
                'world_coords_data': np.zeros((0, 4), dtype=np.float32),
                'world_coords_map': world_coords_map,
                'depth_map': depth_map
            }

        # 提取LiDAR点的3D坐标
        lidar_points = lidar_data[:, :3]

        # CARLA坐标系到OpenCV坐标系的转换矩阵
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 创建4xN的齐次坐标
        ones = np.ones((lidar_points.shape[0], 1), dtype=np.float32)
        lidar_points_hom = np.hstack((lidar_points.astype(np.float32), ones))

        # 如果提供了lidar_extrinsic，则将点从lidar坐标系转换到世界坐标系
        if lidar_extrinsic is not None:
            world_points_hom = (lidar_extrinsic @ lidar_points_hom.T).T
        else:
            # 如果没有提供lidar_extrinsic，假设点已经在世界坐标系中
            world_points_hom = lidar_points_hom

        # 相机到世界的变换矩阵的逆
        world_to_camera = np.linalg.inv(extrinsic) if extrinsic.shape == (4, 4) else np.eye(4, dtype=np.float32)

        # 转换到相机坐标系（CARLA系）
        camera_points_carla = (world_to_camera @ world_points_hom.T).T[:, :3]

        # 转换到OpenCV坐标系
        camera_points_carla_hom = np.hstack([camera_points_carla, np.ones((len(camera_points_carla), 1), dtype=np.float32)])
        camera_points_opencv = (carla_to_opencv @ camera_points_carla_hom.T).T[:, :3]

        # 过滤OpenCV系前方的点（Z>0，确保投影有效）
        valid_z_indices = camera_points_opencv[:, 2] > 0
        if not np.any(valid_z_indices):
            return {
                'image_points': np.zeros((0, 2), dtype=np.float32),
                'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
                'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
                'lidar_data': np.zeros((0, 4), dtype=np.float32),
                'world_coords_data': np.zeros((0, 4), dtype=np.float32),
                'world_coords_map': world_coords_map,
                'depth_map': depth_map
            }

        # 只对有效点进行后续处理
        valid_camera_points_opencv = camera_points_opencv[valid_z_indices]
        valid_lidar_data = lidar_data[valid_z_indices]
        valid_camera_points_carla = camera_points_carla[valid_z_indices]

        # 使用cv2.projectPoints进行投影（仿真无畸变：distCoeffs=None）
        image_points, _ = cv2.projectPoints(
            valid_camera_points_opencv,
            np.zeros(3, dtype=np.float32),  # 旋转向量
            np.zeros(3, dtype=np.float32),  # 平移向量
            intrinsic.astype(np.float32),   # 内参矩阵
            None                            # 畸变参数
        )

        # 重塑投影结果
        image_points = image_points.squeeze().astype(np.float32)
        if len(image_points.shape) == 1:
            image_points = np.expand_dims(image_points, axis=0)

        # 只保留在图像范围内的点（浮点像素）
        valid_indices = (
            (image_points[:, 0] >= 0) & (image_points[:, 0] < self.new_im_width) &
            (image_points[:, 1] >= 0) & (image_points[:, 1] < self.new_im_hight)
        )

        if not np.any(valid_indices):
            return {
                'image_points': np.zeros((0, 2), dtype=np.float32),
                'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
                'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
                'lidar_data': np.zeros((0, 4), dtype=np.float32),
                'world_coords_data': np.zeros((0, 4), dtype=np.float32),
                'world_coords_map': world_coords_map,
                'depth_map': depth_map
            }

        # 从world_points_hom中提取世界坐标（先按Z过滤，再按图像范围过滤）
        valid_world_points = world_points_hom[valid_z_indices][valid_indices][:, :3]

        # 创建世界坐标系下的点云数据（保持与lidar_data相同的格式）
        world_coords_data = np.zeros_like(valid_lidar_data[valid_indices])
        world_coords_data[:, :3] = valid_world_points
        world_coords_data[:, 3:] = valid_lidar_data[valid_indices][:, 3:]

        # 获取有效的投影点
        valid_img_points = image_points[valid_indices]

        # 转换为整数坐标（四舍五入）
        y_coords = np.round(valid_img_points[:, 1]).astype(int)
        x_coords = np.round(valid_img_points[:, 0]).astype(int)

        # 过滤出在图像范围内的整数坐标
        in_bounds = (
            (x_coords >= 0) & (x_coords < self.new_im_width) &
            (y_coords >= 0) & (y_coords < self.new_im_hight)
        )

        if np.any(in_bounds):
            valid_idx = np.where(in_bounds)[0]
            valid_y = y_coords[valid_idx]
            valid_x = x_coords[valid_idx]

            # 对应世界坐标
            valid_world = valid_world_points[valid_idx].astype(np.float32)

            # 对应深度：OpenCV相机系 Z（最近表面用最小Z）
            valid_depths = valid_camera_points_opencv[valid_indices][valid_idx, 2].astype(np.float32)

            # ✅ 改：z-buffer 更新（同像素多点，保留最小Z）
            for j in range(valid_depths.shape[0]):
                y = int(valid_y[j])
                x = int(valid_x[j])
                z = float(valid_depths[j])

                if z < float(depth_zbuf[y, x]):
                    depth_zbuf[y, x] = z
                    world_coords_map[0, y, x] = float(valid_world[j, 0])
                    world_coords_map[1, y, x] = float(valid_world[j, 1])
                    world_coords_map[2, y, x] = float(valid_world[j, 2])

        # ✅ 输出三通道depth_map：无效像素设为0（与之前 invalid_val=0 的习惯一致）
        mask = np.isfinite(depth_zbuf)
        depth_map[:, mask] = depth_zbuf[mask]
        depth_map[:, ~mask] = 0.0

        return {
            'image_points': image_points[valid_indices],
            'camera_points_opencv': valid_camera_points_opencv[valid_indices],
            'camera_points_carla': valid_camera_points_carla[valid_indices],
            'lidar_data': valid_lidar_data[valid_indices],
            'world_coords_data': world_coords_data,
            'world_coords_map': world_coords_map,
            'depth_map': depth_map
        }


    def _find_corresponding_points(self, rgb_proj_info, ir_proj_info, gt_boxes, rgb_image=None, ir_image=None, visualize=False, save_path=None, ir_gt_boxes=None):
        """
        Find corresponding points on RGB and IR images based on world coordinates from LiDAR point cloud (optimized direct matching version)
        Use world coordinates as unique identifiers to achieve efficient and accurate corresponding point matching
        
        Parameters:
            rgb_proj_info: RGB image projection information generator/list, each element is (u, v, x, y, z)
            ir_proj_info: IR image projection information generator/list, each element is (u, v, x, y, z)
            gt_boxes: List of drone bounding boxes in world coordinate system
            rgb_image: Optional, RGB image data for visualization
            ir_image: Optional, IR image data for visualization
            visualize: Whether to visualize corresponding points
            save_path: Path to save visualization results
            ir_gt_boxes: Optional, ground truth bounding boxes on IR image for visualization
        
        Returns:
            Corresponding point tensor with shape [N, 4], each row is (rgb_u, rgb_v, ir_u, ir_v)
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 1. 将投影信息转换为列表
        rgb_list = list(rgb_proj_info)
        ir_list = list(ir_proj_info)
        
        # 2. 构建RGB世界坐标到图像坐标的映射
        rgb_world_to_pixel = {}
        
        for (u, v, x, y, z) in rgb_list:
            # 使用精确的世界坐标作为键
            rgb_world_to_pixel[(x, y, z)] = (u, v)
        
        # 3. 查找同时在RGB和IR中投影成功的点
        correspondences_list = []
        # 保存对应点的世界坐标，用于后续边界框过滤
        correspondences_world = []
        
        for u_ir, v_ir, x, y, z in ir_list:
            # 检查这个世界坐标点是否也在RGB投影中
            if (x, y, z) in rgb_world_to_pixel:
                u_rgb, v_rgb = rgb_world_to_pixel[(x, y, z)]
                correspondences_list.append([u_rgb, v_rgb, u_ir, v_ir])
                correspondences_world.append([x, y, z])
        
        # 4. 转换为张量并确保形状正确
        if correspondences_list:
            correspondences = torch.tensor(correspondences_list, dtype=torch.float32, device=device)
            # 转换世界坐标为numpy数组，用于边界框过滤
            correspondences_world_np = np.array(correspondences_world, dtype=np.float32)
        else:
            print("WARNING: No matching world coordinate points found")
            print(f"RGB mapping table size: {len(rgb_world_to_pixel)}, IR point count: {len(ir_list)}")
            correspondences = torch.zeros((0, 4), dtype=torch.float32, device=device)
            correspondences_world_np = np.zeros((0, 3), dtype=np.float32)
        
        # 调用可视化函数（如果需要）
        if visualize and rgb_image is not None and ir_image is not None:
            self._visualize_correspondences(rgb_image, ir_image, correspondences, save_path, 
                                          correspondences_world_np=correspondences_world_np, 
                                          gt_boxes=gt_boxes, ir_gt_boxes=ir_gt_boxes)
        
        # 返回所有找到的对应点
        return correspondences
    
    def _fuse_images_at_original(self, rgb_image, ir_image, correspondences):
        """
        在原图层面进行RGB和IR图像融合
        对应位置进行特征拼接，非对应位置补0
        
        Parameters:
            rgb_image: RGB图像数据 (H, W, 3)
            ir_image: IR图像数据 (H, W, 3)
            correspondences: 对应点张量，形状为[N, 4]，每行是(rgb_u, rgb_v, ir_u, ir_v)
        
        Returns:
            fused_image: 融合后的图像数据 (H, W, 6)，前3通道是RGB，后3通道是IR（对应位置填充，非对应位置补0）
        """
        # 获取图像尺寸
        h, w = rgb_image.shape[:2]
        
        # 初始化融合图像，前3通道是RGB，后3通道初始化为0
        fused_image = np.zeros((h, w, 6), dtype=np.float32)
        
        # 复制RGB到融合图像的前3通道
        fused_image[:, :, :3] = rgb_image.astype(np.float32)
        
        # 处理对应点，将IR图像中对应位置的值复制到融合图像的后3通道
        if len(correspondences) > 0:
            # 将对应点转换为numpy数组
            if isinstance(correspondences, list):
                # 如果是列表，直接转换为numpy数组
                corr_np = np.array(correspondences)
            else:
                # 如果是PyTorch张量，先转移到CPU再转换为numpy数组
                corr_np = correspondences.cpu().numpy()
            
            # 确保坐标是整数且在有效范围内
            rgb_u_coords = np.clip(corr_np[:, 0].astype(int), 0, w-1)
            rgb_v_coords = np.clip(corr_np[:, 1].astype(int), 0, h-1)
            ir_u_coords = np.clip(corr_np[:, 2].astype(int), 0, w-1)
            ir_v_coords = np.clip(corr_np[:, 3].astype(int), 0, h-1)
            
            # 将IR图像中对应位置的值复制到融合图像的RGB对应位置的后3通道
            for i in range(len(corr_np)):
                rgb_u, rgb_v = rgb_u_coords[i], rgb_v_coords[i]
                ir_u, ir_v = ir_u_coords[i], ir_v_coords[i]
                
                # 将IR图像中(ir_v, ir_u)位置的值复制到融合图像的(rgb_v, rgb_u)位置的后3通道
                fused_image[rgb_v, rgb_u, 3:] = ir_image[ir_v, ir_u].astype(np.float32)
        
        return fused_image
    
    def _visualize_correspondences(self, rgb_image, ir_image, correspondences, save_path=None, correspondences_world_np=None, gt_boxes=None, ir_gt_boxes=None):
        """
        Visualize corresponding points on RGB and IR images, connect corresponding points with lines, display three images side by side
        This function also filters points within drone bounding boxes when gt_boxes is provided
        
        Parameters:
            rgb_image: RGB image data
            ir_image: IR image data
            correspondences: Corresponding point tensor with shape [N, 4], each row is (rgb_u, rgb_v, ir_u, ir_v)
            save_path: Path to save visualization results
            correspondences_world_np: World coordinates of corresponding points, used for filtering
            gt_boxes: List of drone bounding boxes in world coordinate system, used for filtering
            ir_gt_boxes: Ground truth bounding boxes on IR image, format [N, 9, 3]
        """
        
        # 过滤出位于无人机边界框内的对应点（如果提供了边界框）
        filtered_correspondences = correspondences
        
        if gt_boxes is not None and len(gt_boxes) > 0 and correspondences_world_np is not None:
            # 定义过滤函数：判断点是否在边界框内
            def point_in_boxes(point, boxes):
                # point: [x, y, z]
                # boxes: 边界框列表，每个边界框是[x_min, y_min, z_min, x_max, y_max, z_max]格式
                for box in boxes:
                    x_min, y_min, z_min, x_max, y_max, z_max = box
                    if (x_min <= point[0] <= x_max and 
                        y_min <= point[1] <= y_max and 
                        z_min <= point[2] <= z_max):
                        return True
                return False
            
            # 转换边界框格式：从(N, 9, 3)转换为[N, 6]的[x_min,y_min,z_min,x_max,y_max,z_max]格式
            def convert_boxes_format(boxes):
                converted_boxes = []
                for box in boxes:
                    # box是(9, 3)格式：[中心点, 8个顶点]
                    # 计算边界框的最小和最大坐标
                    x_min = np.min(box[:, 0])
                    y_min = np.min(box[:, 1])
                    z_min = np.min(box[:, 2])
                    x_max = np.max(box[:, 0])
                    y_max = np.max(box[:, 1])
                    z_max = np.max(box[:, 2])
                    converted_boxes.append([x_min, y_min, z_min, x_max, y_max, z_max])
                return converted_boxes
            
            # 转换边界框格式
            converted_gt_boxes = convert_boxes_format(gt_boxes)
            
            # 快速过滤：直接检查每个对应点的世界坐标是否在边界框内
            valid_indices = []
            for i, world_point in enumerate(correspondences_world_np):
                if point_in_boxes(world_point, converted_gt_boxes):
                    valid_indices.append(i)
            
            # 重置filtered_correspondences变量，避免之前调用的影响
            filtered_correspondences = None
            if valid_indices:
                # 确保valid_indices是唯一的
                valid_indices = list(set(valid_indices))
                filtered_correspondences = correspondences[valid_indices]
                print(f"Number of drone correspondences retained after filtering: {len(filtered_correspondences)}")
            else:
                print("WARNING: No corresponding points found within drone bounding boxes")
        else:
            # 如果没有边界框但需要可视化，直接可视化所有对应点
            print(f"No bounding boxes provided, visualizing all {len(correspondences)} corresponding points")

        
        # 确保图像是numpy数组
        if not isinstance(rgb_image, np.ndarray):
            rgb_image = np.array(rgb_image)
        if not isinstance(ir_image, np.ndarray):
            ir_image = np.array(ir_image)
            # 如果IR图像是单通道，转换为三通道
            if len(ir_image.shape) == 2:
                ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        
        # 使用过滤后的对应点进行可视化
        # 使用更可靠的检查方式，确保只有当确实存在有效过滤点时才使用
        if filtered_correspondences is not None and len(filtered_correspondences) > 0:
            # 使用过滤后的点
            corr_np = filtered_correspondences.cpu().numpy()
            print(f"Visualization function internal: using filtered correspondences, shape={corr_np.shape}, valid correspondence points={len(corr_np)}")
        else:
            # 如果没有过滤或过滤后没有点，则使用原始点
            corr_np = correspondences.cpu().numpy()
            print(f"Visualization function internal: using original correspondences, shape={corr_np.shape}, valid correspondence points={len(corr_np)}")
        
        # Create visualization image: three images side by side (RGB image, IR image with connecting lines, IR reference image)
        # Ensure all three images have the same height
        h_rgb, w_rgb = rgb_image.shape[:2]
        h_ir, w_ir = ir_image.shape[:2]
        max_h = max(h_rgb, h_ir)
        
        # Create a canvas large enough to hold three images
        canvas = np.zeros((max_h, w_rgb + 2 * w_ir, 3), dtype=np.uint8)
        
        # 复制图像到画布：左侧RGB图像，中间和右侧IR图像
        canvas[:h_rgb, :w_rgb] = rgb_image[:h_rgb, :w_rgb]
        canvas[:h_ir, w_rgb:w_rgb + w_ir] = ir_image[:h_ir, :w_ir]  # 中间：带连线的IR图像
        canvas[:h_ir, w_rgb + w_ir:w_rgb + 2 * w_ir] = ir_image[:h_ir, :w_ir].copy()  # 右侧：IR参考原图
        
        # Draw corresponding points and connecting lines (randomly select 100 points to avoid visual clutter)
        num_points_to_draw = min(100, len(corr_np))
        
        if num_points_to_draw > 0:
            # Randomly select points if there are more than 100
            if len(corr_np) > num_points_to_draw:
                # Generate random indices without replacement
                random_indices = np.random.choice(len(corr_np), num_points_to_draw, replace=False)
                selected_points = corr_np[random_indices]
            else:
                selected_points = corr_np
            
            colors = np.random.randint(0, 255, size=(num_points_to_draw, 3), dtype=np.uint8)
            
            for i in range(num_points_to_draw):
                rgb_u, rgb_v, ir_u, ir_v = selected_points[i]
                
                # 确保坐标在有效范围内
                rgb_u = int(max(0, min(rgb_u, w_rgb - 1)))
                rgb_v = int(max(0, min(rgb_v, h_rgb - 1)))
                ir_u = int(max(0, min(ir_u, w_ir - 1)))
                ir_v = int(max(0, min(ir_v, h_ir - 1)))
                
                # 调整IR图像的u坐标（加上RGB图像的宽度，对应中间的IR图像）
                ir_u_canvas = ir_u + w_rgb
                
                # 获取随机颜色
                color = tuple(colors[i].tolist())
                
                # 在RGB图像上绘制点
                cv2.circle(canvas, (rgb_u, rgb_v), 5, color, -1)
                # 在IR图像上绘制点
                cv2.circle(canvas, (ir_u_canvas, ir_v), 5, color, -1)
                # 绘制连线
                cv2.line(canvas, (rgb_u, rgb_v), (ir_u_canvas, ir_v), color, 1, cv2.LINE_AA)
        else:
            print("Warning: No drawable corresponding points found")
        
        # 添加标签
        cv2.putText(canvas, 'RGB Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'IR Image (with Correspondences)', (w_rgb + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, 'IR Reference Image', (w_rgb + w_ir + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f'Corresponding Points: {num_points_to_draw}', (10, max_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f'IR GT Boxes: {len(ir_gt_boxes)}', (w_rgb + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存或显示结果
        if save_path:
            cv2.imwrite(save_path, canvas)
            print(f"Visualization result saved to: {save_path}")
        else:
            try:
                cv2.imshow('RGB-IR Correspondences', canvas)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Cannot display image, please specify save_path to save image in non-interactive environment")

      # 内部用单通道 z-buffer 存最近深度，用于深度遮挡判断
    
    def _save_lidar_as_ply(self, lidar_data, seq_id, frame_id):
        """
        将LiDAR点云保存为PLY格式用于可视化调试
        
        Args:
            lidar_data: np.ndarray, shape (N, 5), [x, y, z, intensity, ring]
            seq_id: 序列ID
            frame_id: 帧ID
        """

        # 创建保存目录
        root_dir= os.path.join(os.getcwd(), 'debug_lidar')
        save_dir = os.path.join(root_dir, 'debug_lidar_vis')
        os.makedirs(save_dir, exist_ok=True)
        
        # 提取点云数据并过滤异常值
        x = lidar_data[:, 0]
        y = lidar_data[:, 1]
        z = lidar_data[:, 2]
        intensity = lidar_data[:, 3]
        ring = lidar_data[:, 4]
        
        # 过滤x、y、z绝对值大于等于500的点
        filter_mask = (np.abs(x) < 500) & (np.abs(y) < 500) & (np.abs(z) < 500)
        x = x[filter_mask]
        y = y[filter_mask]
        z = z[filter_mask]
        intensity = intensity[filter_mask]
        ring = ring[filter_mask]
        
        # 归一化强度和ring用于颜色映射
        intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        ring_normalized = (ring - ring.min()) / (ring.max() - ring.min() + 1e-6)
        
        # 根据强度计算颜色 (RGB)
        r = (intensity_normalized * 255).astype(np.uint8)
        g = ((1 - intensity_normalized) * 128).astype(np.uint8)
        b = (ring_normalized * 255).astype(np.uint8)
        
        # 生成PLY文件头
        ply_header = f'''ply
            format ascii 1.0
            element vertex {len(x)}
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            ''' 
        
        # 保存PLY文件
        seq_id_fixed = seq_id.replace('\\', '_')
        filename = f"lidar_{seq_id_fixed}_{frame_id}.ply"
        filepath = os.path.join(save_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                f.write(ply_header)
                for i in range(len(x)):
                    f.write(f"{x[i]:.4f} {y[i]:.4f} {z[i]:.4f} {r[i]} {g[i]} {b[i]}\n")
            
            print(f"✅ 点云已保存: {filepath}")
            print(f"   - 点数: {len(x)}")
            print(f"   - X范围: [{x.min():.2f}, {x.max():.2f}]")
            print(f"   - Y范围: [{y.min():.2f}, {y.max():.2f}]")
            print(f"   - Z范围: [{z.min():.2f}, {z.max():.2f}]")
            print(f"   - 强度范围: [{intensity.min():.4f}, {intensity.max():.4f}]")
            print(f"   - Ring范围: [{ring.min():.0f}, {ring.max():.0f}]")
            
        except Exception as e:
            print(f"❌ 保存PLY失败: {e}")
    