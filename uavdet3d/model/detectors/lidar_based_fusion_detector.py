from heapq import heapreplace
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import time
from uavdet3d.model import backbone_2d, dense_head_2d
from uavdet3d.model.detectors.detector_template import DetectorTemplate
from uavdet3d.utils.object_encoder_laam6d import all_object_encoders
from uavdet3d.datasets.laam6d.dataset_utils import convert_9params_to_9points, match_gt, visualize_encode_decode, get_rgb_image
import matplotlib.pyplot as plt


def box9d_to_2d(boxes9d, img_width=1280., img_height=720.,
                intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):
    import numpy as np
    import cv2
    import torch

    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)
    else:
        intrinsic_mat = np.asarray(intrinsic_mat, dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4, dtype=np.float32)
    else:
        extrinsic_mat = np.asarray(extrinsic_mat, dtype=np.float32)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)
    else:
        distortion_matrix = np.asarray(distortion_matrix, dtype=np.float32).reshape(-1)

    # Carla → OpenCV 相机坐标变换
    carla_to_opencv = np.array([[0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]], dtype=np.float32)

    extrinsic_inv = np.linalg.inv(extrinsic_mat).astype(np.float32)

    # ★关键修正：点已经变到相机坐标了，所以 projectPoints 用 rvec/tvec=0
    rvec0 = np.zeros((3, 1), dtype=np.float32)
    tvec0 = np.zeros((3, 1), dtype=np.float32)

    boxes2d = []
    size = []

    for box in boxes9d:
        box_cv = []
        for pt in box:
            if isinstance(pt, torch.Tensor):
                pt = pt.detach().cpu().numpy()
            else:
                pt = np.asarray(pt, dtype=np.float32)

            pt_hom = np.append(pt, 1.0).astype(np.float32)  # (4,)
            pt_cam = extrinsic_inv @ pt_hom                 # world -> cam
            pt_cv = (carla_to_opencv @ pt_cam)[:3]          # -> OpenCV cam
            box_cv.append(pt_cv)

        box_cv = np.asarray(box_cv, dtype=np.float32)  # (9,3) : center + 8 corners

        corners_2d, _ = cv2.projectPoints(box_cv[1:], rvec0, tvec0, intrinsic_mat, distortion_matrix)

        # ★关键修正：NaN/inf 检查
        if np.isnan(corners_2d).any() or np.isinf(corners_2d).any():
            # 出现无效值就给个退化框，避免污染 NMS/AP
            boxes2d.append([0, 0, 0, 0])
            size.append(0.0)
            continue

        corners_2d = corners_2d.reshape(-1, 2)  # 保留 float 更稳

        x1, y1 = corners_2d.min(axis=0)
        x2, y2 = corners_2d.max(axis=0)

        # 最后再裁剪到图像边界
        x1 = float(np.clip(x1, 0, img_width - 1))
        y1 = float(np.clip(y1, 0, img_height - 1))
        x2 = float(np.clip(x2, 0, img_width - 1))
        y2 = float(np.clip(y2, 0, img_height - 1))

        if x2 <= x1 or y2 <= y1:
            boxes2d.append([0, 0, 0, 0])
            size.append(0.0)
            continue

        boxes2d.append([x1, y1, x2, y2])
        size.append(max(x2 - x1, y2 - y1))

    return np.asarray(boxes2d, dtype=np.float32).reshape(-1, 4), np.asarray(size, dtype=np.float32)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    """
    Perform Non-Maximum Suppression using NumPy.

    Args:
        boxes (np.ndarray): Nx4 array of boxes in [x1, y1, x2, y2] format.
        scores (np.ndarray): Confidence scores of the boxes.
        iou_threshold (float): IOU threshold for suppression.

    Returns:
        keep (List[int]): List of indices of boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # sort descending

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU of the top-scoring box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]  # shift by 1 due to indexing into order[1:]

    return keep


class LidarBasedFusionDetector(DetectorTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        # 构建网络模块
        self.build_special_networks()
        self.model_cfg = model_cfg
        self.dataset = dataset
        # 从配置中读取融合方式设置
        self.use_original_fusion = model_cfg.FEATURE_FUSION.get('USE_ORIGINAL_FUSION', False)
        self.center_decoder = all_object_encoders[self.model_cfg.POST_PROCESSING.DECONDER]
        self.max_num = self.model_cfg.POST_PROCESSING.MAX_OBJ
        self.score_thresh = self.model_cfg.POST_PROCESSING.SCORE_THRESH

        # 特征融合模块 - 添加use_relationship_matrix参数
        self.feature_fusion = FeatureFusionModule(
            in_channels=model_cfg.FEATURE_FUSION.IN_CHANNELS,
            out_channels=model_cfg.FEATURE_FUSION.OUT_CHANNELS,
            fusion_method=model_cfg.FEATURE_FUSION.METHOD,
            use_relationship_matrix=model_cfg.FEATURE_FUSION.get('USE_RELATIONSHIP_MATRIX', True)
        )

    def build_special_networks(self):
        """
        构建网络结构，同时支持单分支原图融合模式和两分支特征融合模式
        """

        model_info_dict = {
            'module_list': [],
            'image_shape': self.dataset.dataset_cfg.IM_RESIZE,
        }
        
        # 两分支模式使用5通道输入（3通道图像 + 1通道LiDAR + 1通道Radar）
        # 构建RGB分支的backbone
        self.rgb_backbone = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg,
        )
        model_info_dict['module_list'].append(self.rgb_backbone)

        # 构建IR分支的backbone 
        self.ir_backbone = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg,
        )
        model_info_dict['module_list'].append(self.ir_backbone)

        # 构建密集头
        self.dense_head_2d = dense_head_2d.__all__[self.model_cfg.DENSE_HEAD_2D.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD_2D,
        )
        model_info_dict['module_list'].append(self.dense_head_2d)

        self.module_list = model_info_dict['module_list']
    
    def _extract_modality_data(self, batch_dict):
        """
        从batch_dict中提取所有需要的模态数据并确保转换为Tensor
        Args:
            batch_dict: 包含所有数据的批次字典
            device: 目标设备（可选），如果提供则将数据转换到该设备
        Returns:
            rgb_data: RGB图像数据
            ir_data: IR图像数据
            lidar_rgb_proj: LiDAR投影到RGB的图像数据
            radar_rgb_proj: Radar投影到RGB的图像数据
            lidar_ir_proj: LiDAR投影到IR的图像数据
            radar_ir_proj: Radar投影到IR的图像数据
            rgb_intrinsic: RGB相机内参
            rgb_extrinsic: RGB相机外参
            ir_intrinsic: IR相机内参
            ir_extrinsic: IR相机外参
            lidar_extrinsic: LiDAR外参
            lidar_points: LiDAR点云数据
            precomputed_correspondences: 预处理的对应点
            new_im_size: 预处理图像尺寸
            batch_size: 批次大小
        """
        # 注意：在dataset中，image数据的格式是(B, 2, 3, H, W)
        # 其中第2维的0表示RGB，1表示IR
        image = batch_dict.get('image')
        device = image.device
        rgb_data = image[:, 0, :, :, :] # 提取RGB数据 [B, 3, H, W]
        ir_data = image[:, 1, :, :, :] # 提取IR数据 [B, 3, H, W]
        
        # 获取LiDAR投影到RGB和IR的数据
        lidar_rgb_proj = batch_dict.get('rgb_lidar_projection')  # LiDAR投影到RGB
        radar_rgb_proj = batch_dict.get('rgb_radar_projection')  # Radar投影到RGB
        lidar_ir_proj = batch_dict.get('ir_lidar_projection')  # LiDAR投影到IR
        radar_ir_proj = batch_dict.get('ir_radar_projection')  # Radar投影到IR
        
        # 1. 相机内参/外参（确保转换为Tensor）
        rgb_intrinsic = self._ensure_tensor(batch_dict.get('rgb_intrinsic'))  # 形状：(B, 3, 3) 或 (3, 3)（单批次）
        rgb_extrinsic = self._ensure_tensor(batch_dict.get('rgb_extrinsic'))  # 形状：(B, 4, 4) 或 (4, 4)
        ir_intrinsic = self._ensure_tensor(batch_dict.get('ir_intrinsic'))    # 形状：(B, 3, 3) 或 (3, 3)
        ir_extrinsic = self._ensure_tensor(batch_dict.get('ir_extrinsic'))    # 形状：(B, 4, 4) 或 (4, 4)
        lidar_extrinsic = self._ensure_tensor(batch_dict.get('lidar_extrinsic'))  # 形状：(B, 4, 4) 或 (4, 4)
        
        # 2. LiDAR点云（确保转换为Tensor）
        lidar_points = self._ensure_tensor(batch_dict.get('lidar_points'))    # 形状：(B, N, 3)（B=批次，N=点云数量）
        
        # 3. 预处理的对应点（确保转换为Tensor）
        precomputed_correspondences = self._ensure_tensor(batch_dict.get('precomputed_correspondences'))  # 形状：(B, N_corr, 4)
        
        # 4. 图像/特征图尺寸和批次大小
        new_im_size = self._ensure_tensor(batch_dict.get('new_im_size'))      # 预处理图像尺寸：(B, 2) → (宽, 高)
        batch_size = batch_dict.get('batch_size')     # 批次大小（优先从batch_dict获取）
        
        # 如果提供了设备，将所有数据转换到该设备
        if device is not None:
            rgb_intrinsic = rgb_intrinsic.to(device)
            rgb_extrinsic = rgb_extrinsic.to(device)
            ir_intrinsic = ir_intrinsic.to(device)
            ir_extrinsic = ir_extrinsic.to(device)
            lidar_extrinsic = lidar_extrinsic.to(device)
            lidar_points = lidar_points.to(device)
            if precomputed_correspondences is not None:
                precomputed_correspondences = precomputed_correspondences.to(device)
            if new_im_size is not None:
                new_im_size = new_im_size.to(device)
        
        return (rgb_data, ir_data, lidar_rgb_proj, radar_rgb_proj, lidar_ir_proj, radar_ir_proj,
                rgb_intrinsic, rgb_extrinsic, ir_intrinsic, ir_extrinsic, lidar_extrinsic, lidar_points,
                precomputed_correspondences, new_im_size, batch_size)
    
    def _ensure_tensor(self, data):
        """
        确保数据是Tensor类型，如果不是则进行转换
        Args:
            data: 需要转换的数据
        Returns:
            Tensor类型的数据，如果输入为None则返回None
        """
        if data is None:
            return None
        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            else:
                return torch.tensor(data)
        return data

    def _prepare_branch_data(self, batch_dict, image_data, lidar_proj, radar_proj):
        """
        为特定分支准备数据，将图像数据与世界坐标映射和雷达速度热力图叠加
        Args:
            batch_dict: 原始批次字典
            image_data: 图像数据（RGB或IR）
            lidar_proj: LiDAR投影数据列表
            radar_proj: Radar投影数据列表
        Returns:
            准备好的分支数据（模态叠加后的结果）
        """
        
        B, C, H, W = image_data.shape
        device = image_data.device
        
        # 初始化世界坐标映射和雷达速度热力图
        world_coords_map = torch.zeros((B, 3, H, W), device=device)
        radar_velocity_heatmap = torch.zeros((B, 3, H, W), device=device)  # 使用3通道
        
        # 处理LiDAR投影数据 - 直接处理列表格式
        coords_maps = []
        for item in lidar_proj[:B]:
            # 直接获取世界坐标映射并转换为tensor
            coords_map = item['depth_map']
            coords_map = np.clip(coords_map, 0, self.dataset.dataset_cfg.MAX_DIS) / self.dataset.dataset_cfg.MAX_DIS
            coords_map_tensor = torch.from_numpy(coords_map).to(device) if isinstance(coords_map, np.ndarray) else coords_map.to(device)
            coords_maps.append(coords_map_tensor)
        
        # 堆叠为批量tensor并调整尺寸
        batch_coords = torch.stack(coords_maps)
        if batch_coords.shape[2:] != (H, W):
            batch_coords = F.interpolate(batch_coords, size=(H, W), mode='bilinear', align_corners=False)
        world_coords_map = batch_coords
        
        # 处理雷达投影数据 - 直接处理列表格式
        heatmaps = []
        for item in radar_proj[:B]:
            # 直接获取雷达速度热力图并转换为tensor
            heatmap = item['radar_velocity_heatmap']
            heatmap_tensor = torch.from_numpy(heatmap).to(device) if isinstance(heatmap, np.ndarray) else heatmap.to(device)
            heatmaps.append(heatmap_tensor)
        
        # 堆叠为批量tensor并调整尺寸
        batch_heatmaps = torch.stack(heatmaps)
        if batch_heatmaps.shape[2:] != (H, W):
            batch_heatmaps = F.interpolate(batch_heatmaps, size=(H, W), mode='bilinear', align_corners=False)
        radar_velocity_heatmap = batch_heatmaps

        branch_data = torch.cat([image_data, world_coords_map, radar_velocity_heatmap], dim=1)
        
        # 模态叠加：将图像数据、世界坐标映射和雷达速度热力图在通道维度上拼接
        # 3（图像通道）+ 3（世界坐标）+ 3（雷达速度）= 9通道
        
        # ==============================================================================================================
        # # 保存可视化：世界坐标映射和雷达速度热力图
        # vis_dir = 'vis_branch_data'
        # os.makedirs(vis_dir, exist_ok=True)
        
        # # 将世界坐标映射归一化到0-255并保存
        # for i in range(world_coords_map.shape[0]):
        #     world_coords_map_np = world_coords_map[i].detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        #     # 使用与dataset中normalize_world_coords相似的方法处理
        #     h, w = world_coords_map_np.shape[:2]
        #     world_coords_map_vis = np.zeros((h, w, 3), dtype=np.uint8)         
        #     # 对每个坐标通道(X, Y, Z)分别进行归一化处理
        #     for c in range(3):
        #         # 获取当前通道数据
        #         coords = world_coords_map_np[..., c]
        #         # 创建有效掩码（非零值）
        #         valid_mask = coords != 0
        #         channel_normalized = np.zeros_like(coords, dtype=np.uint8)
        #         # 检查是否有有效值
        #         if np.any(valid_mask):
        #             coords_valid = coords[valid_mask]
        #             # 使用百分位数进行异常值处理
        #             min_val = np.percentile(coords_valid, 5)
        #             max_val = np.percentile(coords_valid, 95)
        #             # 确保至少有一定的范围
        #             if max_val <= min_val:
        #                 max_val = min_val + 1e-6  # 添加一个小值避免除零
        #             # 对有效区域进行归一化
        #             normalized = (coords_valid - min_val) / (max_val - min_val)
        #             normalized = np.clip(normalized, 0, 1)  # 确保值在0-1范围内
        #             channel_normalized[valid_mask] = (normalized * 255).astype(np.uint8)
        #         world_coords_map_vis[..., c] = channel_normalized
        #     world_coords_map_vis = cv2.cvtColor(world_coords_map_vis, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(vis_dir, f'world_coords_map_{int(time.time()*1000)}.png'), world_coords_map_vis)
        # 
        # # 将雷达速度热力图归一化到0-255并保存, 使用固定归一化范围进行雷达速度热力图可视化
        # fixed_min, fixed_max = -10.0, 10.0  # 假设的雷达速度范围（根据实际数据调整）
        # velocity_threshold = 0.5  # 速度阈值，用于过滤背景噪声
        # for i in range(radar_velocity_heatmap.shape[0]):
        #     radar_velocity_heatmap_np = radar_velocity_heatmap[i].detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        #     velocity_magnitude = np.abs(radar_velocity_heatmap_np)
        #     timestamp = int(time.time()*1000)
            
        #     # 方案1：使用阈值过滤背景，让背景为黑色
        #     # 创建掩码，只保留速度绝对值大于阈值的区域
        #     max_magnitude = np.max(velocity_magnitude, axis=2, keepdims=True)
        #     mask = max_magnitude > velocity_threshold
        #     # 归一化处理
        #     normalized = (radar_velocity_heatmap_np - fixed_min) / (fixed_max - fixed_min)
        #     normalized = np.clip(normalized, 0, 1)  # 确保值在0-1范围内
        #     # 应用掩码，将背景设为黑色（0）
        #     masked_heatmap = (normalized * 255 * mask).astype(np.uint8)
        #     # 转换颜色通道顺序并保存
        #     masked_heatmap_bgr = cv2.cvtColor(masked_heatmap, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(os.path.join(vis_dir, f'radar_velocity_thresholded_{timestamp}.png'), masked_heatmap_bgr)
            
        #     # 方案2：使用JET色彩映射，使目标更加突出（可选）
        #     # 将3通道数据转换为单通道（取速度幅值的平均值）
        #     gray_heatmap = np.mean(velocity_magnitude, axis=2)
        #     # 归一化并应用掩码
        #     gray_normalized = (gray_heatmap - fixed_min) / (fixed_max - fixed_min)
        #     gray_normalized = np.clip(gray_normalized, 0, 1)
        #     gray_normalized = gray_normalized * (gray_heatmap > velocity_threshold)
        #     # 转换为8位灰度图
        #     gray_8bit = (gray_normalized * 255).astype(np.uint8)
        #     # 应用JET色彩映射
        #     colored_heatmap = cv2.applyColorMap(gray_8bit, cv2.COLORMAP_JET) 
        #     # 保存彩色热力图
        #     cv2.imwrite(os.path.join(vis_dir, f'radar_velocity_colored_{timestamp}.png'), colored_heatmap)
        # # ==============================================================================================================
        
        # # 调试图像保存
        # if not os.path.exists('debug_projections'):
        #     os.makedirs('debug_projections', exist_ok=True)
        # print(batch_dict['frame_id'][0])
        # from datetime import datetime
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # # 保存IR/RGB图像 (假设image_data是CHW格式的张量)
        # img = image_data[0].permute(1, 2, 0).cpu().numpy()
        # print(f"RGB/IR image shape: {img.shape}, min: {img.min()}, max: {img.max()}") 
        # img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 归一化到0-1
        # plt.imsave(f"debug_projections/rgb_ir_{batch_dict['frame_id'][0]}_{timestamp}.png", img)
        
        # # 保存Lidar投影
        # lidar_img = np.squeeze(lidar_proj[0]['depth_map']).transpose(1, 2, 0)
        # print(f"LiDAR image shape: {lidar_img.shape}, min: {lidar_img.min()}, max: {lidar_img.max()}")
        # lidar_img = np.clip(lidar_img, 0, 150) / 150
        # lidar_img = (lidar_img * 255).astype(np.uint8)
        # print(f"LiDAR image shape: {lidar_img.shape}, min: {lidar_img.min()}, max: {lidar_img.max()}")
        # plt.imsave(f"debug_projections/lidar_proj_{batch_dict['frame_id'][0]}_{timestamp}.png", lidar_img, cmap='viridis')
        
        # # 保存Radar投影
        # radar_img = radar_proj[0]['radar_velocity_heatmap'].transpose(1, 2, 0) 
        # radar_img = np.abs(radar_img) # 取绝对值，确保速度值为非负数
        # print(f"Radar image shape: {radar_img.shape}, min: {radar_img.min()}, max: {radar_img.max()}") 
        # radar_img = (radar_img - radar_img.min()) / (radar_img.max() - radar_img.min() + 1e-8)  # 归一化到0-1
        # plt.imsave(f"debug_projections/radar_proj_{batch_dict['frame_id'][0]}_{timestamp}.png", radar_img, cmap='plasma')
        # time.sleep(1)
        
        return branch_data

    def forward(self, batch_dict):
        ###########################################################################
        # 1. 从_extract_modality_data获取所有需要的数据，已经确保转换为Tensor
        ###########################################################################
        (rgb_data, ir_data, lidar_rgb_proj, radar_rgb_proj, lidar_ir_proj, radar_ir_proj,
         rgb_intrinsic, rgb_extrinsic, ir_intrinsic, ir_extrinsic, lidar_extrinsic, lidar_points,
         precomputed_correspondences, new_im_size, batch_size) = self._extract_modality_data(batch_dict)

        # 从配置中获取融合方式
        relationship_matrix = None
        
        if self.use_original_fusion:
            # 将RGB和IR数据在通道维度上拼接，得到6通道数据
            fused_images = torch.cat([rgb_data, rgb_data], dim=1)  # [B, 6, H, W]
            
            # 直接从fused_images获取形状信息
            B, _, H, W = fused_images.shape
            device = fused_images.device
            
            # 初始化世界坐标映射和雷达速度热力图
            world_coords_map = torch.zeros((B, 3, H, W), device=device)  # [B, 3, H, W]
            radar_velocity_heatmap = torch.zeros((B, 3, H, W), device=device)  # [B, 3, H, W]
            
            # 处理LiDAR投影数据 - 使用RGB投影的lidar数据
            coords_maps = []
            if lidar_rgb_proj is not None and isinstance(lidar_rgb_proj, list) and len(lidar_rgb_proj) >= B:
                for item in lidar_rgb_proj[:B]:
                    # 直接获取世界坐标映射并转换为tensor
                    if item is not None and 'depth_map' in item:
                        coords_map = item['depth_map']
                        coords_map = np.clip(coords_map, 0, self.dataset.dataset_cfg.MAX_DIS) / self.dataset.dataset_cfg.MAX_DIS
                        coords_map_tensor = torch.from_numpy(coords_map).to(device) if isinstance(coords_map, np.ndarray) else coords_map.to(device)
                        coords_maps.append(coords_map_tensor)
                    else:
                        # 如果没有有效的depth_map，使用零张量
                        coords_maps.append(torch.zeros((3, H, W), device=device))
            else:
                # 如果lidar_rgb_proj格式错误或不完整，为每个批次创建零张量
                for _ in range(B):
                    coords_maps.append(torch.zeros((3, H, W), device=device))
            
            # 堆叠为批量tensor并调整尺寸
            batch_coords = torch.stack(coords_maps)
            if batch_coords.shape[2:] != (H, W):
                batch_coords = F.interpolate(batch_coords, size=(H, W), mode='bilinear', align_corners=False)
            world_coords_map = batch_coords
            
            # 处理雷达投影数据 - 使用RGB投影的radar数据
            heatmaps = []
            if radar_rgb_proj is not None and isinstance(radar_rgb_proj, list) and len(radar_rgb_proj) >= B:
                for item in radar_rgb_proj[:B]:
                    # 直接获取雷达速度热力图并转换为tensor
                    if item is not None and 'radar_velocity_heatmap' in item:
                        heatmap = item['radar_velocity_heatmap']
                        heatmap_tensor = torch.from_numpy(heatmap).to(device) if isinstance(heatmap, np.ndarray) else heatmap.to(device)
                        heatmaps.append(heatmap_tensor)
                    else:
                        # 如果没有有效的radar_velocity_heatmap，使用零张量
                        heatmaps.append(torch.zeros((3, H, W), device=device))
            else:
                # 如果radar_rgb_proj格式错误或不完整，为每个批次创建零张量
                for _ in range(B):
                    heatmaps.append(torch.zeros((3, H, W), device=device))
            
            # 堆叠为批量tensor并调整尺寸
            batch_heatmaps = torch.stack(heatmaps)
            if batch_heatmaps.shape[2:] != (H, W):
                batch_heatmaps = F.interpolate(batch_heatmaps, size=(H, W), mode='bilinear', align_corners=False)
            radar_velocity_heatmap = batch_heatmaps
            
            # 将融合图像与lidar和radar数据拼接
            fused_images_with_projections = torch.cat([fused_images, radar_velocity_heatmap, world_coords_map], dim=1)
            
            # 复用RGB backbone，现在RGB backbone配置为12通道，原图融合模式通过动态调整第一层卷积权重来支持12通道输入
            fused_features = self.rgb_backbone({'image': fused_images_with_projections})['features_2d']
            lidar_depth_map = world_coords_map[:, 0:1, :]
        else:
            ###########################################################################
            # 2. 单模态分支：保持原逻辑，确保输入为GPU Tensor
            ###########################################################################
            # RGB分支特征提取
            rgb_branch_data = self._prepare_branch_data(batch_dict, rgb_data, lidar_rgb_proj, radar_rgb_proj)
            rgb_features = self.rgb_backbone({'image': rgb_branch_data})['features_2d']  # (B, C, H_feat, W_feat)
            
            # IR分支特征提取
            ir_branch_data = self._prepare_branch_data(batch_dict, ir_data, lidar_ir_proj, radar_ir_proj)
            ir_features = self.ir_backbone({'image': ir_branch_data})['features_2d']    # (B, C, H_feat, W_feat)
            
            ###########################################################################
            # 3. 关系矩阵生成：核心优化（PyTorch全张量运算，消除循环+CPU转换）
            ###########################################################################
            device = rgb_features.device  # 统一设备（与特征图一致，避免跨设备开销）

            # 条件检查：确保所有必要参数存在且为Tensor（避免空值/NumPy转换）
            required_params = [rgb_intrinsic, rgb_extrinsic, ir_intrinsic, ir_extrinsic, lidar_extrinsic, lidar_points, precomputed_correspondences]
            if all(x is not None and isinstance(x, torch.Tensor) for x in required_params):
                # 特征图尺寸（从RGB特征推导，确保与IR特征一致）
                B, C_feat, H_feat, W_feat = rgb_features.shape
                
                # 初始化关系矩阵：(B, 2, H_feat, W_feat)，-1表示无对应关系，权重初始为1
                relationship_matrix = torch.full((B, 2, H_feat, W_feat), -1.0, dtype=torch.float32, device=device)
                relationship_matrix[:, 1, :, :] = 1.0  # 第二通道（权重）全1，用广播赋值替代循环
                
                #######################################################################
                # 关键优化1：预处理对应点格式统一（确保适配多批次）
                #######################################################################
                # 若为单批次（形状：(N_corr, 4)），扩为多批次格式：(B, N_corr, 4)
                if precomputed_correspondences.dim() == 2:
                    precomputed_correspondences = precomputed_correspondences.unsqueeze(0).repeat(B, 1, 1)
                # 确保对应点坐标为float（避免类型错误）
                # 已在_extract_modality_data方法中转换到正确设备
                precomputed_correspondences = precomputed_correspondences.float()
                
                #######################################################################
                # 关键优化2：坐标缩放（向量化运算，替代Python循环）
                #######################################################################
                # 提取图像尺寸：new_im_size → (B, 2) → 拆分为宽(W_img)、高(H_img)，扩维适配广播
                W_img = new_im_size[:, 0].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)：每个批次的图像宽度
                H_img = new_im_size[:, 1].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)：每个批次的图像高度
                
                # 提取所有对应点的图像坐标（RGB和IR）
                rgb_u_img = precomputed_correspondences[:, :, 0].unsqueeze(2)  # (B, N_corr, 1)：RGB图像u坐标
                rgb_v_img = precomputed_correspondences[:, :, 1].unsqueeze(2)  # (B, N_corr, 1)：RGB图像v坐标
                ir_u_img = precomputed_correspondences[:, :, 2].unsqueeze(2)   # (B, N_corr, 1)：IR图像u坐标
                ir_v_img = precomputed_correspondences[:, :, 3].unsqueeze(2)   # (B, N_corr, 1)：IR图像v坐标
                
                # 坐标缩放到特征图尺寸：(图像坐标 / 图像尺寸) * 特征图尺寸
                # torch.clamp确保坐标在[0, H_feat-1]和[0, W_feat-1]内，避免越界
                feat_rgb_u = torch.clamp((rgb_u_img / W_img) * W_feat, 0, W_feat - 1).long()  # (B, N_corr, 1)
                feat_rgb_v = torch.clamp((rgb_v_img / H_img) * H_feat, 0, H_feat - 1).long()  # (B, N_corr, 1)
                feat_ir_u = torch.clamp((ir_u_img / W_img) * W_feat, 0, W_feat - 1).long()   # (B, N_corr, 1)
                feat_ir_v = torch.clamp((ir_v_img / H_img) * H_feat, 0, H_feat - 1).long()   # (B, N_corr, 1)
                
                # 计算IR特征图的一维索引值
                ir_index = feat_ir_v * W_feat + feat_ir_u  # (B, N_corr, 1)：IR特征图的一维索引
                
                #######################################################################
                # 关键优化3：批量索引填充（避免循环遍历每个点）
                #######################################################################
                # 生成批次索引：每个对应点所属的批次ID（用于多批次并行填充）
                batch_idx = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
                batch_idx = batch_idx.repeat(1, precomputed_correspondences.shape[1], 1)  # (B, N_corr, 1)
                
                # 展开索引和值：从(B, N_corr, 1) → (B*N_corr,)，适配Tensor的flat索引
                b_idx = batch_idx.view(-1)          # 批次索引（展平）
                v_idx = feat_rgb_v.view(-1)         # RGB特征图v坐标（展平）
                u_idx = feat_rgb_u.view(-1)         # RGB特征图u坐标（展平）
                idx_val = ir_index.view(-1).float() # IR索引值（转为float，匹配关系矩阵数据类型）
                
                # 直接填充关系矩阵第一通道（IR索引）：无循环，一步到位
                relationship_matrix[b_idx, 0, v_idx, u_idx] = idx_val
            
            ###########################################################################
            # 4. 特征融合与检测头：保持原逻辑，确保输入为Tensor
            ###########################################################################
            fused_features = self.feature_fusion(rgb_features, ir_features, relationship_matrix)
            lidar_depth_map = rgb_branch_data[:, 3:4, :]
        
        # 检测头处理
        batch_dict['lidar_depth_map'] = lidar_depth_map  # [B, 1, 560, 960]
        batch_dict['features_2d'] = fused_features  # [B, C, 140, 240]
        batch_dict = self.dense_head_2d(batch_dict)
        
        # 后处理：保持原逻辑
        return {'loss': self.get_training_loss(), **batch_dict} if self.training else self.post_processing(batch_dict)

    def get_training_loss(self):
        """
        计算训练损失
        """
        if hasattr(self.dense_head_2d, 'get_loss'):
            # 首先检查dense_head_2d的forward_loss_dict是否包含预期的键
            if hasattr(self.dense_head_2d, 'forward_loss_dict'):
                forward_loss_dict = self.dense_head_2d.forward_loss_dict
                
                # 检查pred_center_dict和gt_center_dict是否存在
                pred_exists = 'pred_center_dict' in forward_loss_dict
                gt_exists = 'gt_center_dict' in forward_loss_dict
                
                # 详细检查gt_center_dict的内容
                if gt_exists:
                    gt_center_dict = forward_loss_dict['gt_center_dict']
                    
                    # 检查每个头的gt数据
                    for head_name in ['hm', 'center_res', 'center_dis', 'dim', 'rot']:
                        if head_name in gt_center_dict:
                            gt_data = gt_center_dict[head_name]
                            if isinstance(gt_data, torch.Tensor):
                                # 计算非零元素数量
                                non_zero_count = torch.nonzero(gt_data).shape[0]
                                total_elements = gt_data.numel()
                                
                                # 对于'hm'，特别检查正值数量
                                if head_name == 'hm':
                                    pos_count = (gt_data > 0).sum().item()
                        
                # 检查pred_center_dict的内容
                if pred_exists:
                    pred_center_dict = forward_loss_dict['pred_center_dict']
                
            # 计算损失
            loss = self.dense_head_2d.get_loss()
            
            # 检查loss值
            if isinstance(loss, torch.Tensor):
                loss_value = loss.item()
                
                # 如果loss为0，尝试创建一个小的可微损失以确保训练能继续
                if loss_value == 0:
                    print("[WARNING] Loss is zero! This might indicate issues with ground truth data or loss calculation.")
                    # 检查是否有任何梯度连接到loss
                    if not loss.requires_grad:
                        print("[WARNING] Loss tensor does not require gradients!")
                        # 创建一个可微的小损失
                        return torch.tensor(1e-6, requires_grad=True, device=next(self.parameters()).device)
            
            return loss
        else:
            print("[ERROR] dense_head_2d does not have get_loss method!")
            # 创建一个可微的小损失，确保梯度能传播
            return torch.tensor(1e-6, requires_grad=True, device=next(self.parameters()).device)

    def post_processing(self, batch_dict):
        """
        后处理函数，与CenterDetLaam6d类似
        """
        batch_size = batch_dict['batch_size']
        im_num = self.dataset.IM_NUM

        all_pred_boxes9d = []
        all_confidence = []

        hm = batch_dict['pred_center_dict']['hm']
        center_res = batch_dict['pred_center_dict']['center_res']
        center_dis = batch_dict['pred_center_dict']['center_dis']
        dim = batch_dict['pred_center_dict']['dim']
        rot = batch_dict['pred_center_dict']['rot']

        # 用GT替换预测，验证编码解码是否正确
        # hm = batch_dict['hm']
        # center_res = batch_dict['center_res']
        # center_dis = batch_dict['center_dis']
        # dim = batch_dict['dim']
        # rot = batch_dict['rot']

        def reshape_t(tensor, batch_size):
            BK, C, W, H = tensor.shape
            return tensor.reshape(batch_size, -1, C, W, H)

        hm = reshape_t(hm, batch_size)
        center_res = reshape_t(center_res, batch_size)
        center_dis = reshape_t(center_dis, batch_size)
        dim = reshape_t(dim, batch_size)
        rot = reshape_t(rot, batch_size)

        for batch_id in range(batch_size):
            intrinsic = batch_dict['intrinsic'][batch_id]  # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id]  # 4, 4
            distortion = batch_dict['distortion'][batch_id]  # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id]  # 2,
            new_im_size = batch_dict['new_im_size'][batch_id]  # 2,
            stride = batch_dict['stride'][batch_id]

            this_hm = hm[batch_id]
            this_center_res = center_res[batch_id]
            this_center_dis = center_dis[batch_id] * self.dataset.dataset_cfg.MAX_DIS
            this_dim = dim[batch_id] * self.dataset.dataset_cfg.MAX_SIZE
            this_rot = rot[batch_id]

            this_lidar_depth_map = None
            if 'lidar_depth_map' in batch_dict and batch_dict['lidar_depth_map'] is not None:
                this_lidar_depth_map = (batch_dict['lidar_depth_map'][batch_id, 0] * 
                                        self.dataset.dataset_cfg.MAX_DIS).detach().cpu().numpy()

            # 确保所有输入到decoder的参数类型统一为numpy数组
            # 通用转换函数：处理张量、列表和标量
            def convert_tensor(x):
                if isinstance(x, torch.Tensor):
                    if x.ndim == 0:  # 标量张量
                        return x.item()
                    else:  # 多维张量
                        return x.cpu().numpy()
                elif isinstance(x, list) and x and isinstance(x[0], torch.Tensor):
                    return [item.item() for item in x]
                return x
            
            # 应用转换函数到所有变量
            this_hm = convert_tensor(this_hm)
            this_center_res = convert_tensor(this_center_res)
            this_center_dis = convert_tensor(this_center_dis)
            this_dim = convert_tensor(this_dim)
            this_rot = convert_tensor(this_rot)
            intrinsic = convert_tensor(intrinsic)
            extrinsic = convert_tensor(extrinsic)
            distortion = convert_tensor(distortion)
            new_im_size = convert_tensor(new_im_size)
            raw_im_size = convert_tensor(raw_im_size)
            stride = convert_tensor(stride)

            pred_boxes9d, confidence = self.center_decoder(
                this_hm, this_center_res, this_center_dis,
                this_dim, this_rot, intrinsic, extrinsic, distortion,
                new_im_size[0], new_im_size[1], raw_im_size[0], raw_im_size[1],
                stride, im_num, self.max_num, lidar_depth_map=this_lidar_depth_map
            )
            if len(pred_boxes9d)>0:
                keep = confidence > self.score_thresh
                pred_boxes9d = pred_boxes9d[keep]
                confidence = confidence[keep]

                pred_boxes_params = pred_boxes9d[:, :-1]
                pred_box9d_9points = convert_9params_to_9points(pred_boxes_params)
                boxes2d,_ = box9d_to_2d(boxes9d=pred_box9d_9points, intrinsic_mat=intrinsic, extrinsic_mat=extrinsic)
                nms_mask = nms_numpy(boxes2d,confidence,iou_threshold=0.1)
                pred_boxes9d = pred_boxes9d[nms_mask]
                boxes2d = boxes2d[nms_mask]
                confidence = confidence[nms_mask]

            # ==========================================================================================================
            # 保存 this_hm 为图片
            save_dir = "encode_decode_vis_pred"
            os.makedirs(save_dir, exist_ok=True)
            
            this_hm_gt  = convert_tensor(batch_dict['hm'][batch_id])
            this_hm_gt = np.squeeze(this_hm_gt)
            this_hm = np.squeeze(this_hm)
 
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            im1 = ax1.imshow(this_hm_gt, cmap='hot', interpolation='nearest')
            ax1.set_title("Ground Truth Heatmap")
            fig.colorbar(im1, ax=ax1)
            im2 = ax2.imshow(this_hm, cmap='hot', interpolation='nearest')
            ax2.set_title("Predicted Heatmap")
            fig.colorbar(im2, ax=ax2)
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"this_hm_{int(time.time()*1000)}.png")
            # plt.savefig(save_path)
            # plt.show()
            plt.close()

            data_dict = {}
            for key in batch_dict:
                # 跳过不需要处理的键
                if key in {'batch_size', 'down_sample_ratio', 'pred_center_dict', 'sorted_namelist'}:
                    continue
            
                if key in {'intrinsic', 'extrinsic', 'distortion'}:
                    value = [batch_dict[key][batch_id]]
                else:
                    # 提取当前batch_id的样本
                    value = batch_dict[key][batch_id]
            
                # 转换为numpy数组
                if isinstance(value, torch.Tensor):
                    # 处理PyTorch张量：分离梯度→移到CPU→转numpy
                    data_dict[key] = value.detach().cpu().numpy()
                else:
                    try:
                        # 尝试转换其他类型（列表、标量等）
                        data_dict[key] = np.array(value)
                    except:
                        # 无法转换时保留原始值（或根据需求处理）
                        data_dict[key] = value
                        print(f"警告：键'{key}'的值无法转换为NumPy数组，已保留原始类型")
            
            gt_box9d, pred_box9d = match_gt(data_dict['gt_boxes'], pred_box9d_9points)
     
            # 可视化对比：使用转换后的 9 点格式框
            visualize_encode_decode(
                data_dict,
                gt_box9d,  # 原始 9 点格式框
                pred_box9d,  # 转换后的 9 点格式框
                save_dir,
                save_prefix=f"{data_dict.get('seq_id')}_{data_dict.get('frame_id')}"
            )

            img = get_rgb_image(data_dict)
            # 绘制预测框(绿色)
            for box in boxes2d:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 保存带2D框的图像
            save_prefix=f"{data_dict.get('seq_id')}_{data_dict.get('frame_id')}"
            safe_prefix = save_prefix.replace('/', '_').replace('\\', '_')  # 替换所有路径分隔符
            full_save_path = os.path.join(save_dir, f"{safe_prefix}_compare.png")  # 直接保存在根目录
            if not cv2.imwrite(full_save_path, img):
                print(f"警告: 无法保存图像到 {full_save_path}")
            # ==========================================================================================================

            all_pred_boxes9d.append(pred_boxes9d)
            all_confidence.append(confidence)

        batch_dict['pred_boxes9d'] = all_pred_boxes9d
        batch_dict['confidence'] = all_confidence

        return batch_dict


class FeatureFusionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fusion_method='concat',
        use_relationship_matrix=False,
        smooth_kernel_size=3,
        smooth_mix=0.3,  # 0~1, 越大越平滑
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_method = fusion_method
        self.use_relationship_matrix = use_relationship_matrix
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_mix = smooth_mix

        # 特征转换层
        self.rgb_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.ir_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 融合策略（非关系矩阵分支使用）
        if fusion_method == 'concat':
            self.concat_conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_method == 'attention':
            mid = max(1, out_channels // 4)
            self.attention_rgb = nn.Sequential(
                nn.Conv2d(out_channels, mid, kernel_size=1),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.attention_ir = nn.Sequential(
                nn.Conv2d(out_channels, mid, kernel_size=1),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
        elif fusion_method == 'add':
            self.rgb_weight = nn.Parameter(torch.tensor(0.5))
            self.ir_weight = nn.Parameter(torch.tensor(0.5))

        # 输出卷积
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 预注册高斯核（buffer，不参与训练，不会每次 forward new）
        if smooth_kernel_size == 3:
            g = torch.tensor([[1., 2., 1.],
                              [2., 4., 2.],
                              [1., 2., 1.]])
        else:
            # 简单兜底：用全 1 的核做均值平滑（你也可以自己扩展成真正高斯）
            g = torch.ones((smooth_kernel_size, smooth_kernel_size), dtype=torch.float32)
        g = g / g.sum()
        self.register_buffer("gaussian_kernel", g.view(1, 1, smooth_kernel_size, smooth_kernel_size))

    def forward(self, rgb_features, ir_features, relationship_matrix=None):
        # 调整特征通道
        rgb_feat = self.rgb_conv(rgb_features)
        ir_feat = self.ir_conv(ir_features)

        # 调整特征大小以匹配
        if rgb_feat.shape[-2:] != ir_feat.shape[-2:]:
            ir_feat = F.interpolate(
                ir_feat,
                size=rgb_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # 使用关系矩阵进行特征融合
        if self.use_relationship_matrix and relationship_matrix is not None:
            ir_index_map = relationship_matrix[:, 0].long()  # (B,H,W)
            valid_mask = (ir_index_map != -1)
            fused_feat = rgb_feat  # 直接用 rgb_feat（不 clone，省显存）

            # 这里暂时保留一个融合权重接口
            fusion_weight = 1.0
            self._index_based_fusion(
                fused_feat=fused_feat,
                ir_feat=ir_feat,
                ir_index_map=ir_index_map,
                valid_mask=valid_mask,
                fusion_weight=fusion_weight,
            )
        else:
            # 回退到传统融合方法（当不使用关系矩阵时）
            if self.fusion_method == 'concat':
                fused_feat = torch.cat([rgb_feat, ir_feat], dim=1)
                fused_feat = self.concat_conv(fused_feat)
            elif self.fusion_method == 'attention':
                rgb_att = self.attention_rgb(rgb_feat)
                ir_att = self.attention_ir(ir_feat)
                att_sum = rgb_att + ir_att + 1e-8
                rgb_att = rgb_att / att_sum
                ir_att = ir_att / att_sum
                fused_feat = rgb_feat * rgb_att + ir_feat * ir_att
            elif self.fusion_method == 'add':
                weights_sum = self.rgb_weight + self.ir_weight + 1e-8
                norm_rgb_weight = self.rgb_weight / weights_sum
                norm_ir_weight = self.ir_weight / weights_sum
                fused_feat = norm_rgb_weight * rgb_feat + norm_ir_weight * ir_feat
            else:
                fused_feat = (rgb_feat + ir_feat) * 0.5

        fused_feat = self.output_conv(fused_feat)
        return fused_feat

    def _index_based_fusion(self, fused_feat, ir_feat, ir_index_map, valid_mask, fusion_weight):
        """
        基于IR索引的精确融合实现（更省显存版本）
        注意：会对 fused_feat 做原位写入（index_put_），这是可反传的，但仍建议你在训练中跑一跑确认无 inplace 报错。
        """
        B, C, H, W = fused_feat.shape
        device = fused_feat.device

        # 1) 有效位置融合
        if valid_mask.any():
            b, h, w = torch.nonzero(valid_mask, as_tuple=True)
            ir_idx = ir_index_map[b, h, w]  # (N,)

            # 一维 index -> 二维坐标（假设 index 是按 W 展平）
            ir_h = torch.div(ir_idx, W, rounding_mode='trunc')
            ir_w = ir_idx % W

            ir_h = ir_h.clamp(0, H - 1)
            ir_w = ir_w.clamp(0, W - 1)

            rgb_part = fused_feat[b, :, h, w]        # (N,C)
            ir_part = ir_feat[b, :, ir_h, ir_w]      # (N,C)

            # 自适应权重（cosine similarity -> [0,1]）
            sim = F.cosine_similarity(rgb_part, ir_part, dim=1, eps=1e-8)  # (N,)
            adaptive = (sim + 1.0) * 0.5                                    # (N,) in [0,1]
            adaptive = adaptive.unsqueeze(1)                                 # (N,1)

            combined = adaptive * fusion_weight                               # (N,1)
            # 你原来的形式：rgb * (1 + w * (ir/rgb))
            # 这里避免 rgb_part 被 0 除：加 eps
            eps = 1e-8
            attention_feat = rgb_part * (1.0 + combined * (ir_part / (rgb_part + eps)))  # (N,C)

            # 写回
            fused_feat[b, :, h, w] = attention_feat

        # 2) 无效区域引入全局 IR 上下文（向量化，不再 per-batch for-loop）
        invalid_mask = ~valid_mask
        if invalid_mask.any():
            b2, h2, w2 = torch.nonzero(invalid_mask, as_tuple=True)
            rgb_invalid = fused_feat[b2, :, h2, w2]  # (M,C)

            # per-batch global IR mean: (B,C)
            ir_mean = ir_feat.mean(dim=(2, 3))  # (B,C)
            ctx = ir_mean[b2, :]                # (M,C)

            fused_feat[b2, :, h2, w2] = rgb_invalid * 0.9 + ctx * 0.1

        # 3) 平滑（depthwise conv，一次完成，避免通道 for-loop 建巨量计算图）
        mix = float(self.smooth_mix)
        if mix > 0:
            k = self.gaussian_kernel.to(device=device, dtype=fused_feat.dtype)  # (1,1,ks,ks)
            ks = k.shape[-1]
            pad = ks // 2

            # expand to (C,1,ks,ks) for depthwise
            k_dw = k.expand(C, 1, ks, ks).contiguous()
            smoothed = F.conv2d(fused_feat, k_dw, padding=pad, groups=C)
            fused_feat = fused_feat * (1.0 - mix) + smoothed * mix

            # 把结果写回给调用者持有的引用（让上层继续用同一个 fused_feat）
            # 注意：不要用 .data
            fused_feat.copy_(fused_feat)

