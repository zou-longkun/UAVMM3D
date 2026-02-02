import re
import numpy as np
import cv2
import numpy as np
import gc
import torch
import time

class LidarBasedAligner:
    """
    通过LiDAR点云构建中间对齐关系矩阵，实现IR和RGB图像的精确对齐
    该方法利用LiDAR点云分别与RGB和IR图像的映射关系，建立两者之间的转换关系
    内存优化版本：减少临时数组创建，使用生成器模式，添加显式内存清理
    """
    
    def __init__(self, min_valid_points=20, ransac_threshold=5.0):
        """
        初始化LiDAR-based对齐器
        
        参数:
            min_valid_points: 执行对齐所需的最小有效点数量
            ransac_threshold: RANSAC算法的误差阈值(像素)
        """
        self.min_valid_points = min_valid_points
        self.ransac_threshold = ransac_threshold
        self.rgb_to_ir_transform = None  # RGB到IR的转换矩阵
        self.rgb_ir_relationship_matrix = None  # 用于特征融合的关系矩阵
        
    def align_ir_to_rgb_based_on_lidar(self, lidar_points, 
                                     lidar_extrinsic, 
                                     rgb_camera_extrinsic, rgb_camera_intrinsic, rgb_image, 
                                     ir_camera_extrinsic, ir_camera_intrinsic, ir_image):
        """
        基于LiDAR点云实现IR图像与RGB图像的对齐
        优化：减少临时数据存储，优化内存使用
        
        参数:
            lidar_points: LiDAR点云数据，形状为(N, 5)，包含[x, y, z, intensity, tag]
            lidar_extrinsic: LiDAR外参矩阵，形状为(4, 4)
            rgb_camera_extrinsic: RGB相机外参矩阵，形状为(4, 4)
            rgb_camera_intrinsic: RGB相机内参矩阵，形状为(3, 3)
            rgb_image: RGB图像数据，形状为(H, W, 3)
            ir_camera_extrinsic: IR相机外参矩阵，形状为(4, 4)
            ir_camera_intrinsic: IR相机内参矩阵，形状为(3, 3)
            ir_image: IR图像数据，形状为(H, W)或(H, W, 3)
        
        返回:
            aligned_ir_image: 与RGB对齐后的IR图像
            transform_matrix: RGB到IR的变换矩阵
        """
        # 快速检查点云数量，如果点云太少直接使用备选方案
        if lidar_points.shape[0] < self.min_valid_points:
            return self._fallback_center_based_alignment(rgb_image, ir_image), None
        
        try:
            # 1. 获取LiDAR点云在RGB图像上的投影
            rgb_proj_info = list(self._project_lidar_to_image(
                lidar_points, lidar_extrinsic, 
                rgb_camera_extrinsic, rgb_camera_intrinsic, rgb_image
            ))
            
            # 如果RGB投影结果太少，直接使用备选方案
            if len(rgb_proj_info) < self.min_valid_points:
                return self._fallback_center_based_alignment(rgb_image, ir_image), None
            
            # 2. 获取LiDAR点云在IR图像上的投影
            ir_proj_info = list(self._project_lidar_to_image(
                lidar_points, lidar_extrinsic, 
                ir_camera_extrinsic, ir_camera_intrinsic, ir_image
            ))
            
            # 如果IR投影结果太少，直接使用备选方案
            if len(ir_proj_info) < self.min_valid_points:
                del rgb_proj_info
                gc.collect()
                return self._fallback_center_based_alignment(rgb_image, ir_image), None
            
            # 3. 构建RGB和IR之间的点对应关系
            rgb_ir_correspondences = self._find_corresponding_points(
                rgb_proj_info, ir_proj_info
            )
            
            # 清理投影信息，释放内存
            del rgb_proj_info, ir_proj_info
            gc.collect()
            
            if not rgb_ir_correspondences or len(rgb_ir_correspondences) < self.min_valid_points:
                # 使用中心点对齐作为备选方案
                return self._fallback_center_based_alignment(rgb_image, ir_image), None
            
            # 4. 计算RGB到IR的变换矩阵
            transform_matrix = self._estimate_transform_matrix(rgb_ir_correspondences)
            
            # 如果变换矩阵计算失败，使用备选方案
            if transform_matrix is None:
                return self._fallback_center_based_alignment(rgb_image, ir_image), None
                
            self.rgb_to_ir_transform = transform_matrix
            
            # 5. 应用变换矩阵对齐IR图像
            h, w = rgb_image.shape[:2]
            
            # 检查变换矩阵类型（透视变换为3x3，仿射变换为2x3）
            if transform_matrix.shape == (3, 3):
                # 使用透视变换
                aligned_ir_image = cv2.warpPerspective(
                    ir_image, 
                    transform_matrix, 
                    (w, h), 
                    borderMode=cv2.BORDER_CONSTANT, 
                    borderValue=0
                )
            else:
                # 使用仿射变换
                aligned_ir_image = cv2.warpAffine(
                    ir_image, 
                    transform_matrix, 
                    (w, h), 
                    borderMode=cv2.BORDER_CONSTANT, 
                    borderValue=0
                )
            
            return aligned_ir_image, transform_matrix
        except Exception as e:
            # 捕获所有异常，确保函数不会崩溃
            # 发生异常时释放所有可能的资源
            gc.collect()
            # 使用备选对齐方法
            return self._fallback_center_based_alignment(rgb_image, ir_image), None
    
    def _project_lidar_to_image(self, lidar_points, lidar_extrinsic, 
                               camera_extrinsic, camera_intrinsic, image):
        """
        将LiDAR点云投影到指定相机的图像平面上
        
        返回:
            投影信息生成器，每个元素为(u, v, x_world, y_world, z_world)
        """
        # 预分配Carla坐标系到OpenCV坐标系的转换矩阵（避免重复创建）
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        h, w = image.shape[:2]
        xyz = lidar_points[:, :3]  # 获取点的3D坐标
        
        # 计算点云规模，如果点云为空直接返回
        if xyz.shape[0] == 0:
            return []
        
        # 一次性创建LiDAR齐次坐标（减少内存分配）
        lidar_homo = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        lidar_homo[:, :3] = xyz
        lidar_homo[:, 3] = 1.0
        
        # Step 1: LiDAR → World
        points_world = (lidar_extrinsic @ lidar_homo.T).T
        
        # Step 2: World → Camera - 计算逆变换矩阵一次
        try:
            cam_ext_inv = np.linalg.inv(camera_extrinsic)
        except np.linalg.LinAlgError:
            return []  # 矩阵不可逆时返回空列表
            
        points_cam = (cam_ext_inv @ points_world.T).T
        
        # 预分配OpenCV坐标系点云（复用内存）
        points_opencv = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        points_opencv[:, :3] = points_cam[:, :3]  # 复制前3维
        points_opencv[:, 3] = 1.0  # 设置齐次坐标
        
        # Step 3: Camera → OpenCV
        points_opencv = (carla_to_opencv @ points_opencv.T).T[:, :3]
        
        # 只保留相机前方的点
        in_front = points_opencv[:, 2] > 0
        
        # 投影到图像平面
        if np.sum(in_front) > 0:
            # 只投影有效点，减少计算量
            valid_points_opencv = points_opencv[in_front]
            valid_points_world = points_world[in_front]
            
            # 投影到图像平面
            uv, _ = cv2.projectPoints(valid_points_opencv, np.eye(3), np.zeros(3), camera_intrinsic, None)
            if uv is not None:
                uv = uv.reshape(-1, 2)
                
                # 使用生成器而非列表，减少内存占用
                for i in range(uv.shape[0]):
                    u, v = uv[i]
                    u_int, v_int = int(round(u)), int(round(v))
                    # 检查点是否在图像范围内
                    if 0 <= u_int < w and 0 <= v_int < h:
                        x_world, y_world, z_world = valid_points_world[i, :3]
                        yield u, v, x_world, y_world, z_world
        
        # 显式释放不再需要的大数组内存
        del lidar_homo, points_world, points_cam, points_opencv
        gc.collect()
    
    def _find_corresponding_points(self, rgb_proj_info, ir_proj_info, device='cuda'):
        """
        基于LiDAR点云的世界坐标，寻找RGB和IR图像上的对应点（PyTorch优化版）
        利用张量运算替代循环和字典，提升速度
        
        参数:
            rgb_proj_info: RGB图像投影信息生成器/列表，每个元素为(u, v, x, y, z)
            ir_proj_info: IR图像投影信息生成器/列表，每个元素为(u, v, x, y, z)
            device: 计算设备（'cuda'或'cpu'）
        
        返回:
            对应点张量，形状为[N, 4]，每行是(rgb_u, rgb_v, ir_u, ir_v)
        """
        start_time = time.time()
        
        # 1. 将投影信息转换为PyTorch张量（向量化存储）
        # 格式: [N, 5]，每行为(u, v, x, y, z)
        rgb_tensor = torch.tensor(list(rgb_proj_info), dtype=torch.float32, device=device)
        ir_tensor = torch.tensor(list(ir_proj_info), dtype=torch.float32, device=device)
        
        # 若任一投影信息为空，直接返回空张量
        if rgb_tensor.numel() == 0 or ir_tensor.numel() == 0:
            print(f"_find_corresponding_points execution time: {(time.time() - start_time) * 1000:.2f} ms")
            return torch.empty((0, 4), device=device)
        
        # 2. 提取世界坐标(x, y, z)并量化（保留1位小数，与原逻辑一致）
        # 量化公式: round(x, 1) = floor(x * 10 + 0.5) / 10
        rgb_world = torch.round(rgb_tensor[:, 2:5] * 10) / 10  # [N_rgb, 3]
        ir_world = torch.round(ir_tensor[:, 2:5] * 10) / 10    # [N_ir, 3]
        
        # 3. 用集合运算寻找匹配的世界坐标（替代字典查找）
        # 为了快速匹配，将3D坐标转换为1D哈希值（减少维度）
        # 注意：哈希系数需根据坐标范围调整，确保唯一性
        hash_coeff = torch.tensor([1e6, 1e3, 1.0], device=device)  # 哈希系数
        rgb_hash = torch.sum(rgb_world * hash_coeff, dim=1)  # [N_rgb]
        ir_hash = torch.sum(ir_world * hash_coeff, dim=1)    # [N_ir]
        
        # 找到两边都存在的哈希值（交集）
        combined_hash = torch.cat([rgb_hash, ir_hash])
        unique_hash, counts = torch.unique(combined_hash, return_counts=True)
        common_hash = unique_hash[counts == 2]  # 只保留两边都有的哈希值
        
        # 4. 提取匹配点的坐标（向量化操作）
        # 找到RGB中匹配的索引
        rgb_mask = torch.isin(rgb_hash, common_hash)
        rgb_matched = rgb_tensor[rgb_mask]  # [N_match, 5]：包含(u, v, x, y, z)
        rgb_matched_hash = rgb_hash[rgb_mask]
        
        # 找到IR中匹配的索引
        ir_mask = torch.isin(ir_hash, common_hash)
        ir_matched = ir_tensor[ir_mask]  # [N_match, 5]
        ir_matched_hash = ir_hash[ir_mask]
        
        # 5. 按哈希值排序并对齐（确保匹配顺序一致）
        rgb_sorted_idx = torch.argsort(rgb_matched_hash)
        ir_sorted_idx = torch.argsort(ir_matched_hash)
        
        rgb_sorted = rgb_matched[rgb_sorted_idx]  # 按哈希值排序
        ir_sorted = ir_matched[ir_sorted_idx]     # 按哈希值排序
        
        # 6. 限制最大匹配点数量（与原逻辑一致）
        max_correspondences = min(2000, rgb_sorted.shape[0], ir_sorted.shape[0])
        rgb_sorted = rgb_sorted[:max_correspondences]
        ir_sorted = ir_sorted[:max_correspondences]
        
        # 7. 组合结果：(rgb_u, rgb_v, ir_u, ir_v)
        correspondences = torch.cat([
            rgb_sorted[:, 0:2],  # RGB的u, v
            ir_sorted[:, 0:2]   # IR的u, v
        ], dim=1)  # [N, 4]
        
        # 打印执行时间（GPU加速下通常比CPU快10-100倍）
        print(f"_find_corresponding_points execution time: {(time.time() - start_time) * 1000:.2f} ms")
        return correspondences

    
    def _find_corresponding_points_(self, rgb_proj_info, ir_proj_info):
        """
        基于LiDAR点云的世界坐标，寻找RGB和IR图像上的对应点
        优化：减少字典大小，使用更少的精度存储坐标键
        
        返回:
            对应点列表，每个元素为(rgb_u, rgb_v, ir_u, ir_v)
        """
        # 记录开始时间
        start_time = time.time()
        
        # 将生成器转换为列表以便重复访问
        rgb_proj_list = list(rgb_proj_info)
        ir_proj_list = list(ir_proj_info)
        
        # 如果任一投影信息为空，直接返回空列表
        if not rgb_proj_list or not ir_proj_list:
            # 打印执行时间
            print(f"_find_corresponding_points execution time: {(time.time() - start_time) * 1000:.2f} ms")
            return []
        
        # 优化：使用更紧凑的数据结构存储点云映射
        # 只保留世界坐标的整数部分作为键，减少字典大小
        rgb_world_to_pixel = {}
        for info in rgb_proj_list:
            u, v, x, y, z = info
            # 只保留到小数点后1位，减少键的数量和内存占用
            key = (round(x, 1), round(y, 1), round(z, 1))
            rgb_world_to_pixel[key] = (u, v)
        
        # 寻找对应点
        correspondences = []
        # 限制匹配点数量，避免过多点导致的内存消耗
        max_correspondences = min(2000, len(rgb_proj_list), len(ir_proj_list))
        count = 0
        
        for info in ir_proj_list:
            if count >= max_correspondences:
                break
                
            u_ir, v_ir, x, y, z = info
            key = (round(x, 1), round(y, 1), round(z, 1))
            if key in rgb_world_to_pixel:
                u_rgb, v_rgb = rgb_world_to_pixel[key]
                correspondences.append((u_rgb, v_rgb, u_ir, v_ir))
                count += 1
        
        # 清理临时列表，释放内存
        del rgb_proj_list, ir_proj_list
        gc.collect()
        
        # 打印执行时间
        print(f"_find_corresponding_points execution time: {(time.time() - start_time) * 1000:.2f} ms")
        
        return correspondences
    
    def _estimate_transform_matrix(self, correspondences):
        """
        使用RANSAC算法估计RGB到IR的透视变换矩阵（考虑视角差）
        优化：减少矩阵大小，优化计算过程
        
        返回:
            变换矩阵，形状为(3, 3)或(2, 3)，取决于使用的变换类型
        """
        # 限制用于计算的点数，避免创建过大的矩阵
        max_points_for_matrix = 5000  # 限制最大点数为5000以控制内存使用
        num_points = min(len(correspondences), max_points_for_matrix)
        
        # 如果点数太少，直接返回None
        if num_points < 4:
            return None
            
        # 随机采样一部分点，减少计算量和内存使用
        if num_points < len(correspondences):
            indices = np.random.choice(len(correspondences), num_points, replace=False)
            sampled_correspondences = [correspondences[i] for i in indices]
        else:
            sampled_correspondences = correspondences
        
        # 准备输入数据
        rgb_points = np.array([(c[0], c[1]) for c in sampled_correspondences], dtype=np.float32)
        ir_points = np.array([(c[2], c[3]) for c in sampled_correspondences], dtype=np.float32)
        
        # 尝试使用透视变换（Homography）
        try:
            # 调整参数以减少计算复杂度
            homography, mask = cv2.findHomography(
                rgb_points, 
                ir_points, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=self.ransac_threshold, 
                maxIters=1000,  # 减少迭代次数以降低内存使用
                confidence=0.99
            )
            
            if homography is not None:
                # 清理临时数组
                del rgb_points, ir_points, sampled_correspondences
                gc.collect()
                return homography
        except Exception as e:
            # 发生异常时静默处理，直接尝试仿射变换
            pass
        
        # 如果透视变换失败，尝试仿射变换，但使用更高效的实现
        try:
            # 使用cv2.estimateAffinePartial2D替代手动构造矩阵，更高效
            transform_matrix, mask = cv2.estimateAffinePartial2D(
                rgb_points, 
                ir_points, 
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=1000,
                confidence=0.99
            )
            
            # 清理临时数组
            del rgb_points, ir_points, sampled_correspondences
            gc.collect()
            
            return transform_matrix
        except Exception as e:
            # 如果所有方法都失败，返回None
            del rgb_points, ir_points, sampled_correspondences
            gc.collect()
            return None
    
    def _fallback_center_based_alignment(self, rgb_image, ir_image):
        """
        基于中心点的对齐方法作为备选方案
        当LiDAR点不足时使用
        """
        h_rgb, w_rgb = rgb_image.shape[:2]
        h_ir, w_ir = ir_image.shape[:2]
        
        # 计算图像中心点
        center_rgb = (w_rgb // 2, h_rgb // 2)
        center_ir = (w_ir // 2, h_ir // 2)
        
        # 计算平移量
        shift_u = center_rgb[0] - center_ir[0]
        shift_v = center_rgb[1] - center_ir[1]
        
        # 创建平移矩阵
        M = np.float32([[1, 0, shift_u], [0, 1, shift_v]])
        
        # 执行平移
        aligned_ir = cv2.warpAffine(
            ir_image, 
            M, 
            (w_rgb, h_rgb), 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )
        
        return aligned_ir
    
    def visualize_alignment(self, rgb_image, original_ir, aligned_ir, save_path=None):
        """
        可视化对齐效果
        优化：减少内存使用，添加显式资源清理
        """
        try:
            import matplotlib.pyplot as plt
            
            # 确保IR图像是3通道的
            if len(original_ir.shape) == 2:
                original_ir = cv2.cvtColor(original_ir, cv2.COLOR_GRAY2BGR)
            if len(aligned_ir.shape) == 2:
                aligned_ir = cv2.cvtColor(aligned_ir, cv2.COLOR_GRAY2BGR)
            
            # 创建对比图，设置较小的figsize减少内存使用
            plt.figure(figsize=(12, 4))  # 缩小尺寸以减少内存占用
            
            plt.subplot(131)
            plt.title('RGB图像')
            plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(132)
            plt.title('原始IR图像')
            plt.imshow(cv2.cvtColor(original_ir, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(133)
            plt.title('LiDAR对齐后的IR图像')
            plt.imshow(cv2.cvtColor(aligned_ir, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                print(f"对齐效果可视化已保存到: {save_path}")
            
            plt.show()
            
            # 显式清理Matplotlib资源
            plt.close('all')
            gc.collect()
        except Exception as e:
            print(f"可视化时出错: {e}")
            # 发生异常时清理资源
            try:
                plt.close('all')
            except:
                pass
            gc.collect()