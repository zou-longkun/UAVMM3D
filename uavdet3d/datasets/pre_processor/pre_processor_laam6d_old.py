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
        self.local_prototypes = []
        self.dataset_cfg, self.training = dataset_cfg, training

        processor_configs = self.dataset_cfg.DATA_PRE_PROCESSOR

        self.data_processor_queue = []
        self.class_name_config = self.dataset_cfg.CLASS_NAMES

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def convert_box9d_to_heatmap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_heatmap, config=config)
        center_rad = self.dataset_cfg.CENTER_RAD
        offset = config.OFFSET
        key_point_encoder = all_object_encoders[config.ENCODER]
        encode_corner = np.array(config.ENCODER_CORNER)

        intrinsic = data_dict['intrinsic']  # (3,3)
        extrinsic = data_dict['extrinsic']  # (4,4)
        distortion = data_dict['distortion']  # (5,)

        raw_w, raw_h = data_dict['raw_im_size']
        new_w, new_h = data_dict['new_im_size']
        stride = data_dict['stride']

        box9d = data_dict['gt_boxes']  # (N, 9, 3)

        extrinsic_inv = np.linalg.inv(extrinsic[0])
        box9d = self.convert_9points_to_9params(box9d)
        centers_world = box9d[:, :3]
        centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)
        box9d[:, :3] = centers_cv

        pts3d, pts2d = key_point_encoder(
            box9d,
            encode_corner=encode_corner,
            intrinsic_mat=intrinsic,
            extrinsic_mat=extrinsic,
            distortion_matrix=distortion,
            offset=offset
        )  # pts2d: (N, num_pts, 2)

        num_obj = pts2d.shape[0]
        num_pts = pts2d.shape[1]

        H, W = new_h // stride, new_w // stride

        heatmap = torch.zeros((num_obj, num_pts, H, W), dtype=torch.float32)
        residual_x = torch.zeros_like(heatmap)
        residual_y = torch.zeros_like(heatmap)

        for ob_id in range(num_obj):
            for k in range(num_pts):
                pt = pts2d[ob_id, k].copy()

                pt[0] *= (new_w / raw_w / stride)
                pt[1] *= (new_h / raw_h / stride)

                heatmap[ob_id, k] = draw_gaussian_to_heatmap(heatmap[ob_id, k], pt, center_rad)

                res_x_np = residual_x[ob_id, k].cpu().numpy()
                res_y_np = residual_y[ob_id, k].cpu().numpy()

                res_x_np, res_y_np = draw_res_to_heatmap(res_x_np, res_y_np, pt)

                residual_x[ob_id, k] = torch.from_numpy(res_x_np)
                residual_y[ob_id, k] = torch.from_numpy(res_y_np)

        data_dict['gt_heatmap'] = heatmap  # shape: (N, K, H, W)
        data_dict['gt_res_x'] = residual_x
        data_dict['gt_res_y'] = residual_y
        data_dict['gt_pts2d'] = pts2d  # shape: (N, K, 2)
        return data_dict

    def map_name_to_index(self, gt_names, class_name_config):
        gt_indices = np.array([class_name_config.index(name) for name in gt_names], dtype=np.int32)
        return gt_indices

    def convert_box9d_to_centermap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_centermap, config=config)

        center_rad = self.dataset_cfg.CENTER_RAD
        offset = config.OFFSET
        stride = data_dict['stride']
        raw_im_size = data_dict['raw_im_size']
        new_im_size = data_dict['new_im_size']
        center_point_encoder = all_object_encoders[config.ENCODER]

        # save_dir = os.path.join(os.getcwd(), "preprocess_vis")
        # os.makedirs(save_dir, exist_ok=True)
        # seq_id = data_dict.get('seq_id', 'unknown_seq').replace('/', '_')
        # frame_id = data_dict.get('frame_id', 'unknown_frame').replace('.png', '')
        # save_prefix = f"{seq_id}_{frame_id}"

        gt_box9d = data_dict['gt_boxes']  # (N, 9, 3)
        if len(gt_box9d) == 0:
            H = new_im_size[1] // stride  # new_im_size[1]
            W = new_im_size[0] // stride  # new_im_size[0]
            data_dict['hm'] = np.zeros((len(self.class_name_config), H, W), dtype=np.float32)
            data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
            data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
            data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
            data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
            # self._save_visualization(data_dict, save_dir, save_prefix)
            return data_dict

        gt_name = data_dict['gt_names']  # (N,)
        intrinsic = data_dict['intrinsic'][0]
        extrinsic = data_dict['extrinsic'][0]
        distortion = data_dict['distortion'][0]

        extrinsic_inv = np.linalg.inv(extrinsic)
        box9d = self.convert_9points_to_9params(gt_box9d)
        # print('=' * 90)
        # print("encoder box points", gt_box9d)
        # print("encoder box params", box9d)
        if len(box9d.shape) == 1:
            box9d = box9d.reshape(1, -1)

        centers_world = box9d[:, :3]  # (N, 3)
        centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)  # (N, 3)
        box9d[:, :3] = centers_cv

        cls_index = self.map_name_to_index(gt_name, self.class_name_config)  # (N,)
        cls_index = cls_index.reshape(-1, 1)  # (N, 1)
        gt_box9d_with_cls = np.concatenate([box9d, cls_index], axis=1)  # (N, 10)

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

        # # ========== 新增：解码验证 ==========
        # if len(gt_box9d) > 0:
        #     # 调用解码函数，还原3D框
        #     pred_boxes9d, pred_conf = center_point_decoder(
        #         gt_hm,  # 编码后的热力图
        #         gt_center_res,  # 残差
        #         gt_center_dis,  # 深度
        #         gt_dim,  # 尺寸
        #         gt_rot,  # 旋转
        #         intrinsic,  # 内参
        #         extrinsic,  # 外参
        #         distortion,  # 畸变
        #         new_im_width=new_im_size[0],
        #         new_im_hight=new_im_size[1],
        #         raw_im_width=raw_im_size[0],
        #         raw_im_hight=raw_im_size[1],
        #         stride=stride,
        #         im_num=1,
        #         max_num=7
        #     )
        #     pred_boxes_params = pred_boxes9d[:, :-1]  # 移除 cls，得到 (N, 9) 的 9 参数框
        #     # 2. 转换为 9 点格式的 3D 框 (N, 9, 3)
        #     pred_box9d_9points = self.convert_9params_to_9points(pred_boxes_params)
        #     # =====================================
        #     print("decoder box points", pred_box9d_9points)
        #     print("decoder box params", pred_boxes_params)
        #     print('=' * 90)
        #     # 可视化对比：使用转换后的 9 点格式框
        #     self._visualize_encode_decode(
        #         data_dict,
        #         gt_box9d,  # 原始 9 点格式框
        #         pred_box9d_9points,  # 转换后的 9 点格式框
        #         save_dir="encode_decode_vis",
        #         save_prefix=f"{data_dict.get('seq_id')}_{data_dict.get('frame_id')}"
        #     )
        # # =====================================

        return data_dict

    def filter_box_outside(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_box_outside, config=config)

        raw_im_width, raw_im_height = data_dict['raw_im_size']
        intrinsic = data_dict['intrinsic']
        extrinsic = data_dict['extrinsic']
        distortion = data_dict['distortion']

        boxes = data_dict['gt_boxes']  # shape (N, 9, 3)
        names = data_dict['gt_names']
        diffs = data_dict.get('gt_diffs', np.zeros(len(boxes)))

        if len(boxes) > 0:
            centers = boxes[:, 0]

            is_valid = ~np.isnan(centers).any(axis=1)
            is_valid &= (np.abs(centers) < 1e9).all(axis=1)
        else:
            is_valid = np.array([], dtype=bool)

        if len(boxes) > 0 and np.any(is_valid):
            valid_boxes = boxes[is_valid]
            valid_names = names[is_valid] if len(names) > 0 else []
            valid_diffs = diffs[is_valid] if len(diffs) > 0 else []

            centers_world = valid_boxes[:, 0]
            extrinsic_inv = np.linalg.inv(extrinsic[0])

            centers_cv = self.decode_box_centers_from_world(centers_world, extrinsic_inv)  # (M, 3)

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

        assert len(data_dict['gt_boxes']) == len(data_dict['gt_names']), "gt box & name not same length"
        return data_dict

    def decode_boxes_from_world_9d(self, boxes_9d, extrinsic_inv):
        carla_to_opencv = np.array([
            [0, 1, 0, 0],  # Carla Y → OpenCV X
            [0, 0, -1, 0],  # Carla Z → OpenCV -Y
            [1, 0, 0, 0],  # Carla X → OpenCV Z
            [0, 0, 0, 1]
        ])

        all_boxes_camera_cv = []

        for box in boxes_9d:  # box shape: (9, 3)
            box_cv = []

            for pt in box:
                pt_hom = np.append(pt, 1.0)  # (4,)
                pt_cam = extrinsic_inv @ pt_hom  # 世界 → 相机
                pt_cv = carla_to_opencv @ pt_cam  # Carla 相机 → OpenCV 相机
                box_cv.append(pt_cv[:3])  # 保留前三维

            all_boxes_camera_cv.append(np.array(box_cv))  # (9, 3)

        return np.array(all_boxes_camera_cv)  # shape: (N, 9, 3)

    def decode_box_centers_from_world(self, centers_world, extrinsic_inv):
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

    def image_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalization, config=config)

        image = data_dict['image']  # (5, 3, H, W)

        for i in range(image.shape[0]):
            for c in range(image.shape[1]):
                img = image[i, c]

                if np.isnan(img).any() or np.isinf(img).any():
                    print(f" [WARNING] NaN/Inf found in modality {i}, channel {c}")
                    finite_vals = img[np.isfinite(img)]
                    max_val = np.max(finite_vals) if len(finite_vals) > 0 else 0.0
                    img = np.nan_to_num(img, nan=0.0, posinf=max_val, neginf=0.0)
                if i in [0, 1, 2]:
                    image[i, c] = img / 255.0
                elif i == 3:
                    image[i, c] = img / self.dataset_cfg.MAX_DIS
                elif i == 4:
                    max_val = np.max(img)
                    if max_val > 0:
                        image[i, c] = img / max_val
                    else:
                        image[i, c] = img

        if not np.isfinite(image).all():
            print(" [ERROR] Image still has NaN or Inf after normalization!")
            print(f"  - Min value: {np.nanmin(image)}, Max value: {np.nanmax(image)}")

        data_dict['image'] = image.astype(np.float32)
        return data_dict

    def __call__(self, data_dict):
        self.col(data_dict)
        for func in self.data_processor_queue:
            data_dict = func(data_dict)

        return data_dict

    def col(self, data_dict):
        data_dict['intrinsic'] = [data_dict['intrinsic'].astype(np.float32)]
        data_dict['extrinsic'] = [data_dict['extrinsic'].astype(np.float32)]
        data_dict['distortion'] = [data_dict['distortion'].astype(np.float32)]

    def draw_box9d_on_image_gt(self, boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
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
                cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), (255, 255, 255), 1)
            for idx, edge in enumerate(middle_edges):
                intensity = int(255 - (idx / 3) * 150)
                color = (255, 255, 255)
                cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), color, 1)
            for edge in top_edges:
                cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), (255, 255, 255), 1)

            diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]
            mid_points = []
            for i, j in diagonals:
                pt1, pt2 = box_cv[1:][i], box_cv[1:][j]
                mid = (pt1 + pt2) / 2
                mid_points.append(mid)
                proj_pts, _ = cv2.projectPoints(np.vstack([pt1, pt2]), np.eye(3), np.zeros(3), intrinsic_mat,
                                                distortion_matrix)
                p1, p2 = proj_pts.reshape(-1, 2).astype(int)
                cv2.line(image, tuple(p1), tuple(p2), (255, 255, 255), 1)

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

    # 修正：保存可视化结果的主函数，增加形状检查
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

        # 绘制原始 3D 框
        rgb_with_gt = self.draw_box9d_on_image_gt(
            gt_box9d,
            rgb_image.copy(),
            img_width=data_dict['new_im_size'][0],
            img_height=data_dict['new_im_size'][1],
            intrinsic_mat=data_dict['intrinsic'],
            extrinsic_mat=data_dict['extrinsic'],
            distortion_matrix=data_dict['distortion']
        )

        # 绘制解码后的 3D 框
        rgb_with_pred = self.draw_box9d_on_image_gt(
            pred_box9d_9points,
            rgb_image.copy(),
            img_width=data_dict['new_im_size'][0],
            img_height=data_dict['new_im_size'][1],
            intrinsic_mat=data_dict['intrinsic'],
            extrinsic_mat=data_dict['extrinsic'],
            distortion_matrix=data_dict['distortion']
        )

        # 拼接并保存
        combined = np.hstack([rgb_with_gt, rgb_with_pred])
        try:
            Image.fromarray(combined).save(full_save_path)
            print(f"已保存对比图: {full_save_path}")
        except Exception as e:
            print(f"保存对比图失败: {e}")

    def convert_9params_to_9points(self, box9d_params):
        """
        将9参数转换为9点框，与修正后的编码函数严格互逆
        局部原型为物体自身坐标系下的归一化角点，确保旋转和缩放逻辑一致
        :param box9d_params: (N, 9) 9参数数组，格式为[x,y,z,l,w,h,a1,a2,a3]
        :return: (N, 9, 3) 9点框，格式为[中心点, 角点1, 角点2, ..., 角点8]
        # """
        if len(box9d_params) != len(self.local_prototypes):
            raise ValueError("参数数量与局部原型数量不匹配，请先调用编码函数")

        all_box_points = []
        for i, params in enumerate(box9d_params):
            # 1. 解析9参数
            x, y, z = params[0:3]  # 世界坐标系下的中心点
            l, w, h = params[3:6]  # 物体自身坐标系下的尺寸（x/y/z轴方向）
            angle1, angle2, angle3 = params[6:9]  # zyx顺序欧拉角（世界坐标系下的旋转）

            # 2. 使用编码时记录的局部原型生成物体自身坐标系下的角点
            # 局部原型：物体自身坐标系下的归一化角点（已去除尺寸影响）
            local_prototype = self.local_prototypes[i]  # (8, 3)
            corners_self = local_prototype * np.array([l, w, h])  # 缩放回实际尺寸（物体坐标系下）

            # 3. 生成旋转矩阵（将物体自身坐标系转换到世界坐标系）
            rot_mat = R.from_euler('zyx', [angle1, angle2, angle3], degrees=False).as_matrix()

            # 确保旋转矩阵行列式为正（与编码逻辑一致，维持右手坐标系）
            if np.linalg.det(rot_mat) < 0:
                rot_mat[:, 2] *= -1

            # 4. 将物体自身坐标系下的角点转换到世界坐标系
            # 步骤：旋转（物体坐标系→世界坐标系）→ 平移（加上中心点）
            corners_world_local = (rot_mat @ corners_self.T).T  # 旋转后的局部坐标（相对于中心点）
            corners_world = corners_world_local + np.array([x, y, z])  # 世界坐标系绝对坐标

            # 5. 组合为9点框（中心点 + 8个世界坐标系角点）
            box9d = np.vstack([[x, y, z], corners_world])
            all_box_points.append(box9d)

        return np.stack(all_box_points)

    def convert_9points_to_9params(self, box9d_points):
        all_box_params = []
        self.local_prototypes.clear()

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

            # 4. 记录局部原型（物体坐标系下的单位尺寸角点）
            scale = np.array([l if l != 0 else 1, w if w != 0 else 1, h if h != 0 else 1])
            local_prototype = corners_self / scale  # 物体坐标系下的归一化角点
            self.local_prototypes.append(local_prototype)

            # 5. 计算欧拉角
            r = R.from_matrix(rotation_matrix)
            angles = r.as_euler('zyx')
            angle1, angle2, angle3 = angles

            box_params = np.array([x, y, z, l, w, h, angle1, angle2, angle3])
            all_box_params.append(box_params)

        return np.array(all_box_params)
