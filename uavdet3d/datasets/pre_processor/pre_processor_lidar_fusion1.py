from uavdet3d.utils.object_encoder_laam6d import all_object_encoders
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap

import torch
import cv2
from functools import partial
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R


class DataPreProcessorLidarFusion:
    def __init__(self, dataset_cfg, training: bool):
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.data_processor_queue = []
        self.class_name_config = self.dataset_cfg.CLASS_NAMES

        # 标准立方体原型（物体自身坐标系下的8个角点，归一化到[-0.5, 0.5]）
        self.standard_prototype = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ], dtype=np.float32)

        # 初始化数据处理器队列（增加健壮性：方法不存在直接报错）
        for cur_cfg in self.dataset_cfg.DATA_PRE_PROCESSOR:
            name = cur_cfg.NAME.strip() if isinstance(cur_cfg.NAME, str) else cur_cfg.NAME
            if not hasattr(self, name):
                raise AttributeError(
                    f"[DataPreProcessorLidarFusion] No method '{name}'. "
                    f"Available: {', '.join([m for m in dir(self) if not m.startswith('_')])}"
                )
            cur_processor = getattr(self, name)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def map_name_to_index(self, gt_names, class_name_config):
        if len(class_name_config) == 1:
            return np.zeros(len(gt_names), dtype=np.int32)
        return np.array([class_name_config.index(name) for name in gt_names], dtype=np.int32)

    def convert_box9d_to_heatmap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_heatmap, config=config)

        center_rad = self.dataset_cfg.CENTER_RAD
        offset = config.OFFSET
        key_point_encoder = all_object_encoders[config.ENCODER]
        encode_corner = np.array(config.ENCODER_CORNER)

        intrinsic = data_dict['rgb_intrinsic']
        extrinsic = data_dict['rgb_extrinsic']
        distortion = data_dict['distortion']
        raw_w, raw_h = data_dict['raw_im_size']
        new_w, new_h = data_dict['new_im_size']
        stride = data_dict['stride']
        box9d = data_dict['gt_boxes']  # (N, 9, 3)

        # 空GT直接返回空
        if box9d is None or len(box9d) == 0:
            data_dict['gt_heatmap'] = torch.zeros((0, 0, new_h // stride, new_w // stride), dtype=torch.float32)
            data_dict['gt_res_x'] = torch.zeros_like(data_dict['gt_heatmap'])
            data_dict['gt_res_y'] = torch.zeros_like(data_dict['gt_heatmap'])
            data_dict['gt_pts2d'] = np.zeros((0, 0, 2), dtype=np.float32)
            return data_dict

        box9d_params, _ = self.convert_9points_to_9params(box9d)
        if box9d_params is None or len(box9d_params) == 0:
            data_dict['gt_heatmap'] = torch.zeros((0, 0, new_h // stride, new_w // stride), dtype=torch.float32)
            data_dict['gt_res_x'] = torch.zeros_like(data_dict['gt_heatmap'])
            data_dict['gt_res_y'] = torch.zeros_like(data_dict['gt_heatmap'])
            data_dict['gt_pts2d'] = np.zeros((0, 0, 2), dtype=np.float32)
            return data_dict

        centers_world = box9d_params[:, :3]
        centers_cv = self.decode_box_centers_from_world(centers_world, np.linalg.inv(extrinsic))
        box9d_params[:, :3] = centers_cv

        pts3d, pts2d = key_point_encoder(
            box9d_params,
            encode_corner=encode_corner,
            intrinsic_mat=np.array([intrinsic]),
            extrinsic_mat=np.array([extrinsic]),
            distortion_matrix=np.array([distortion]),
            offset=offset
        )

        num_obj, num_pts = pts2d.shape[0], pts2d.shape[1]
        H, W = new_h // stride, new_w // stride

        heatmap = torch.zeros((num_obj, num_pts, H, W), dtype=torch.float32)
        residual_x = torch.zeros_like(heatmap)
        residual_y = torch.zeros_like(heatmap)

        # 你原来每次都 cpu->numpy->torch，这里保留 draw_res_to_heatmap 的接口，
        # 但减少不必要的临时对象
        for ob_id in range(num_obj):
            for k in range(num_pts):
                pt = pts2d[ob_id, k].copy()
                pt[0] *= (new_w / raw_w / stride)
                pt[1] *= (new_h / raw_h / stride)

                heatmap[ob_id, k] = draw_gaussian_to_heatmap(heatmap[ob_id, k], pt, center_rad)

                # residual：只把当前切片转 numpy，画完再写回
                rx = residual_x[ob_id, k].numpy()
                ry = residual_y[ob_id, k].numpy()
                rx, ry = draw_res_to_heatmap(rx, ry, pt)
                residual_x[ob_id, k] = torch.from_numpy(rx)
                residual_y[ob_id, k] = torch.from_numpy(ry)

        data_dict['gt_heatmap'] = heatmap
        data_dict['gt_res_x'] = residual_x
        data_dict['gt_res_y'] = residual_y
        data_dict['gt_pts2d'] = pts2d
        return data_dict

    def safe_center_point_encoder(
        self,
        gt_box9d_with_cls,
        intrinsic_mat,
        extrinsic_mat,
        distortion_matrix,
        new_im_width,
        new_im_hight,
        raw_im_width,
        raw_im_hight,
        stride,
        im_num,
        class_name_config,
        center_rad
    ):
        cls_num = len(class_name_config)

        gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = [], [], [], [], []
        scale_heatmap_w = 1.0 / stride
        scale_heatmap_h = 1.0 / stride

        xyz = gt_box9d_with_cls[:, 0:3].copy()

        hm_height = new_im_hight // stride
        hm_width = new_im_width // stride

        for _ in range(im_num):
            this_heat_map = torch.zeros(cls_num, hm_height, hm_width)
            this_res_map = np.zeros((2, hm_height, hm_width), dtype=np.float32)
            this_dis_map = np.zeros((1, hm_height, hm_width), dtype=np.float32)
            this_size_map = np.zeros((3, hm_height, hm_width), dtype=np.float32)
            this_angle_map = np.zeros((6, hm_height, hm_width), dtype=np.float32)

            fx, fy = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
            cx, cy = intrinsic_mat[0, 2], intrinsic_mat[1, 2]

            for obj_i, obj in enumerate(gt_box9d_with_cls):
                this_cls = int(obj[-1])
                X, Y, Z = xyz[obj_i]
                if Z <= 1e-6:
                    continue

                u = (X / Z) * fx + cx
                v = (Y / Z) * fy + cy

                u_heatmap = u * scale_heatmap_w
                v_heatmap = v * scale_heatmap_h

                if not (0 <= v_heatmap < hm_height and 0 <= u_heatmap < hm_width):
                    continue

                center_heatmap = [u_heatmap, v_heatmap]

                # 避免 in-place 影响 autograd（虽然这里本来就不需要梯度）
                hm_slice = this_heat_map[this_cls].clone()
                hm_slice = draw_gaussian_to_heatmap(hm_slice, center_heatmap, center_rad)
                this_heat_map[this_cls] = hm_slice

                rx = this_res_map[0].copy()
                ry = this_res_map[1].copy()
                rx, ry = draw_res_to_heatmap(rx, ry, center_heatmap)
                this_res_map[0] = rx
                this_res_map[1] = ry

                h_idx = int(v_heatmap)
                w_idx = int(u_heatmap)
                if 0 <= h_idx < hm_height and 0 <= w_idx < hm_width:
                    this_dis_map[0, h_idx, w_idx] = float(Z)
                    l, w, h, a1, a2, a3 = obj[3], obj[4], obj[5], obj[6], obj[7], obj[8]
                    this_size_map[:, h_idx, w_idx] = [l, w, h]
                    this_angle_map[:, h_idx, w_idx] = [
                        np.cos(a1), np.sin(a1),
                        np.cos(a2), np.sin(a2),
                        np.cos(a3), np.sin(a3),
                    ]

            # 统一转 numpy（减少临时对象）
            gt_hm.append(this_heat_map.numpy())
            gt_center_res.append(this_res_map)
            gt_center_dis.append(this_dis_map)
            gt_dim.append(this_size_map)
            gt_rot.append(this_angle_map)

        return (
            np.array(gt_hm, dtype=np.float32),
            np.array(gt_center_res, dtype=np.float32),
            np.array(gt_center_dis, dtype=np.float32),
            np.array(gt_dim, dtype=np.float32),
            np.array(gt_rot, dtype=np.float32),
        )

    def convert_box9d_to_centermap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_centermap, config=config)

        center_rad = self.dataset_cfg.CENTER_RAD
        stride = data_dict['stride']
        raw_im_size = data_dict['raw_im_size']
        new_im_size = data_dict['new_im_size']
        gt_name = data_dict['gt_names']
        gt_box9d = data_dict['gt_boxes']
        intrinsic = data_dict['rgb_intrinsic']
        extrinsic = data_dict['rgb_extrinsic']
        distortion = data_dict['distortion']

        H, W = new_im_size[1] // stride, new_im_size[0] // stride
        num_classes = len(self.class_name_config)

        if gt_box9d is None or len(gt_box9d) == 0:
            data_dict['hm'] = np.zeros((num_classes, H, W), dtype=np.float32)
            data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
            data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
            data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
            data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
            return data_dict

        box9d_params, _ = self.convert_9points_to_9params(gt_box9d)
        if box9d_params is None or box9d_params.size == 0:
            data_dict['hm'] = np.zeros((num_classes, H, W), dtype=np.float32)
            data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
            data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
            data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
            data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
            return data_dict

        centers_world = box9d_params[:, :3]
        centers_cv = self.decode_box_centers_from_world(centers_world, np.linalg.inv(extrinsic))
        box9d_params[:, :3] = centers_cv

        cls_index = self.map_name_to_index(gt_name, self.class_name_config).reshape(-1, 1)
        min_length = min(len(box9d_params), len(cls_index))
        if min_length <= 0:
            data_dict['hm'] = np.zeros((num_classes, H, W), dtype=np.float32)
            data_dict['center_res'] = np.zeros((2, H, W), dtype=np.float32)
            data_dict['center_dis'] = np.zeros((1, H, W), dtype=np.float32)
            data_dict['dim'] = np.zeros((3, H, W), dtype=np.float32)
            data_dict['rot'] = np.zeros((6, H, W), dtype=np.float32)
            return data_dict

        gt_box9d_with_cls = np.concatenate([box9d_params[:min_length], cls_index[:min_length]], axis=1)

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

        data_dict['hm'] = gt_hm
        data_dict['center_res'] = gt_center_res
        data_dict['center_dis'] = gt_center_dis / self.dataset_cfg.MAX_DIS
        data_dict['dim'] = gt_dim / self.dataset_cfg.MAX_SIZE
        data_dict['rot'] = gt_rot
        return data_dict

    def filter_box_outside(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_box_outside, config=config)

        new_im_width, new_im_height = data_dict['new_im_size']
        intrinsic = data_dict['rgb_intrinsic']
        extrinsic = data_dict['rgb_extrinsic']
        distortion = data_dict['distortion']
        boxes = data_dict['gt_boxes']
        names = data_dict['gt_names']

        if boxes is None or len(boxes) == 0:
            data_dict['gt_boxes'] = boxes
            data_dict['gt_names'] = names
            return data_dict

        centers = boxes[:, 0]
        is_valid = ~np.isnan(centers).any(axis=1) & (np.abs(centers) < 1e9).all(axis=1)
        if not np.any(is_valid):
            data_dict['gt_boxes'] = boxes[:0]
            data_dict['gt_names'] = names[:0] if len(names) > 0 else []
            return data_dict

        valid_boxes = boxes[is_valid]
        centers_world = valid_boxes[:, 0]
        centers_cv = self.decode_box_centers_from_world(centers_world, np.linalg.inv(extrinsic))

        rvec, _ = cv2.Rodrigues(np.eye(3))
        tvec = np.zeros((3, 1), dtype=np.float32)
        corners_2d, _ = cv2.projectPoints(centers_cv, rvec, tvec, intrinsic, distortion)
        corners_2d = corners_2d.reshape(-1, 2)

        mask = (
            (corners_2d[:, 0] > 0) & (corners_2d[:, 0] < new_im_width) &
            (corners_2d[:, 1] > 0) & (corners_2d[:, 1] < new_im_height)
        )

        final_mask = np.zeros(len(boxes), dtype=bool)
        final_mask[np.where(is_valid)[0]] = mask

        data_dict['gt_boxes'] = boxes[final_mask]
        data_dict['gt_names'] = names[final_mask] if len(names) > 0 else []
        return data_dict

    # 兼容配置里可能写成 filter_boxes_outside
    def filter_boxes_outside(self, data_dict=None, config=None):
        return self.filter_box_outside(data_dict, config)

    def decode_box_centers_from_world(self, centers_world, extrinsic_inv):
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        centers_world = np.asarray(centers_world, dtype=np.float32).reshape(-1, 3)
        centers_hom = np.hstack([centers_world, np.ones((centers_world.shape[0], 1), dtype=np.float32)])
        centers_cam = (extrinsic_inv @ centers_hom.T).T
        centers_cv = (carla_to_opencv @ centers_cam.T).T
        return centers_cv[:, :3].astype(np.float32)

    def convert_9points_to_9params(self, box9d_points):
        all_box_params = []
        local_prototypes = []

        if box9d_points is None or len(box9d_points) == 0:
            return np.array([]), []

        for single_box in box9d_points:
            center = single_box[0].copy()
            x, y, z = center
            corners_local = single_box[1:] - center

            H = self.standard_prototype.T @ corners_local
            U, S, Vt = np.linalg.svd(H)
            rotation_matrix = Vt.T @ U.T
            orthogonal_rot, _ = np.linalg.qr(rotation_matrix)
            if np.linalg.det(orthogonal_rot) < 0:
                orthogonal_rot[:, 2] *= -1

            r = R.from_matrix(orthogonal_rot)
            angle1, angle2, angle3 = r.as_euler('zyx')

            corners_self = (orthogonal_rot.T @ corners_local.T).T
            l = np.max(corners_self[:, 0]) - np.min(corners_self[:, 0])
            w = np.max(corners_self[:, 1]) - np.min(corners_self[:, 1])
            h = np.max(corners_self[:, 2]) - np.min(corners_self[:, 2])

            if (l * w * h) != 0:
                normalized_prototype = corners_self / np.array([l, w, h], dtype=np.float32)
            else:
                normalized_prototype = self.standard_prototype.copy()

            local_prototypes.append(normalized_prototype.astype(np.float32))
            all_box_params.append([x, y, z, l, w, h, angle1, angle2, angle3])

        return np.array(all_box_params, dtype=np.float32), local_prototypes

    def image_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalization, config=config)

        if 'image' in data_dict and isinstance(data_dict['image'], np.ndarray):
            image = data_dict['image']
            image = image.astype(np.float32, copy=False)

            # 清 NaN/Inf（避免后面归一化导致异常）
            nan_mask = np.isnan(image)
            inf_mask = np.isinf(image)
            if np.any(nan_mask) or np.any(inf_mask):
                valid = image[np.isfinite(image)]
                max_valid = float(np.max(valid)) if valid.size > 0 else 0.0
                image[nan_mask] = 0.0
                image[inf_mask & (image > 0)] = max_valid
                image[inf_mask & (image < 0)] = 0.0

            num_modalities = min(2, image.shape[0])
            image[:num_modalities] /= 255.0
            data_dict['image'] = image

        if 'fused_image' in data_dict and isinstance(data_dict['fused_image'], np.ndarray):
            fused = data_dict['fused_image'].astype(np.float32, copy=False)
            fused[:6] /= 255.0
            data_dict['fused_image'] = fused

        return data_dict

    def __call__(self, data_dict):
        # 相机参数统一 float32
        for k in ['rgb_intrinsic', 'rgb_extrinsic', 'ir_intrinsic', 'ir_extrinsic', 'distortion']:
            if k in data_dict:
                data_dict[k] = np.array(data_dict[k], dtype=np.float32)

        for func in self.data_processor_queue:
            data_dict = func(data_dict)

        return data_dict
