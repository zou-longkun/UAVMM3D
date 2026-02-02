from collections import defaultdict
from uavdet3d.datasets import DatasetTemplate
import numpy as np
import os
import torch
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib
import random
from glob import glob
from PIL import Image
from .ads_metric_laam6d import LAA3D_ADS_Metric
from .dataset_utils import convert_9params_to_9points, convert_box_opencv_to_world, xyz_to_uv, \
    register_images_by_center, project_lidar_and_get_uvz_rgb_tag, generate_world_coords_map, \
    radar_to_velocity_heatmap, evaluation_by_name, draw_box9d_on_image, draw_box9d_on_image_gt,\
    visualize_multi_modal_align_old

try:
    # 尝试用 tkAGG（有界面环境可用）
    matplotlib.use('tkAGG')
except ImportError:
    # 无界面环境 fallback 到 Agg
    matplotlib.use('Agg')


class LAAM6D_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training, root_path, logger):
        super(LAAM6D_Det_Dataset, self).__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)

        self.dataset_cfg = dataset_cfg
        self.root_path = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.training = training
        self.logger = logger
        self.im_num = dataset_cfg.IM_NUM
        self.lidar_offset = dataset_cfg.LIDAR_OFFSET
        self.radar_offset = dataset_cfg.RADAR_OFFSET

        self.sequence_frames = {}
        self.modalities = ['rgb', 'ir', 'dvs']
        self.intrinsics = {}
        self.extrinsics = {}
        self.distortions = {}
        self.default_distortion = np.zeros(5, dtype=np.float32)

        self.im_path_names = ['images_rgb', 'images_ir', 'images_dvs']
        self.label_path_names = ['boxes_rgb', 'boxes_ir', 'boxes_dvs']

        self.raw_im_width = dataset_cfg.IM_SIZE[0]
        self.raw_im_hight = dataset_cfg.IM_SIZE[1]
        self.new_im_width = dataset_cfg.IM_RESIZE[0]
        self.new_im_hight = dataset_cfg.IM_RESIZE[1]
        self.obj_size = np.array(self.dataset_cfg.OB_SIZE)
        self.stride = self.dataset_cfg.STRIDE
        self.sorted_namelist = self.dataset_cfg.CLASS_NAMES
        self.set_split(0.8)

    def __len__(self):

        return len(self.infos)

    def __getitem__(self, item):
        each_info = self.infos[item]

        im_paths = each_info['im_paths']
        label_paths = each_info['label_paths']
        im_info_path = each_info['im_info_path']
        cls_name = each_info['cls_name']
        seq_id = each_info['seq_id']
        frame_id = each_info['frame_id']

        lidar_path = each_info.get('lidar_path', None)
        if lidar_path is not None and os.path.exists(lidar_path):
            lidar_data = np.load(lidar_path)  # (N, 5): x, y, z, intensity, tag
            assert lidar_data.shape[1] == 5, f"{lidar_data.shape} is not (N,5)"
        else:
            self.logger.warning(f"LiDAR file missing for frame: {frame_id}")
            lidar_data = np.empty((0, 5), dtype=np.float32)
        radar_path = each_info.get('radar_path', None)
        if radar_path is not None and os.path.exists(radar_path):
            radar_data = np.load(radar_path)  # (N, 5)
        else:
            self.logger.warning(f"Radar file missing for frame: {frame_id}")
            radar_data = np.empty((0, 5), dtype=np.float32)

        with open(im_info_path, 'rb') as f:
            camera_info = pickle.load(f)

        intrinsic = np.array(camera_info['rgb']['intrinsic'])
        extrinsic = np.array(camera_info['rgb']['extrinsic'])
        image_modal_stack = []

        box_file = label_paths['rgb']
        gt_boxes, gt_names = [], []

        center_world_cal = []
        if os.path.exists(box_file):
            with open(box_file, 'rb') as f:
                raw_data = pickle.load(f)

            for row in raw_data:
                if isinstance(row[0], str):
                    raw_name = row[0]
                    pts = np.array(row[1:], dtype=np.float32).reshape(8, 3)
                else:
                    raw_name = "Unknown"
                    pts = np.array(row, dtype=np.float32).reshape(8, 3)

                pts_world = convert_box_opencv_to_world(pts, extrinsic)

                center_world = ((pts_world[0] + pts_world[6]) +
                                (pts_world[1] + pts_world[7]) +
                                (pts_world[2] + pts_world[4]) +
                                (pts_world[3] + pts_world[5])
                                ) / 8.0
                center_world = center_world.reshape(1, 3)
                center_world_cal.append(center_world)
                box_9pts_world = np.concatenate([center_world, pts_world], axis=0)
                gt_boxes.append(box_9pts_world)

                matched = "Unknown"
                for cls in cls_name:
                    if cls in raw_name:
                        matched = cls
                        break
                gt_names.append(matched)
            assert len(gt_boxes) == len(gt_names), f"Boxes/names mismatch: {len(gt_boxes)} vs {len(gt_names)}"
            gt_boxes = np.array(gt_boxes, dtype=np.float32)  # (N, 9, 3)
            gt_names = np.array(gt_names)
            assert len(gt_boxes) == len(gt_names), f"Boxes/names mismatch: {len(gt_boxes)} vs {len(gt_names)}"
        else:
            self.logger.warning(f"No label file for RGB: {box_file}")
            gt_boxes = np.empty((0, 9, 3), dtype=np.float32)
            gt_names = np.empty((0,), dtype='<U32')

        distortion = self.default_distortion.copy()

        intrinsic_rgb = np.array(camera_info['rgb']['intrinsic'])
        extrinsic_rgb = np.array(camera_info['rgb']['extrinsic'])
        intrinsic_ir = np.array(camera_info['ir']['intrinsic'])
        extrinsic_ir = np.array(camera_info['ir']['extrinsic'])
        intrinsic_dvs = np.array(camera_info['dvs']['intrinsic'])
        extrinsic_dvs = np.array(camera_info['dvs']['extrinsic'])

        center_uv_rgb = xyz_to_uv(center_world_cal, self.raw_im_width, self.raw_im_hight, intrinsic_rgb,
                                  extrinsic_rgb, distortion)
        center_uv_ir = xyz_to_uv(center_world_cal, self.raw_im_width, self.raw_im_hight, intrinsic_ir,
                                 extrinsic_ir, distortion)
        center_uv_dvs = xyz_to_uv(center_world_cal, self.raw_im_width, self.raw_im_hight, intrinsic_dvs,
                                  extrinsic_dvs, distortion)

        rgb_image = cv2.imread(im_paths['rgb'], cv2.IMREAD_COLOR)
        ir_img = cv2.imread(im_paths['ir'], cv2.IMREAD_GRAYSCALE)
        dvs_img = cv2.imread(im_paths['dvs'], cv2.IMREAD_COLOR)

        registered = register_images_by_center(
            rgb_image, ir_img, dvs_img,
            center_uv_rgb, center_uv_ir, center_uv_dvs,
            intrinsic_rgb, intrinsic_ir, intrinsic_dvs
        )

        # 获取配准后的图像和内参
        aligned_rgb = registered['rgb']
        aligned_ir = registered['ir']
        aligned_dvs = registered['dvs']
        aligned_intrinsics_rgb = registered['intrinsics']['rgb']
        aligned_intrinsics_ir = registered['intrinsics']['ir']
        aligned_intrinsics_dvs = registered['intrinsics']['dvs']

        # 计算缩放比例
        # 原始配准后图像的尺寸
        orig_height, orig_width = aligned_rgb.shape[:2]
        # 目标尺寸
        target_width, target_height = self.new_im_width, self.new_im_hight

        # 计算x和y方向的缩放因子
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height

        # 缩放图像
        rgb_image = cv2.resize(aligned_rgb, (target_width, target_height))
        ir_img = cv2.resize(aligned_ir, (target_width, target_height))
        dvs_img = cv2.resize(aligned_dvs, (target_width, target_height))

        # 更新RGB内参
        resized_intrinsics_rgb = aligned_intrinsics_rgb.copy()
        resized_intrinsics_rgb[0, 0] *= scale_x  # fx
        resized_intrinsics_rgb[1, 1] *= scale_y  # fy
        resized_intrinsics_rgb[0, 2] *= scale_x  # cx
        resized_intrinsics_rgb[1, 2] *= scale_y  # cy

        # 更新IR内参
        resized_intrinsics_ir = aligned_intrinsics_ir.copy()
        resized_intrinsics_ir[0, 0] *= scale_x
        resized_intrinsics_ir[1, 1] *= scale_y
        resized_intrinsics_ir[0, 2] *= scale_x
        resized_intrinsics_ir[1, 2] *= scale_y

        # 更新DVS内参
        resized_intrinsics_dvs = aligned_intrinsics_dvs.copy()
        resized_intrinsics_dvs[0, 0] *= scale_x
        resized_intrinsics_dvs[1, 1] *= scale_y
        resized_intrinsics_dvs[0, 2] *= scale_x
        resized_intrinsics_dvs[1, 2] *= scale_y

        intrinsic = resized_intrinsics_rgb

        for mode in self.modalities:
            if mode == 'rgb':
                img = rgb_image
            elif mode == 'ir':
                img = ir_img
            elif mode == 'dvs':
                img = dvs_img
            else:
                img = cv2.imread(im_paths[mode], cv2.IMREAD_COLOR)
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

        lidar_proj_info = project_lidar_and_get_uvz_rgb_tag(
            lidar_data,
            lidar_extrinsic=np.array(each_info['lidar_extrinsic']),
            camera_extrinsic=extrinsic,
            camera_intrinsic=intrinsic,
            image=rgb_image
        )  # shape (M, 10): [u, v, z, x_world, y_world, z_world, r, g, b, tag]

        world_coords_map = generate_world_coords_map(
            lidar_proj_info,
            image_shape=rgb_image.shape[:2]
        )
        world_coords_tensor = world_coords_map.transpose(2, 0, 1)  # (3, H, W)
        world_coords_tensor = world_coords_tensor.astype(np.float32)

        image_modal_stack.append(world_coords_tensor)

        radar_velocity_heatmap = radar_to_velocity_heatmap(
            radar_data=radar_data,
            radar_extrinsic=np.array(each_info['radar_extrinsic']),
            camera_extrinsic=extrinsic,
            camera_intrinsic=intrinsic,
            image_shape=rgb_image.shape[:2]
        )
        radar_velocity_heatmap = np.repeat(radar_velocity_heatmap, 3, axis=0)
        image_modal_stack.append(radar_velocity_heatmap)

        image = np.stack(image_modal_stack, axis=0)  # shape: (5, 3, H, W)

        with open(im_info_path, 'rb') as f:
            camera_info = pickle.load(f)

        data_dict = {
            'image': image,  # (5, 3, H, W)
            'gt_boxes': gt_boxes,  # (N, 9, 3)
            'gt_names': gt_names,  # (N,)
            'intrinsic': intrinsic,  # (3, 3)
            'extrinsic': extrinsic,  # (4, 4)
            'distortion': distortion,  # (5,)
            'raw_im_size': np.array([self.raw_im_width, self.raw_im_hight]),
            'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': np.array(self.dataset_cfg.OB_SIZE),
            'seq_id': seq_id,
            'frame_id': frame_id,
            'stride': self.stride,
            'sorted_namelist': self.sorted_namelist
        }
        data_dict = self.data_pre_processor(data_dict)

        # ====================================multimodal_visualizations_with_gt=========================================
        # velocity_img = radar_velocity_heatmap[0, :, :]
        # if velocity_img.max() != velocity_img.min():
        #     velocity_img = (velocity_img - velocity_img.min()) / (velocity_img.max() - velocity_img.min()) * 255
        # else:
        #     velocity_img = np.zeros_like(velocity_img)
        # velocity_img = velocity_img.astype(np.uint8)
        # rgb_with_gt = draw_box9d_on_image_gt(
        #     gt_boxes,
        #     rgb_image.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # ir_with_gt = draw_box9d_on_image_gt(
        #     gt_boxes,
        #     ir_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # dvs_with_gt = draw_box9d_on_image_gt(
        #     gt_boxes,
        #     dvs_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # velicity_with_gt = draw_box9d_on_image_gt(
        #     gt_boxes,
        #     velocity_img.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # world_coords_resized_with_gt = draw_box9d_on_image_gt(
        #     gt_boxes,
        #     world_coords_map.copy(),
        #     img_width=self.new_im_width,
        #     img_height=self.new_im_hight,
        #     intrinsic_mat=intrinsic,
        #     extrinsic_mat=extrinsic,
        #     distortion_matrix=distortion
        # )
        # self.vis_save_dir = os.path.join(os.getcwd(), 'multimodal_visualizations_with_gt')
        # os.makedirs(self.vis_save_dir, exist_ok=True)
        # save_filename = f"{seq_id.replace('/', '_')}_{frame_id.replace('.png', '.jpg')}"
        # save_path = os.path.join(self.vis_save_dir, save_filename)

        # visualize_multi_modal_align_old(
        #     rgb_image=rgb_with_gt,
        #     ir_image=ir_with_gt,
        #     dvs_image=dvs_with_gt,
        #     velocity_image=velicity_with_gt,
        #     world_coords_map=world_coords_resized_with_gt,
        #     world_coords_map_original=world_coords_map,
        #     save_path=save_path
        # )
        # ==============================================================================================================

        return data_dict

    def include_CARLA_data(self, mode):
        self.logger.info('Loading CARLA dataset')
        CARLA_infos = []

        from collections import defaultdict
        seq_groups = defaultdict(list)  # {seq_name: [(frame_name1), (frame_name2), ...], ...}
        for seq_name, frame_name in self.sample_scene_list:
            seq_groups[seq_name].append(frame_name)

        for seq_name, frame_list in seq_groups.items():
            total_frames_in_seq = len(frame_list)

            valid_end = total_frames_in_seq - max(self.lidar_offset, self.radar_offset)
            if valid_end <= 0:
                self.logger.warning(
                    f"序列 {seq_name} 帧数量不足（共{total_frames_in_seq}帧，偏移量{max(self.lidar_offset, self.radar_offset)}），跳过该序列")
                continue

            base_path = os.path.join(self.root_path, seq_name)
            for i in range(0, valid_end, self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                curr_frame = frame_list[i]

                lidar_frame = frame_list[i + self.lidar_offset]
                radar_frame = frame_list[i + self.radar_offset]

                im_paths = {}
                for mode_name in self.modalities:
                    image_dir = f'images_{mode_name}'
                    image_path = os.path.join(base_path, image_dir, curr_frame)
                    if not os.path.exists(image_path):
                        self.logger.warning(f"序列 {seq_name} 未找到图像: {image_path}")
                    im_paths[mode_name] = image_path

                label_paths = {}
                for mode_name in self.modalities:
                    label_dir = f'boxes_{mode_name}'
                    label_path = os.path.join(base_path, label_dir, curr_frame.replace('.png', '.pkl'))
                    if not os.path.exists(label_path):
                        self.logger.warning(f"序列 {seq_name} 未找到标签: {label_path}")
                    label_paths[mode_name] = label_path

                lidar_path = os.path.join(base_path, 'lidar_1', lidar_frame.replace('.png', '.npy'))
                radar_path = os.path.join(base_path, 'radar_1', radar_frame.replace('.png', '.npy'))

                im_info_path = os.path.join(base_path, 'im_info.pkl')
                lidar_radar_info_path = os.path.join(base_path, 'lidar_radar_info.pkl')
                if not os.path.exists(lidar_radar_info_path):
                    self.logger.warning(f"序列 {seq_name} 未找到外参文件: {lidar_radar_info_path}")
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

                weather_dirs = [
                    w for w in os.listdir(seq_path)
                    if os.path.isdir(os.path.join(seq_path, w))
                ]
                for weather_name in weather_dirs:
                    weather_path = os.path.join(seq_path, weather_name)
                    drone_dirs = [
                        d for d in os.listdir(weather_path)
                        if os.path.isdir(os.path.join(weather_path, d))
                    ]
                    for drone_name in drone_dirs:
                        # full_path = f"{map_name}/carla_data/{seq_id}/{weather_name}/{drone_name}"
                        full_path = f"{map_name}/carla_data/{seq_id}/clear_day/{drone_name}"
                        # full_path = f"Town01_Opt/carla_data/00001/clear_day/m210-rtk"
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
        super(LAAM6D_Det_Dataset, self).__init__(
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
        data_dict = defaultdict(list)
        for i, cur_sample in enumerate(batch_list):
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key == 'image':
                    merged = np.stack(val, axis=0)  # (B, 5, 3, H, W)
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
                elif key == 'gt_diffs':
                    max_gt = max([v.shape[0] for v in val])
                    merged = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for i in range(batch_size):
                        cur_len = val[i].shape[0]
                        merged[i, :cur_len] = val[i]
                    ret[key] = merged
                elif key in ['intrinsic', 'extrinsic', 'distortion']:
                    stacked = np.stack(val, axis=0)
                    stacked = stacked.squeeze(1)
                    ret[key] = stacked
                elif key in ['raw_im_size', 'new_im_size', 'obj_size']:
                    stacked = np.stack(val, axis=0)
                    if key == 'obj_size':
                        stacked = stacked.squeeze(1)
                    ret[key] = stacked
                elif key in ['seq_id', 'frame_id']:
                    ret[key] = val
                elif key == 'stride':
                    ret[key] = val
                elif key == 'sorted_namelist':
                    continue
                elif key in ['hm', 'center_res', 'center_dis', 'dim', 'rot']:
                    try:
                        if key == 'hm':
                            val = [v.squeeze(0) if v.ndim == 4 else v for v in val]
                            # print(f"hm shapes before pad: {[v.shape for v in val]}")
                            max_cls = max([v.shape[0] for v in val])
                            padded_val = []
                            for v in val:
                                # print(f"current hm shape: {v.shape}, ndim: {v.ndim}")
                                if v.shape[0] < max_cls:
                                    pad_width = ((0, max_cls - v.shape[0]), (0, 0), (0, 0))
                                    v = np.pad(v, pad_width, mode='constant')
                                padded_val.append(v)
                            stacked = np.stack(padded_val, axis=0)
                        else:
                            val = [v.squeeze(0) if v.ndim == 4 else v for v in val]
                            # print(f"{key} shapes before stack: {[v.shape for v in val]}")
                            stacked = np.stack(val, axis=0)

                        ret[key] = stacked
                    except Exception as e:
                        raise e
                elif key in ['gt_heatmap', 'gt_res_x', 'gt_res_y', 'gt_pts2d']:
                    sample_dim = val[0].ndim
                    max_num_obj = max([v.shape[0] for v in val])
                    pad_width = []
                    for dim in range(sample_dim):
                        if dim == 0:
                            pad_width.append((0, max_num_obj - val[0].shape[0]))
                        else:
                            pad_width.append((0, 0))
                    pad_width = tuple(pad_width)
                    padded_val = []
                    for v in val:
                        if v.ndim != sample_dim:
                            raise ValueError(f"{v.ndim} vs {sample_dim}")
                        cur_pad = max_num_obj - v.shape[0]
                        if cur_pad > 0:
                            cur_pad_width = list(pad_width)
                            cur_pad_width[0] = (0, cur_pad)
                            v_padded = np.pad(v, tuple(cur_pad_width), mode='constant', constant_values=0)
                            padded_val.append(v_padded)
                        else:
                            padded_val.append(v)
                    stacked = np.stack(padded_val, axis=0)
                    ret[key] = stacked
                else:
                    print(f"Unknown key '{key}' ignored")

            except Exception as e:
                print(f"Error in collate_batch for key={key}: {e}")
                raise e
        ret['batch_size'] = batch_size
        ret['sorted_namelist'] = self.dataset_cfg.CLASS_NAMES
        # print("="*50)
        # print(f"batch_size：{batch_size}")
        # for key in ret:
        #     if isinstance(ret[key], np.ndarray):
        #         print(f"  {key}: {ret[key].shape}")
        #     elif isinstance(ret[key], list):
        #         print(f"  {key}: 列表，长度={len(ret[key])}")
        #     else:
        #         print(f"  {key}: {type(ret[key])}")
        # print("="*50)
        # print("collate_batch finished\n")
        return ret

    def name_from_code(self, name_indices):
        names = [self.dataset_cfg.CLASS_NAMES[int(x)] for x in name_indices]
        return np.array(names)

    def generate_prediction_dicts(self, batch_dict, output_path):
        batch_size = batch_dict['batch_size']
        annos = []

        base_debug_dir = os.path.join(os.getcwd(), 'debug_images')
        os.makedirs(base_debug_dir, exist_ok=True)

        # 仅在本地调试时启用显示（通过环境变量控制，默认关闭:export ENABLE_IMSHOW=true）
        enable_imshow = os.environ.get('ENABLE_IMSHOW', 'False').lower() == 'true'

        for batch_id in range(batch_size):
            seq_id = batch_dict['seq_id'][batch_id]
            frame_id = batch_dict['frame_id'][batch_id]

            raw_im_size = batch_dict['raw_im_size'][batch_id]  # (2,)
            obj_size = batch_dict['obj_size'][batch_id]
            intrinsic = batch_dict['intrinsic'][batch_id]  # (3, 3)
            extrinsic = batch_dict['extrinsic'][batch_id]  # (4, 4)
            distortion = batch_dict['distortion'][batch_id]  # (5,)
            gt_boxes = batch_dict['gt_boxes'][batch_id]  # (N, 9, 3)
            gt_names = batch_dict['gt_names'][batch_id]  # (N, )

            # [center_x, center_y, center_z, l, w, h, a1, a2, a3, class_id]
            pred_boxes9d = batch_dict['pred_boxes9d'][batch_id]  # (N, 10)
            pred_class_ids = pred_boxes9d[:, -1].astype(int)
            pred_names = self.name_from_code(pred_class_ids)
            pred_boxes9d = convert_9params_to_9points(pred_boxes9d[:, :-1])  # world -> world, (N, 9, 3)

            confidence = batch_dict['confidence'][batch_id]  # (N,)
            sorted_namelist = batch_dict['sorted_namelist'][batch_id]
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
            #     pred_boxes9d, image,
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
            #         gt_box9d, image,
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

    def evaluation_all(self, annos, metric_root_path):

        laa_ads_fun = LAA3D_ADS_Metric(eval_config=self.dataset_cfg.LAA3D_ADS_METRIC,
                                       classes=self.sorted_namelist,
                                       metric_save_path=metric_root_path)

        final_str = laa_ads_fun.eval(annos)

        return final_str

    def evaluation_private(self, annos, metric_root_path):

        MetricPrivate = self.dataset_cfg.MetricPrivate

        setting_mask = MetricPrivate.SetttingMask
        wheather_mask = MetricPrivate.WheatherMask
        brightness_mask = MetricPrivate.BrightnessMask
        drop_rain = MetricPrivate.DropRain

        new_annos = []

        for info in annos:
            setting_id = info['setting_id']
            seq_id = info['seq_id']
            brightness = info['brightness']
            frame_id = info['frame_id']

            if len(setting_mask) > 0:
                setting_flag = False
                for each_name in setting_mask:
                    if each_name in setting_id:
                        setting_flag = True
            else:
                setting_flag = True

            if len(wheather_mask) > 0:
                wheather_flag = False
                for each_name in wheather_mask:
                    if each_name in seq_id:
                        wheather_flag = True
            else:
                wheather_flag = True

            if drop_rain and 'rain' in seq_id:
                drop_rain_flag = False
            else:
                drop_rain_flag = True

            if brightness_mask[0] < brightness < brightness_mask[1]:
                brightness_flag = True
            else:
                brightness_flag = False

            if setting_flag and wheather_flag and drop_rain_flag and brightness_flag:
                new_annos.append(info)

        laa_ads_fun = LAA3D_ADS_Metric(eval_config=self.dataset_cfg.LAA3D_ADS_METRIC,
                                       classes=self.sorted_namelist,
                                       metric_save_path=metric_root_path)

        final_str = laa_ads_fun.eval(annos)

        return final_str

    def evaluation(self, annos, metric_root_path):

        print('evaluating !!!')

        i_str = self.evaluation_all(annos, metric_root_path)

        return i_str

    def evaluation_(self, annos, metric_root_path):
        all_names = self.sorted_namelist
        final_str = ''
        print(f"[DEBUG] Total annotations: {len(annos)}, Available classes: {all_names}")

        for cls_n in all_names:
            cls_annos = []
            print(f"\n[DEBUG] Processing class: {cls_n}")

            for i, each_anno in enumerate(annos):
                gt_boxes = each_anno['gt_boxes']  # (N,9,3)
                pred_boxes = each_anno['pred_boxes']
                gt_names = each_anno['gt_names']  # (N,)
                pred_names = each_anno['pred_names']

                print(f"\nAnno {i}:")
                print(f"  gt_boxes shape: {gt_boxes.shape}")
                print(f"  pred_boxes shape: {pred_boxes.shape}")
                print(f"  gt_names: {gt_names}")
                print(f"  pred_names: {pred_names}")
                if len(gt_boxes) > 0:
                    gt_centers = gt_boxes[:, 0, :]
                    print(f"  所有真实框中心点 ({len(gt_centers)} 个):")
                    for idx in range(len(gt_centers)):
                        x, y, z = gt_centers[idx]
                        print(f"    真实框 {idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

                if len(pred_boxes) > 0:
                    pred_centers = pred_boxes[:, 0, :]
                    print(f"  所有预测框中心点 ({len(pred_centers)} 个):")
                    for idx in range(len(pred_centers)):
                        x, y, z = pred_centers[idx]
                        print(f"    预测框 {idx}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

                if len(gt_boxes) > len(gt_names):
                    if len(gt_boxes) > 0:
                        print("  gt_boxes content:")
                        for box_idx in range(gt_boxes.shape[0]):
                            print(f"    Box {box_idx}:")
                            for point_idx in range(gt_boxes.shape[1]):
                                x, y, z = gt_boxes[box_idx, point_idx]
                                print(f"      Point {point_idx:2d}: x={x:8.4f}, y={y:8.4f}, z={z:8.4f}")
                    else:
                        print("  gt_boxes content: [No boxes]")
                    gt_boxes = gt_boxes[:len(gt_names)]

                mask_gt = (gt_names == cls_n)
                mask_pred = (pred_names == cls_n)

                if mask_gt.sum() > 0:
                    filtered_gt = gt_boxes[mask_gt]
                    filtered_names = gt_names[mask_gt]

                    cls_annos.append({
                        'gt_boxes': filtered_gt,
                        'gt_names': filtered_names,
                        'pred_boxes': pred_boxes[mask_pred],
                        'pred_names': pred_names[mask_pred]
                    })

            if len(cls_annos) >= 2:
                final_str += evaluation_by_name(cls_annos, metric_root_path, cls_name=cls_n,
                                                max_ass_dis=self.dataset_cfg.MAX_DIS)

        merged_annos = []
        for each_anno in annos:
            gt_boxes = each_anno['gt_boxes']
            pred_boxes = each_anno['pred_boxes']

            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()

            if len(gt_boxes.shape) != 3 or gt_boxes.shape[1:] != (9, 3):
                print(f"[警告] 跳过非9点格式的gt_boxes，形状: {gt_boxes.shape}")
                continue
            if len(pred_boxes.shape) != 3 or pred_boxes.shape[1:] != (9, 3):
                print(f"[警告] 跳过非9点格式的pred_boxes，形状: {pred_boxes.shape}")
                continue

            if len(gt_boxes) > 0:
                merged_annos.append({
                    'gt_boxes': gt_boxes,
                    'gt_names': each_anno['gt_names'],
                    'pred_boxes': pred_boxes,
                    'pred_names': each_anno['pred_names']
                })

        if merged_annos:
            final_str += evaluation_by_name(merged_annos, metric_root_path, cls_name='all',
                                            max_ass_dis=self.dataset_cfg.MAX_DIS)
        else:
            print("merged_annos empty")

        return final_str
