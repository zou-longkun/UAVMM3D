from collections import defaultdict, OrderedDict
from uavdet3d.datasets import DatasetTemplate
import numpy as np
import os
import torch
import cv2
import pickle
import matplotlib
import random
from glob import glob
from PIL import Image

from .ads_metric_laam6d import LAA3D_ADS_Metric
from .dataset_utils import (
    convert_9params_to_9points, convert_9points_to_9params, convert_box9d_to_box_param,
    convert_box_opencv_to_world, radar_to_velocity_heatmap, draw_box9d_on_image, draw_box9d_on_image_gt
)

try:
    matplotlib.use('tkAGG')
except ImportError:
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
        self.im_path_names = {'rgb': 'images_rgb', 'ir': 'images_ir', 'dvs': 'images_dvs'}

        # 其他参数
        self.im_resize = dataset_cfg.get('IM_RESIZE')
        self.max_dis = dataset_cfg.get('MAX_DIS')
        self.max_size = dataset_cfg.get('MAX_SIZE')
        self.obj_size = np.array(dataset_cfg.OB_SIZE)
        self.stride = dataset_cfg.get('STRIDE')
        self.sorted_namelist = dataset_cfg.get('CLASS_NAMES')

        # 默认畸变参数
        self.default_distortion = np.zeros(5, dtype=np.float32)

        # 先 split
        self.set_split(0.9)
        self.logger.info("Total samples: {}".format(len(self.infos)))

        # 图像尺寸
        self.raw_im_width, self.raw_im_hight = self.dataset_cfg.get('IM_SIZE')
        self.new_im_width, self.new_im_hight = self.im_resize
        self.scale_x = self.new_im_width / self.raw_im_width
        self.scale_y = self.new_im_hight / self.raw_im_hight

        # ===================== 内存控制开关 =====================
        self.precompute_correspondences = dataset_cfg.get('PRECOMPUTE_CORRESPONDENCES', True)

        # 是否缓存对应点（建议 True，但一定要限制）
        self.cache_correspondences = dataset_cfg.get('CACHE_CORRESPONDENCES', True)
        self.cache_size_limit = int(dataset_cfg.get('CORR_CACHE_SIZE', 200))  # 建议小一点更安全
        self.max_corr_points = int(dataset_cfg.get('MAX_CORR_POINTS', 4096))  # 对应点上限(下采样)

        # 是否保留 dense 投影图（world_coords_map/depth_map/velocity_heatmap），默认 False 防止 batch 变巨
        self.keep_projection_maps = dataset_cfg.get('KEEP_PROJECTION_MAPS', False)

        # 是否保留 fused_image_visual（H,W,6），默认 False
        self.keep_fused_vis = dataset_cfg.get('KEEP_FUSED_VIS', False)

        # LRU 缓存
        self.correspondence_cache = OrderedDict()

    def __len__(self):
        return len(self.infos)

    # -------------------- LRU cache helpers --------------------
    def _lru_cache_get(self, key):
        if not self.cache_correspondences:
            return None
        if key not in self.correspondence_cache:
            return None
        self.correspondence_cache.move_to_end(key)
        return self.correspondence_cache[key]

    def _lru_cache_put(self, key, value):
        if not self.cache_correspondences:
            return
        if key in self.correspondence_cache:
            self.correspondence_cache.move_to_end(key)
            self.correspondence_cache[key] = value
        else:
            self.correspondence_cache[key] = value
            self.correspondence_cache.move_to_end(key)

        while len(self.correspondence_cache) > self.cache_size_limit:
            self.correspondence_cache.popitem(last=False)

    # -------------------- labels loader --------------------
    def _load_labels(self, label_path, extrinsic, cls_name_list):
        gt_boxes, gt_names = [], []
        if not os.path.exists(label_path):
            return np.empty((0, 9, 3), dtype=np.float32), np.empty((0,), dtype='<U32')

        with open(label_path, 'rb') as f:
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
                            (pts_world[3] + pts_world[5])) / 8.0
            center_world = center_world.reshape(1, 3)
            box_9pts_world = np.concatenate([center_world, pts_world], axis=0)
            gt_boxes.append(box_9pts_world)

            matched = "Unknown"
            for cls in cls_name_list:
                if cls in raw_name:
                    matched = cls
                    break
            gt_names.append(matched)

        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        gt_names = np.array(gt_names)
        return gt_boxes, gt_names

    # -------------------- main getitem --------------------
    def __getitem__(self, index):
        # ✅ 改：避免递归，最多重试 N 次
        max_retry = 20

        for _ in range(max_retry):
            each_info = self.infos[index]
            seq_id = each_info['seq_id']
            frame_id = each_info['frame_id']

            # 读取图像
            rgb_image = cv2.imread(each_info['im_paths']['rgb'], cv2.IMREAD_COLOR)
            ir_img = cv2.imread(each_info['im_paths']['ir'], cv2.IMREAD_COLOR)
            dvs_img = cv2.imread(each_info['im_paths']['dvs'], cv2.IMREAD_COLOR)

            if rgb_image is None or ir_img is None or dvs_img is None:
                self.logger.warning(f"Image load failed: {seq_id}_{frame_id}")
                index = (index + 1) % len(self)
                continue

            rgb_image = cv2.resize(rgb_image, (self.new_im_width, self.new_im_hight))
            ir_img = cv2.resize(ir_img, (self.new_im_width, self.new_im_hight))
            dvs_img = cv2.resize(dvs_img, (self.new_im_width, self.new_im_hight))

            # 堆叠三模态
            image_modal_stack = []
            for mode in self.modalities:
                img = rgb_image if mode == 'rgb' else (ir_img if mode == 'ir' else dvs_img)
                img = img.astype(np.float32).transpose(2, 0, 1)
                image_modal_stack.append(img)
            image = np.stack(image_modal_stack, axis=0)  # (3,3,H,W)

            # LiDAR/Radar
            try:
                lidar_data = np.load(each_info['lidar_path'])
                radar_data = np.load(each_info['radar_path'])
                assert lidar_data.ndim == 2 and lidar_data.shape[1] == 5, "LiDAR format error"
                assert radar_data.ndim == 2 and radar_data.shape[1] == 5, "Radar format error"
            except Exception as e:
                self.logger.error(f"Sample {seq_id}_{frame_id} loading failed: {e}")
                index = (index + 1) % len(self)
                continue

            if radar_data is None or len(lidar_data) == 0:
                index = (index + 1) % len(self)
                continue

            # 相机参数
            with open(each_info['im_info_path'], 'rb') as f:
                camera_info = pickle.load(f)

            intrinsic_rgb = np.array(camera_info['rgb']['intrinsic'], dtype=np.float32)
            extrinsic_rgb = np.array(camera_info['rgb']['extrinsic'], dtype=np.float32)
            intrinsic_ir = np.array(camera_info['ir']['intrinsic'], dtype=np.float32)
            extrinsic_ir = np.array(camera_info['ir']['extrinsic'], dtype=np.float32)
            intrinsic_dvs = np.array(camera_info['dvs']['intrinsic'], dtype=np.float32)
            extrinsic_dvs = np.array(camera_info['dvs']['extrinsic'], dtype=np.float32)

            def resize_intr(intri):
                out = intri.copy()
                out[0, 0] *= self.scale_x
                out[1, 1] *= self.scale_y
                out[0, 2] *= self.scale_x
                out[1, 2] *= self.scale_y
                return out

            resized_intrinsics_rgb = resize_intr(intrinsic_rgb)
            resized_intrinsics_ir = resize_intr(intrinsic_ir)
            resized_intrinsics_dvs = resize_intr(intrinsic_dvs)

            lidar_extrinsic = each_info['lidar_extrinsic']
            radar_extrinsic = each_info['radar_extrinsic']
            distortion = self.default_distortion.copy()

            # labels
            rgb_label_path = each_info['label_paths']['rgb']
            ir_label_path = each_info['label_paths']['ir']
            dvs_label_path = each_info['label_paths']['dvs']

            rgb_gt_boxes, rgb_gt_names = self._load_labels(rgb_label_path, extrinsic_rgb, each_info['cls_name'])
            ir_gt_boxes, ir_gt_names = self._load_labels(ir_label_path, extrinsic_ir, each_info['cls_name'])
            dvs_gt_boxes, dvs_gt_names = self._load_labels(dvs_label_path, extrinsic_dvs, each_info['cls_name'])

            # 投影
            rgb_radar_projection = self.project_radar_to_image(radar_data, resized_intrinsics_rgb, extrinsic_rgb, radar_extrinsic)
            rgb_lidar_projection = self.project_lidar_to_image(lidar_data, resized_intrinsics_rgb, extrinsic_rgb, lidar_extrinsic)

            ir_radar_projection = self.project_radar_to_image(radar_data, resized_intrinsics_ir, extrinsic_ir, radar_extrinsic)
            ir_lidar_projection = self.project_lidar_to_image(lidar_data, resized_intrinsics_ir, extrinsic_ir, lidar_extrinsic)

            # ✅ 默认不保留 dense map（防止 batch 巨大）
            if not self.keep_projection_maps:
                for proj in [rgb_lidar_projection, ir_lidar_projection]:
                    proj.pop('world_coords_map', None)

            # swap
            if self.dataset_cfg.get('SWAP_RGB_IR', True):
                image[0], image[1] = image[1].copy(), image[0].copy()
                rgb_image, ir_img = ir_img, rgb_image

                resized_intrinsics_rgb, resized_intrinsics_ir = resized_intrinsics_ir, resized_intrinsics_rgb
                extrinsic_rgb, extrinsic_ir = extrinsic_ir, extrinsic_rgb

                rgb_gt_boxes, ir_gt_boxes = ir_gt_boxes, rgb_gt_boxes
                rgb_gt_names, ir_gt_names = ir_gt_names, rgb_gt_names

                rgb_radar_projection, ir_radar_projection = ir_radar_projection, rgb_radar_projection
                rgb_lidar_projection, ir_lidar_projection = ir_lidar_projection, rgb_lidar_projection

            # correspondences
            correspondences = None
            if (self.precompute_correspondences and
                rgb_lidar_projection is not None and ir_lidar_projection is not None and
                len(rgb_lidar_projection.get('image_points', [])) > 0 and
                len(ir_lidar_projection.get('image_points', [])) > 0):

                cache_key = f"{seq_id}_{frame_id}"
                cached = self._lru_cache_get(cache_key)
                if cached is not None:
                    correspondences = cached
                else:
                    rgb_proj_info = [
                        (u, v, x, y, z)
                        for (u, v), (x, y, z, *_) in zip(
                            rgb_lidar_projection['image_points'],
                            rgb_lidar_projection['world_coords_data']
                        )
                    ]
                    ir_proj_info = [
                        (u, v, x, y, z)
                        for (u, v), (x, y, z, *_) in zip(
                            ir_lidar_projection['image_points'],
                            ir_lidar_projection['world_coords_data']
                        )
                    ]

                    corr_tensor = self._find_corresponding_points(
                        rgb_proj_info, ir_proj_info,
                        gt_boxes=rgb_gt_boxes,
                        rgb_image=None, ir_image=None,
                        visualize=False, save_path=None,
                        ir_gt_boxes=None
                    )

                    corr_np = corr_tensor.detach().cpu().numpy().astype(np.float16)

                    # ✅ 下采样，避免单帧对应点超大
                    if corr_np.shape[0] > self.max_corr_points:
                        idx = np.random.choice(corr_np.shape[0], self.max_corr_points, replace=False)
                        corr_np = corr_np[idx]

                    correspondences = corr_np  # (M,4) float16
                    self._lru_cache_put(cache_key, correspondences)

            data_dict = {
                'image': image,
                'gt_boxes': rgb_gt_boxes,
                'gt_names': rgb_gt_names,
                'rgb_gt_boxes': rgb_gt_boxes,
                'ir_gt_boxes': ir_gt_boxes,
                'rgb_intrinsic': resized_intrinsics_rgb,
                'ir_intrinsic': resized_intrinsics_ir,
                'rgb_extrinsic': extrinsic_rgb,
                'ir_extrinsic': extrinsic_ir,
                'intrinsic': resized_intrinsics_rgb,
                'extrinsic': extrinsic_rgb,
                'distortion': distortion,
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
                'precomputed_correspondences': correspondences
            }

            # 原图层融合（默认不返回 fused_image_visual）
            if correspondences is not None:
                rgb_image_original = cv2.imread(each_info['im_paths']['rgb'], cv2.IMREAD_COLOR)
                ir_image_original = cv2.imread(each_info['im_paths']['ir'], cv2.IMREAD_COLOR)
                if rgb_image_original is not None and ir_image_original is not None:
                    rgb_image_processed = cv2.resize(rgb_image_original, (self.new_im_width, self.new_im_hight))
                    ir_image_processed = cv2.resize(ir_image_original, (self.new_im_width, self.new_im_hight))

                    fused_image = self._fuse_images_at_original(
                        rgb_image_processed, ir_image_processed, correspondences
                    ).astype(np.float32)

                    if self.keep_fused_vis:
                        data_dict['fused_image_visual'] = fused_image  # (H,W,6)

                    data_dict['fused_image'] = fused_image.transpose(2, 0, 1)  # (6,H,W)

            # data preprocessor
            data_dict = self.data_pre_processor(data_dict)
            return data_dict

        raise RuntimeError(f"Too many invalid samples encountered starting from index={index}")

    # -------------------- dataset building --------------------
    def include_CARLA_data(self, mode):
        self.logger.info('Loading CARLA dataset')
        CARLA_infos = []

        seq_groups = defaultdict(list)
        for seq_name, frame_name in self.sample_scene_list:
            seq_groups[seq_name].append(frame_name)

        for seq_name, frame_list in seq_groups.items():
            total_frames_in_seq = len(frame_list)

            valid_end = total_frames_in_seq - max(self.LIDAR_OFFSET, self.RADAR_OFFSET)
            if valid_end <= 0:
                self.logger.warning(
                    f"Sequence {seq_name} has insufficient frames (total {total_frames_in_seq}, offset {max(self.LIDAR_OFFSET, self.RADAR_OFFSET)}), skipping"
                )
                continue

            base_path = os.path.join(self.root_path, seq_name)
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
                    lidar_extrinsic = np.eye(4, dtype=np.float32)
                    radar_extrinsic = np.eye(4, dtype=np.float32)
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
        map_dirs = sorted([d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))])

        for map_name in map_dirs:
            carla_data_root = os.path.join(self.root_path, map_name, "carla_data")
            if not os.path.exists(carla_data_root):
                continue

            seq_dirs = sorted(glob(os.path.join(carla_data_root, "0000*")))
            for seq_path in seq_dirs:
                seq_id = os.path.basename(seq_path)
                if seq_id not in self.dataset_cfg.get('SEQ_IDS'):
                    continue

                weather_dirs = [w for w in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, w))]
                for weather_name in weather_dirs:
                    if weather_name not in self.dataset_cfg.get('WEATHER_NAMES'):
                        continue

                    weather_path = os.path.join(seq_path, weather_name)
                    drone_dirs = [d for d in os.listdir(weather_path) if os.path.isdir(os.path.join(weather_path, d))]
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

    # -------------------- collate_batch (keep your fusion version) --------------------
    def collate_batch(self, batch_list, _unused=False):
        valid_samples = []
        for sample in batch_list:
            corr = sample.get('precomputed_correspondences')
            if corr is not None:
                try:
                    corr_array = np.array(corr, dtype=np.float32)
                    if corr_array.size > 0 and corr_array.ndim == 2 and corr_array.shape[1] == 4:
                        valid_samples.append(sample)
                except:
                    continue

        if len(valid_samples) == 0:
            print("Warning: No valid samples in batch, returning empty batch")
            ret = {'batch_size': 0, 'sorted_namelist': self.dataset_cfg.CLASS_NAMES}
            return ret

        data_dict = defaultdict(list)
        for cur_sample in valid_samples:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(valid_samples)
        ret = {}

        for key, val in data_dict.items():
            if key == 'image':
                ret[key] = np.stack(val, axis=0)
            elif key == 'gt_boxes':
                max_gt = max([v.shape[0] for v in val])
                merged = np.full((batch_size, max_gt, 9, 3), np.nan, dtype=np.float32)
                for i in range(batch_size):
                    cur_len = val[i].shape[0]
                    merged[i, :cur_len] = val[i]
                ret[key] = merged
            elif key == 'gt_names':
                ret[key] = val
            elif key in ['rgb_intrinsic', 'ir_intrinsic', 'rgb_extrinsic', 'ir_extrinsic', 'intrinsic', 'extrinsic']:
                ret[key] = np.stack(val, axis=0)
            elif key in ['raw_im_size', 'new_im_size', 'obj_size', 'stride', 'lidar_extrinsic', 'radar_extrinsic']:
                ret[key] = np.stack(val, axis=0) if len(val) > 0 else val
            elif key in ['seq_id', 'frame_id', 'distortion']:
                ret[key] = val
            elif key == 'lidar_points':
                max_points = max([v.shape[0] for v in val])
                merged = np.zeros((batch_size, max_points, 3), dtype=np.float32)
                for i in range(batch_size):
                    num_points = val[i].shape[0]
                    if num_points > 0:
                        merged[i, :num_points] = val[i][:, :3]
                ret[key] = merged
            elif key in ['rgb_radar_projection', 'rgb_lidar_projection', 'ir_radar_projection', 'ir_lidar_projection']:
                # 这里保持 list（避免 dict 被 stack 成 object 数组）
                ret[key] = val
            elif key == 'precomputed_correspondences':
                max_corr_length = max([len(c) for c in val])
                padded = []
                for corr in val:
                    corr_array = np.array(corr, dtype=np.float32)
                    if corr_array.shape[0] < max_corr_length:
                        pad = np.zeros((max_corr_length - corr_array.shape[0], 4), dtype=np.float32)
                        corr_array = np.vstack((corr_array, pad))
                    else:
                        corr_array = corr_array[:max_corr_length]
                    padded.append(corr_array)
                ret[key] = np.stack(padded, axis=0)
            else:
                # 其他键尽量 stack；stack 失败就保持 list
                try:
                    ret[key] = np.stack(val, axis=0)
                except:
                    ret[key] = val

        ret['batch_size'] = batch_size
        ret['sorted_namelist'] = self.dataset_cfg.CLASS_NAMES
        return ret

    # -------------------- misc utils --------------------
    def name_from_code(self, name_indices):
        if len(self.dataset_cfg.CLASS_NAMES) == 1:
            return np.array([self.dataset_cfg.CLASS_NAMES[0]] * len(name_indices))
        names = [self.dataset_cfg.CLASS_NAMES[int(x)] for x in name_indices]
        return np.array(names)

    def generate_prediction_dicts(self, batch_dict, output_path):
        batch_size = batch_dict['batch_size']
        annos = []

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

        return annos

    def evaluation(self, annos, metric_root_path):
        print('evaluating !!!')
        return self.evaluation_all(annos, metric_root_path)

    def evaluation_all(self, annos, metric_root_path):
        laa_ads_fun = LAA3D_ADS_Metric(
            eval_config=self.dataset_cfg.LAA3D_ADS_METRIC,
            classes=self.sorted_namelist,
            metric_save_path=metric_root_path
        )
        return laa_ads_fun.eval(annos)

    # -------------------- radar projection --------------------
    def project_radar_to_image(self, radar_data, intrinsic, extrinsic, radar_extrinsic=None):
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

        velocity = radar_data[:, 0]
        azim = radar_data[:, 1]
        alt = radar_data[:, 2]
        depth = radar_data[:, 3]

        x = depth * np.cos(alt) * np.cos(azim)
        y = depth * np.cos(alt) * np.sin(azim)
        z = depth * np.sin(alt)
        radar_xyz = np.stack([x, y, z], axis=1)

        radar_homo = np.hstack([radar_xyz, np.ones((radar_xyz.shape[0], 1))])
        if radar_extrinsic is None:
            radar_extrinsic = np.eye(4)
        pts_world = (radar_extrinsic @ radar_homo.T).T[:, :3]

        world_to_camera = np.linalg.inv(extrinsic) if extrinsic.shape == (4, 4) else np.eye(4)
        world_homo = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
        camera_points_carla = (world_to_camera @ world_homo.T).T[:, :3]

        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        camera_points_carla_hom = np.hstack([camera_points_carla, np.ones((len(camera_points_carla), 1))])
        camera_points_opencv = (carla_to_opencv @ camera_points_carla_hom.T).T[:, :3]

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

        valid_camera_points_opencv = camera_points_opencv[valid_z_indices]
        valid_velocity = velocity[valid_z_indices]
        valid_camera_points_carla = camera_points_carla[valid_z_indices]
        valid_radar_xyz = radar_xyz[valid_z_indices]

        image_points, _ = cv2.projectPoints(
            valid_camera_points_opencv,
            np.zeros(3),
            np.zeros(3),
            intrinsic,
            None
        )

        image_points = image_points.squeeze().astype(np.float32)
        if len(image_points.shape) == 1:
            image_points = np.expand_dims(image_points, axis=0)

        valid_indices = (
            (image_points[:, 0] >= 0) & (image_points[:, 0] < self.new_im_width) &
            (image_points[:, 1] >= 0) & (image_points[:, 1] < self.new_im_hight)
        )

        if not np.any(valid_indices):
            return {
                'image_points': np.zeros((0, 2), dtype=np.float32),
                'camera_points_opencv': np.zeros((0, 3), dtype=np.float32),
                'camera_points_carla': np.zeros((0, 3), dtype=np.float32),
                'velocity': np.zeros((0,), dtype=np.float32),
                'radar_xyz': np.zeros((0, 3), dtype=np.float32),
                'radar_velocity_heatmap': radar_velocity_heatmap
            }

        valid_img_pts = image_points[valid_indices]
        valid_vels = valid_velocity[valid_indices]

        y_coords = np.round(valid_img_pts[:, 1]).astype(int)
        x_coords = np.round(valid_img_pts[:, 0]).astype(int)
        in_bounds = (x_coords >= 0) & (x_coords < self.new_im_width) & (y_coords >= 0) & (y_coords < self.new_im_hight)

        if np.any(in_bounds):
            valid_idx = np.where(in_bounds)[0]
            valid_y = y_coords[valid_idx]
            valid_x = x_coords[valid_idx]
            valid_vel_values = valid_vels[valid_idx].astype(float)
            radar_velocity_heatmap[:, valid_y, valid_x] = valid_vel_values

        return {
            'image_points': image_points[valid_indices],
            'camera_points_opencv': valid_camera_points_opencv[valid_indices],
            'camera_points_carla': valid_camera_points_carla[valid_indices],
            'velocity': valid_velocity[valid_indices],
            'radar_xyz': valid_radar_xyz[valid_indices],
            'radar_velocity_heatmap': radar_velocity_heatmap
        }

    # -------------------- lidar projection (z-buffer version) --------------------
    def project_lidar_to_image(self, lidar_data, intrinsic, extrinsic, lidar_extrinsic=None):
        world_coords_map = np.zeros((3, self.new_im_hight, self.new_im_width), dtype=np.float32)
        depth_zbuf = np.full((self.new_im_hight, self.new_im_width), np.inf, dtype=np.float32)
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

        lidar_points = lidar_data[:, :3]

        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        ones = np.ones((lidar_points.shape[0], 1), dtype=np.float32)
        lidar_points_hom = np.hstack((lidar_points.astype(np.float32), ones))

        if lidar_extrinsic is not None:
            world_points_hom = (lidar_extrinsic @ lidar_points_hom.T).T
        else:
            world_points_hom = lidar_points_hom

        world_to_camera = np.linalg.inv(extrinsic) if extrinsic.shape == (4, 4) else np.eye(4, dtype=np.float32)

        camera_points_carla = (world_to_camera @ world_points_hom.T).T[:, :3]
        camera_points_carla_hom = np.hstack([camera_points_carla, np.ones((len(camera_points_carla), 1), dtype=np.float32)])
        camera_points_opencv = (carla_to_opencv @ camera_points_carla_hom.T).T[:, :3]

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

        valid_camera_points_opencv = camera_points_opencv[valid_z_indices]
        valid_lidar_data = lidar_data[valid_z_indices]
        valid_camera_points_carla = camera_points_carla[valid_z_indices]

        image_points, _ = cv2.projectPoints(
            valid_camera_points_opencv,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            intrinsic.astype(np.float32),
            None
        )

        image_points = image_points.squeeze().astype(np.float32)
        if len(image_points.shape) == 1:
            image_points = np.expand_dims(image_points, axis=0)

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

        valid_world_points = world_points_hom[valid_z_indices][valid_indices][:, :3]

        world_coords_data = np.zeros_like(valid_lidar_data[valid_indices])
        world_coords_data[:, :3] = valid_world_points
        world_coords_data[:, 3:] = valid_lidar_data[valid_indices][:, 3:]

        valid_img_points = image_points[valid_indices]
        y_coords = np.round(valid_img_points[:, 1]).astype(int)
        x_coords = np.round(valid_img_points[:, 0]).astype(int)

        in_bounds = (
            (x_coords >= 0) & (x_coords < self.new_im_width) &
            (y_coords >= 0) & (y_coords < self.new_im_hight)
        )

        if np.any(in_bounds):
            valid_idx = np.where(in_bounds)[0]
            valid_y = y_coords[valid_idx]
            valid_x = x_coords[valid_idx]

            valid_world = valid_world_points[valid_idx].astype(np.float32)
            valid_depths = valid_camera_points_opencv[valid_indices][valid_idx, 2].astype(np.float32)

            for j in range(valid_depths.shape[0]):
                y = int(valid_y[j])
                x = int(valid_x[j])
                z = float(valid_depths[j])

                if z < float(depth_zbuf[y, x]):
                    depth_zbuf[y, x] = z
                    world_coords_map[0, y, x] = float(valid_world[j, 0])
                    world_coords_map[1, y, x] = float(valid_world[j, 1])
                    world_coords_map[2, y, x] = float(valid_world[j, 2])

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

    # -------------------- correspondences --------------------
    def _find_corresponding_points(self, rgb_proj_info, ir_proj_info, gt_boxes,
                                  rgb_image=None, ir_image=None, visualize=False, save_path=None, ir_gt_boxes=None):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        rgb_list = list(rgb_proj_info)
        ir_list = list(ir_proj_info)

        rgb_world_to_pixel = {}
        for (u, v, x, y, z) in rgb_list:
            rgb_world_to_pixel[(x, y, z)] = (u, v)

        correspondences_list = []
        correspondences_world = []
        for u_ir, v_ir, x, y, z in ir_list:
            if (x, y, z) in rgb_world_to_pixel:
                u_rgb, v_rgb = rgb_world_to_pixel[(x, y, z)]
                correspondences_list.append([u_rgb, v_rgb, u_ir, v_ir])
                correspondences_world.append([x, y, z])

        if correspondences_list:
            correspondences = torch.tensor(correspondences_list, dtype=torch.float32, device=device)
            correspondences_world_np = np.array(correspondences_world, dtype=np.float32)
        else:
            correspondences = torch.zeros((0, 4), dtype=torch.float32, device=device)
            correspondences_world_np = np.zeros((0, 3), dtype=np.float32)

        if visualize and rgb_image is not None and ir_image is not None:
            self._visualize_correspondences(
                rgb_image, ir_image, correspondences, save_path,
                correspondences_world_np=correspondences_world_np,
                gt_boxes=gt_boxes, ir_gt_boxes=ir_gt_boxes
            )

        return correspondences

    # -------------------- fusion --------------------
    def _fuse_images_at_original(self, rgb_image, ir_image, correspondences):
        h, w = rgb_image.shape[:2]
        fused_image = np.zeros((h, w, 6), dtype=np.float32)
        fused_image[:, :, :3] = rgb_image.astype(np.float32)

        if correspondences is None:
            return fused_image

        if isinstance(correspondences, list):
            corr_np = np.array(correspondences)
        else:
            corr_np = np.asarray(correspondences)

        if corr_np.size == 0:
            return fused_image

        rgb_u_coords = np.clip(corr_np[:, 0].astype(int), 0, w - 1)
        rgb_v_coords = np.clip(corr_np[:, 1].astype(int), 0, h - 1)
        ir_u_coords = np.clip(corr_np[:, 2].astype(int), 0, w - 1)
        ir_v_coords = np.clip(corr_np[:, 3].astype(int), 0, h - 1)

        for i in range(len(corr_np)):
            rgb_u, rgb_v = rgb_u_coords[i], rgb_v_coords[i]
            ir_u, ir_v = ir_u_coords[i], ir_v_coords[i]
            fused_image[rgb_v, rgb_u, 3:] = ir_image[ir_v, ir_u].astype(np.float32)

        return fused_image

    # -------------------- visualization (kept) --------------------
    def _visualize_correspondences(self, rgb_image, ir_image, correspondences, save_path=None,
                                   correspondences_world_np=None, gt_boxes=None, ir_gt_boxes=None):
        filtered_correspondences = correspondences

        if gt_boxes is not None and len(gt_boxes) > 0 and correspondences_world_np is not None:
            def point_in_boxes(point, boxes):
                for box in boxes:
                    x_min, y_min, z_min, x_max, y_max, z_max = box
                    if (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max and z_min <= point[2] <= z_max):
                        return True
                return False

            def convert_boxes_format(boxes):
                converted = []
                for box in boxes:
                    x_min = np.min(box[:, 0]); y_min = np.min(box[:, 1]); z_min = np.min(box[:, 2])
                    x_max = np.max(box[:, 0]); y_max = np.max(box[:, 1]); z_max = np.max(box[:, 2])
                    converted.append([x_min, y_min, z_min, x_max, y_max, z_max])
                return converted

            converted_gt_boxes = convert_boxes_format(gt_boxes)
            valid_indices = []
            for i, wp in enumerate(correspondences_world_np):
                if point_in_boxes(wp, converted_gt_boxes):
                    valid_indices.append(i)

            if valid_indices:
                valid_indices = list(set(valid_indices))
                filtered_correspondences = correspondences[valid_indices]

        if not isinstance(rgb_image, np.ndarray):
            rgb_image = np.array(rgb_image)
        if not isinstance(ir_image, np.ndarray):
            ir_image = np.array(ir_image)
            if len(ir_image.shape) == 2:
                ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

        if filtered_correspondences is not None and len(filtered_correspondences) > 0:
            corr_np = filtered_correspondences.detach().cpu().numpy()
        else:
            corr_np = correspondences.detach().cpu().numpy()

        h_rgb, w_rgb = rgb_image.shape[:2]
        h_ir, w_ir = ir_image.shape[:2]
        max_h = max(h_rgb, h_ir)

        canvas = np.zeros((max_h, w_rgb + 2 * w_ir, 3), dtype=np.uint8)
        canvas[:h_rgb, :w_rgb] = rgb_image[:h_rgb, :w_rgb]
        canvas[:h_ir, w_rgb:w_rgb + w_ir] = ir_image[:h_ir, :w_ir]
        canvas[:h_ir, w_rgb + w_ir:w_rgb + 2 * w_ir] = ir_image[:h_ir, :w_ir].copy()

        num_points_to_draw = min(100, len(corr_np))
        if num_points_to_draw > 0:
            if len(corr_np) > num_points_to_draw:
                idx = np.random.choice(len(corr_np), num_points_to_draw, replace=False)
                selected = corr_np[idx]
            else:
                selected = corr_np

            colors = np.random.randint(0, 255, size=(num_points_to_draw, 3), dtype=np.uint8)
            for i in range(num_points_to_draw):
                rgb_u, rgb_v, ir_u, ir_v = selected[i]
                rgb_u = int(max(0, min(rgb_u, w_rgb - 1)))
                rgb_v = int(max(0, min(rgb_v, h_rgb - 1)))
                ir_u = int(max(0, min(ir_u, w_ir - 1)))
                ir_v = int(max(0, min(ir_v, h_ir - 1)))
                ir_u_canvas = ir_u + w_rgb
                color = tuple(colors[i].tolist())
                cv2.circle(canvas, (rgb_u, rgb_v), 5, color, -1)
                cv2.circle(canvas, (ir_u_canvas, ir_v), 5, color, -1)
                cv2.line(canvas, (rgb_u, rgb_v), (ir_u_canvas, ir_v), color, 1, cv2.LINE_AA)

        cv2.putText(canvas, 'RGB Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, 'IR Image (with Correspondences)', (w_rgb + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, 'IR Reference Image', (w_rgb + w_ir + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if save_path:
            cv2.imwrite(save_path, canvas)
        else:
            try:
                cv2.imshow('RGB-IR Correspondences', canvas)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                pass
