from uavdet3d.datasets import DatasetTemplate
from uavdet3d.datasets.mav6d.eval import eval_6d_pose
import numpy as np
import os
import torch
import pandas as pd
import cv2
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from uavdet3d.datasets.mav6d.mav6d_utils import draw_box9d_on_image
import copy

try:
    # 尝试用 tkAGG（有界面环境可用）
    matplotlib.use('tkAGG')
except ImportError:
    # 无界面环境 fallback 到 Agg
    matplotlib.use('Agg')

class CARLA_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training, root_path, logger):
        super(CARLA_Det_Dataset, self).__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)

        self.dataset_cfg = dataset_cfg
        self.root_path = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.training = training
        self.logger = logger

        self.modalities = ['rgb', 'ir', 'dvs']
        self.intrinsics = {}
        self.extrinsics = {}
        self.distortions = {}
        self.default_distortion = np.array([0, 0, 0, 0, 0])

        self.im_path_names = ['images_rgb', 'images_ir', 'images_dvs']
        self.label_path_names = ['boxes_rgb', 'boxes_ir', 'boxes_dvs']

        self.raw_im_width = dataset_cfg.IM_SIZE[0]
        self.raw_im_hight = dataset_cfg.IM_SIZE[1]
        self.new_im_width = dataset_cfg.IM_RESIZE[0]
        self.new_im_hight = dataset_cfg.IM_RESIZE[1]
        self.obj_size = np.array(self.dataset_cfg.OB_SIZE)
        self.stride = self.dataset_cfg.STRIDE

        self.seq_list = self._build_seq_list_all_maps(split_ratio=0.8)

        self.sample_scene_list = []
        self.sorted_namelist = self.dataset_cfg.CLASS_NAMES

        for seq_name in self.seq_list:
            seq_path = os.path.join(str(self.root_path), seq_name, 'images_rgb')
            if not os.path.exists(seq_path):
                self.logger.warning(f"No image path: {seq_path}")
                continue

            all_frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame in all_frames:
                self.sample_scene_list.append([seq_name, frame])

        self.infos = []
        self.include_CARLA_data(self.mode)

    def include_CARLA_data(self, mode):
        self.logger.info('Loading CARLA dataset')
        CARLA_infos = []

        for i in range(0, len(self.sample_scene_list), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
            seq_name, frame_name = self.sample_scene_list[i]
            base_path = os.path.join(self.root_path, seq_name)

            im_paths = {}
            for mode_name in self.modalities:
                image_dir = f'images_{mode_name}'
                image_path = os.path.join(base_path, image_dir, frame_name)
                if not os.path.exists(image_path):
                    self.logger.warning(f" No image found: {image_path}")
                im_paths[mode_name] = image_path

            label_paths = {}
            for mode_name in self.modalities:
                label_dir = f'boxes_{mode_name}'
                label_path = os.path.join(base_path, label_dir, frame_name.replace('.png', '.pkl'))
                if not os.path.exists(label_path):
                    self.logger.warning(f"No label found: {label_path}")
                label_paths[mode_name] = label_path

            lidar_path = os.path.join(base_path, 'lidar_1', frame_name.replace('.png', '.npy'))
            if not os.path.exists(lidar_path):
                self.logger.warning(f"LiDAR file not found: {lidar_path}")

            im_info_path = os.path.join(base_path, 'im_info.pkl')

            lidar_radar_info_path = os.path.join(base_path, 'lidar_radar_info.pkl')
            if not os.path.exists(lidar_radar_info_path):
                self.logger.warning(f"LiDAR radar info not found: {lidar_radar_info_path}")
                lidar_extrinsic = np.eye(4)
            else:
                with open(lidar_radar_info_path, 'rb') as f:
                    lidar_radar_info = pickle.load(f)
                lidar_extrinsic = np.array(lidar_radar_info['lidars'][0]['extrinsic'], dtype=np.float32)

            data_info = {
                'im_paths': im_paths,
                'label_paths': label_paths,
                'lidar_path': lidar_path,
                'im_info_path': im_info_path,
                'lidar_extrinsic': lidar_extrinsic,
                'seq_id': seq_name,
                'frame_id': frame_name,
                'cls_name': self.sorted_namelist
            }

            CARLA_infos.append(data_info)

        self.infos = CARLA_infos


    def __len__(self):

        return len(self.infos)

    def project_lidar_and_get_uvz_rgb_tag(self,
        lidar_points,             # (N, 5): x, y, z, intensity, tag
        lidar_extrinsic,          # 4x4
        camera_extrinsic,         # 4x4
        camera_intrinsic,         # 3x3
        image                     # (H, W, 3)
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
        uv = uv.reshape(-1, 2)

        results = []
        for i in range(uv.shape[0]):
            u, v = uv[i]
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                z = points_cam_xyz[i, 2]
                r, g, b = image[v_int, u_int].tolist()
                tag = tags[i]
                results.append([u, v, z, r, g, b, tag])

        return np.array(results)  # shape: (M, 7)

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

        with open(im_info_path, 'rb') as f:
            camera_info = pickle.load(f)
        intrinsic = np.array(camera_info['rgb']['intrinsic'])
        extrinsic = np.array(camera_info['rgb']['extrinsic'])
        lidar_extrinsic = np.array(each_info['lidar_extrinsic'])
        image_rgb = cv2.imread(im_paths['rgb'], cv2.IMREAD_COLOR)
        if image_rgb is None:
            raise FileNotFoundError(f"Cannot read image: {im_paths['rgb']}")

        lidar_proj_info = self.project_lidar_and_get_uvz_rgb_tag(
            lidar_data,
            lidar_extrinsic=lidar_extrinsic,
            camera_extrinsic=extrinsic,
            camera_intrinsic=intrinsic,
            image=image_rgb
        ) # shape (M, 7): u, v, depth, r, g, b, tag

        image_modal_stack = []
        depth_map, tag_map = self.generate_depth_and_tag_map(
            lidar_proj_info,
            image_shape=(self.raw_im_hight, self.raw_im_width)
        )
        for mode in self.modalities:
            im_path = im_paths[mode]
            if mode == 'ir':
                img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
                img = img.astype(np.float32)
                img = cv2.resize(img, (self.new_im_width, self.new_im_hight))
                img = img[np.newaxis, :, :]          # (1, H, W)
                img = np.repeat(img, 3, axis=0)      # → (3, H, W)
            else:
                img = cv2.imread(im_path, cv2.IMREAD_COLOR)  # (H, W, 3)
                img = img.astype(np.float32)
                img = cv2.resize(img, (self.new_im_width, self.new_im_hight))
                img = img.transpose(2, 0, 1)  # → (3, H, W)

            image_modal_stack.append(img)

        # Depth map (原始是 1 channel)，变成 3 通道
        depth_map_resized = cv2.resize(depth_map, (self.new_im_width, self.new_im_hight), interpolation=cv2.INTER_NEAREST)
        depth_map_tensor = depth_map_resized[np.newaxis, :, :]  # (1, H, W)
        depth_map_tensor = np.repeat(depth_map_tensor, 3, axis=0)  # → (3, H, W)

        # 将 4 个模态合成 4x3xHxW
        image_modal_stack.append(depth_map_tensor)

        # → 最终 shape: (4, 3, H, W)
        image = np.stack(image_modal_stack, axis=0)

        box_file = label_paths['rgb']
        gt_boxes, gt_names = [], []

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

                pts_world = self.convert_box_opencv_to_world(pts, extrinsic)

                center_world = (
                    (pts_world[0] + pts_world[6]) +
                    (pts_world[1] + pts_world[7]) +
                    (pts_world[2] + pts_world[4]) +
                    (pts_world[3] + pts_world[5])
                ) / 8.0
                center_world = center_world.reshape(1, 3)

                box_9pts_world = np.concatenate([center_world, pts_world], axis=0)
                gt_boxes.append(box_9pts_world)

                matched = "Unknown"
                for cls in cls_name:
                    if cls in raw_name:
                        matched = cls
                        break
                gt_names.append(matched)
            gt_boxes = np.array(gt_boxes, dtype=np.float32)  # (N, 9, 3)
            gt_names = np.array(gt_names)
        else:
            self.logger.warning(f"No label file for RGB: {box_file}")
            gt_boxes = np.empty((0, 9, 3), dtype=np.float32)
            gt_names = np.empty((0,), dtype='<U32')
        with open(im_info_path, 'rb') as f:
            camera_info = pickle.load(f)

        distortion = self.default_distortion.copy()

        data_dict = {
            'image': image,                     # shape (8, H, W)
            'gt_boxes': gt_boxes,               # (N, 9, 3)
            'gt_names': gt_names,               # (N,)
            'intrinsic': intrinsic,             # (3, 3)
            'extrinsic': extrinsic,             # (4, 4)
            'distortion': distortion,           # (5,)
            'raw_im_size': np.array([self.raw_im_width, self.raw_im_hight]),
            'new_im_size': np.array([self.new_im_width, self.new_im_hight]),
            'obj_size': np.array(self.dataset_cfg.OB_SIZE),
            'seq_id': seq_id,
            'frame_id': frame_id,
            'stride': self.stride,
            'sorted_namelist': self.sorted_namelist
        }

        data_dict = self.data_pre_processor(data_dict)
        return data_dict

    def collate_batch(self, batch_list, _unused=False):
        print("collate_batch called, batch size =", len(batch_list))

        data_dict = defaultdict(list)
        for i, cur_sample in enumerate(batch_list):
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key == 'image':
                    merged = np.stack(val, axis=0)  # (B, 4, 3, H, W)
                    print(f"[Debug] image shape for backbone: {merged.shape}")
                    ret[key] = merged
                elif key == 'gt_boxes':
                    max_gt = max([v.shape[0] for v in val])
                    merged = np.zeros((batch_size, max_gt, 9, 3), dtype=np.float32)
                    for i in range(batch_size):
                        cur_len = val[i].shape[0]
                        merged[i, :cur_len] = val[i]
                    print(f"  - gt_boxes: shape = {merged.shape}, max_gt = {max_gt}")
                    ret[key] = merged

                elif key == 'gt_names':
                    print(f"  - gt_names: example = {val[0]}")
                    ret[key] = val

                elif key == 'gt_diffs':
                    max_gt = max([v.shape[0] for v in val])
                    merged = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for i in range(batch_size):
                        cur_len = val[i].shape[0]
                        merged[i, :cur_len] = val[i]
                    print(f"  - gt_diffs: shape = {merged.shape}, max_gt = {max_gt}")
                    ret[key] = merged

                elif key in ['intrinsic', 'extrinsic', 'distortion']:
                    stacked = np.stack(val, axis=0)
                    print(f"  - {key}: shape = {stacked.shape}")
                    ret[key] = stacked

                elif key in ['raw_im_size', 'new_im_size', 'obj_size']:
                    stacked = np.stack(val, axis=0)
                    print(f"  - {key}: shape = {stacked.shape}")
                    ret[key] = stacked

                elif key in ['seq_id', 'frame_id']:
                    print(f"  - {key}: {val[:2]} ... ({len(val)} total)")
                    ret[key] = val

                elif key == 'stride':
                    print(f"  - stride: {val[0]}")
                    ret[key] = val[0]

                elif key == 'sorted_namelist':
                    continue

                elif key in ['hm', 'center_res', 'center_dis', 'dim', 'rot']:
                    try:
                        if key == 'hm':
                            max_cls = max([v.shape[0] for v in val])
                            padded_val = []
                            for v in val:
                                if v.shape[0] < max_cls:
                                    pad_width = ((0, max_cls - v.shape[0]), (0, 0), (0, 0))
                                    v = np.pad(v, pad_width, mode='constant')
                                padded_val.append(v)
                            stacked = np.stack(padded_val, axis=0)
                        else:
                            stacked = np.stack(val, axis=0)
                        print(f"  - {key}: shape = {stacked.shape}")
                        ret[key] = stacked
                    except Exception as e:
                        print(f"[Error] Failed to stack key '{key}': {e}")
                        raise e


                else:
                    print(f"Unknown key '{key}' ignored")

            except Exception as e:
                print(f"Error in collate_batch for key={key}: {e}")
                raise e

        ret['batch_size'] = batch_size
        ret['sorted_namelist'] = self.dataset_cfg.CLASS_NAMES

        print("collate_batch finished\n")
        return ret


    def generate_prediction_dicts(self, batch_dict, output_path):
        batch_size = batch_dict['batch_size']
        annos = []

        for batch_id in range(batch_size):
            seq_id = batch_dict['seq_id'][batch_id]
            frame_id = batch_dict['frame_id'][batch_id]

            raw_im_size = batch_dict['raw_im_size'][batch_id]  # (2,)
            obj_size = batch_dict['obj_size'][batch_id]
            intrinsic = batch_dict['intrinsic'][batch_id]       # (3, 3)
            extrinsic = batch_dict['extrinsic'][batch_id]       # (4, 4)
            distortion = batch_dict['distortion'][batch_id]     # (5,)
            #[center_x, center_y, center_z, l, w, h, a1, a2, a3, class_id]
            pred_boxes9d = batch_dict['pred_boxes9d'][batch_id]     # (N, 10)
            pred_class_ids = pred_boxes9d[:, -1].astype(int)
            pred_names = self.name_from_code(pred_class_ids)
            pred_boxes9d = self.convert_pred_box10_to_box9d(pred_boxes9d)

            confidence = batch_dict['confidence'][batch_id]         # (N,)
            sorted_namelist = batch_dict['sorted_namelist'][batch_id]
            frame_dict = {
                'seq_id': seq_id,
                'frame_id': frame_id,
                'obj_size': obj_size,
                'pred_boxes': pred_boxes9d,
                'pred_names': pred_names,
                'confidence': confidence,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'distortion': distortion,
            }

            im_path = os.path.join(self.root_path, seq_id, 'images_rgb', frame_id)
            frame_dict['im_path'] = im_path

            image = cv2.imread(im_path)
            if image is None:
                print(f"Failed to read image: {im_path}")
                continue

            # Draw predicted boxes
            image = self.draw_box9d_on_image(
                pred_boxes9d, image,
                img_width=raw_im_size[0],
                img_height=raw_im_size[1],
                color=(255, 0, 0),
                intrinsic_mat=intrinsic,
                extrinsic_mat=extrinsic,
                distortion_matrix=distortion
            )

            # If GT exists
            if 'gt_boxes' in batch_dict:
                gt_box9d = batch_dict['gt_boxes'][batch_id]      # (N, 9, 3)
                gt_names = batch_dict['gt_names'][batch_id]        # (N,)
                frame_dict['gt_boxes'] = gt_box9d
                frame_dict['gt_names'] = gt_names

                image = self.draw_box9d_on_image(
                    gt_box9d, image,
                    img_width=raw_im_size[0],
                    img_height=raw_im_size[1],
                    color=(0, 0, 255),
                    intrinsic_mat=intrinsic,
                    extrinsic_mat=extrinsic,
                    distortion_matrix=distortion
                )

            cv2.imshow('rgb_image', image)
            cv2.waitKey(1)

            annos.append(frame_dict)

        return annos

    def evaluation_by_name(self, annos, metric_root_path, cls_name):

        orin_error, pos_error = self.eval_6d_pose(annos, max_dis=self.dataset_cfg.MAX_DIS, target_cls=cls_name)

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
        print(f"{cls_name}_pose_error: {pos_error.mean():.4f}")
        print(f"{cls_name}_orin_error_median: {np.median(orin_error):.4f}")
        print(f"{cls_name}_pose_error_median: {np.median(pos_error):.4f}")
        print(f"{cls_name}_orin_error_min： {orin_error.min():.4f}")
        print(f"{cls_name}_pose_error_min: {pos_error.min():.4f}")
        print(f"{cls_name}_orin_error_max： {orin_error.max():.4f}")
        print(f"{cls_name}_pose_error_max: {pos_error.max():.4f}")

        return_str = f'''
            {cls_name}: 
            orin_error： {orin_error.mean():.4f}
            pose_error: {pos_error.mean():.4f}
            orin_error_median: {np.median(orin_error):.4f}
            pos_error_median: {np.median(pos_error):.4f}
            orin_error_min: {orin_error.min():.4f}
            pos_error_min: {pos_error.min():.4f}
            orin_error_max: {orin_error.max():.4f}
            pose_error_max: {pos_error.max():.4f}
        '''

        pd.DataFrame({f'{cls_name}_orin_error': orin_error.tolist()}).to_csv(
            os.path.join(metric_root_path, f'{cls_name}_orin_error.csv'), index=False)
        pd.DataFrame({f'{cls_name}_pos_error': pos_error.tolist()}).to_csv(
            os.path.join(metric_root_path, f'{cls_name}_pos_error.csv'), index=False)
        pd.DataFrame({
            f'{cls_name}_orin_error': [orin_error.mean()],
            f'{cls_name}_pose_error': [pos_error.mean()]
        }).to_csv(os.path.join(metric_root_path, f'{cls_name}_mean_error.csv'), index=False)

        return return_str

    def evaluation(self, annos, metric_root_path):
        all_names = self.sorted_namelist
        final_str = ''

        for cls_n in all_names:
            cls_annos = []
            for each_anno in annos:
                gt_boxs = each_anno['gt_boxes']
                pred_boxs = each_anno['pred_boxes']
                gt_names = each_anno['gt_names']
                pred_names = each_anno['pred_names']

                mask_gt = (gt_names == cls_n)
                mask_pred = (pred_names == cls_n)

                if mask_gt.sum() > 0:
                    cls_annos.append({
                        'gt_boxes': gt_boxs[mask_gt],
                        'gt_names': gt_names[mask_gt],
                        'pred_boxes': pred_boxs[mask_pred],
                        'pred_names': pred_names[mask_pred]
                    })

            if len(cls_annos) >= 2:
                final_str += self.evaluation_by_name(cls_annos, metric_root_path, cls_name=cls_n)

        merged_annos = []
        for each_anno in annos:
            if len(each_anno['gt_boxes']) > 0:
                merged_annos.append({
                    'gt_boxes': self.convert_box9d_to_box_param(each_anno['gt_boxes']),
                    'gt_names': each_anno['gt_names'],
                    'pred_boxes': self.convert_box9d_to_box_param(each_anno['pred_boxes']) ,
                    'pred_names': each_anno['pred_names']
                })

        final_str += self.evaluation_by_name(merged_annos, metric_root_path, cls_name='all')

        return final_str


    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    ## Util funtcions
    def draw_box9d_on_image(self,boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0),
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
                cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), (255, 0, 0), 1)
            for idx, edge in enumerate(middle_edges):
                intensity = int(255 - (idx / 3) * 150)
                color = (0, intensity, 0)
                cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), color, 1)
            for edge in top_edges:
                cv2.line(image, tuple(corners_2d[edge[0]]), tuple(corners_2d[edge[1]]), (0, 0, 255), 1)

            diagonals = [(0, 6), (1, 7), (2, 4), (3, 5)]
            mid_points = []
            for i, j in diagonals:
                pt1, pt2 = box_cv[1:][i], box_cv[1:][j]
                mid = (pt1 + pt2) / 2
                mid_points.append(mid)
                proj_pts, _ = cv2.projectPoints(np.vstack([pt1, pt2]), np.eye(3), np.zeros(3), intrinsic_mat, distortion_matrix)
                p1, p2 = proj_pts.reshape(-1, 2).astype(int)
                cv2.line(image, tuple(p1), tuple(p2), (0, 255, 255), 1)

        return image

    def convert_pred_box10_to_box9d(self, boxes10):
        all_box9d = []

        for box in boxes10:
            center = box[0:3]
            l, w, h = box[3:6]
            a1, a2, a3 = box[6:9]
            x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
            y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
            z_corners = [-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2]
            corners_local = np.stack([x_corners, y_corners, z_corners], axis=1)  # (8, 3)
            rot_mat = R.from_euler('zyx', [a1, a2, a3], degrees=False).as_matrix()
            corners_rotated = np.dot(corners_local, rot_mat.T)
            corners_world = corners_rotated + center
            box9d = np.vstack([center.reshape(1, 3), corners_world])  # (9, 3)
            all_box9d.append(box9d)

        return np.stack(all_box9d)  # (N, 9, 3)

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
                for list_file in ["list_easy.txt", "list_moderate.txt", "list_hard.txt"]:
                    list_path = os.path.join(seq_path, list_file)
                    if not os.path.exists(list_path):
                        continue
                    with open(list_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                weather, drone = line.split('/')[:2]
                                full_path = f"{map_name}/carla_data/{seq_id}/{weather}/{drone}"
                                all_samples.add(full_path)

        all_samples = sorted(list(all_samples))
        self.logger.info(f"Total sequences: {len(all_samples)}")

        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * split_ratio)

        train_list = all_samples[:split_idx]
        val_list = all_samples[split_idx:]

        if self.training:
            self.logger.info(f"Train Sequences: {len(train_list)}")
            return train_list
        else:
            self.logger.info(f"Test Sequences: {len(val_list)}")
            return val_list

    def set_split(self, split):
        super(CARLA_Det_Dataset, self).__init__(
            dataset_cfg=self.dataset_cfg,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger
        )

        self.sample_scene_list = []

        self.seq_list = self._build_seq_list_all_maps(split_ratio=0.8)

        for seq_name in self.seq_list:
            seq_path = os.path.join(str(self.root_path), seq_name, 'images_rgb')
            if not os.path.exists(seq_path):
                self.logger.warning(f"No image path: {seq_path}")
                continue

            all_frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            for frame in all_frames:
                self.sample_scene_list.append([seq_name, frame])
        self.infos = []
        self.include_CARLA_data(self.mode)

    def generate_depth_and_tag_map(self, lidar_proj_info, image_shape):
        h, w = image_shape
        depth_map = np.zeros((h, w), dtype=np.float32)
        tag_map = np.full((h, w), -1, dtype=np.int32)

        for u, v, z, r, g, b, tag in lidar_proj_info:
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                if depth_map[v_int, u_int] == 0 or z < depth_map[v_int, u_int]:
                    depth_map[v_int, u_int] = z
                    tag_map[v_int, u_int] = int(tag)

        return depth_map, tag_map

    def convert_box_opencv_to_world(self,pts_opencv, extrinsic):
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

    def name_from_code(self, name_indices):
        names = [self.dataset_cfg.CLASS_NAMES[int(x)] for x in name_indices]
        return np.array(names)

    def convert_box9d_to_box_param(self, boxes_9d):
        all_box_params = []

        for box in boxes_9d:
            center = box[0]
            corners = box[1:9]

            l = np.linalg.norm(corners[0] - corners[1])
            w = np.linalg.norm(corners[1] - corners[2])
            h = np.linalg.norm(corners[0] - corners[4])

            x_axis = (corners[1] - corners[0])
            y_axis = (corners[3] - corners[0])
            z_axis = (corners[4] - corners[0])
            rot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
            rot_mat = rot_mat / np.linalg.norm(rot_mat, axis=0, keepdims=True)

            try:
                r = R.from_matrix(rot_mat)
                euler = r.as_euler('zyx', degrees=False)
            except:
                euler = np.zeros(3)

            box_param = np.concatenate([center, [l, w, h], euler])
            all_box_params.append(box_param)

        return np.array(all_box_params)


    def eval_key_points_error(self,annos):

        all_error = []

        for i, each_anno in enumerate(annos) :
            key_points_2d = each_anno['key_points_2d']
            gt_pts2d = each_anno['gt_pts2d']

            all_error.append(np.abs(key_points_2d-gt_pts2d).reshape(-1))

        all_error = np.concatenate(all_error)

        return all_error


    def euler_angle_error(self,pred, gt):
        """
        计算弧度制下欧拉角的角度误差。

        参数:
        pred (numpy.ndarray): 预测的欧拉角，形状为 (N, 3)。
        gt (numpy.ndarray): 真实的欧拉角，形状为 (N, 3)。

        返回:
        numpy.ndarray: 每个样本的角度误差，形状为 (N,)。
        """
        diff = pred - gt

        diff = np.arctan2(np.sin(diff), np.cos(diff))

        error = np.linalg.norm(diff, axis=1)

        return error

    def euler_angle_error_rad(self,pred, gt):

        diff = pred - gt

        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        error = np.sum(np.abs(diff), axis=1)
        return error


    def limit(self,ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang

    def ang_weight(self,pred, gt):

        a = np.abs(pred - gt)
        b = 2 * np.pi - np.abs(pred - gt)

        res = np.stack([a, b])

        res = np.min(res, axis=0)

        return res

    def hungarian_match_allow_unmatched(self,x: np.ndarray, y: np.ndarray, unmatched_cost: float = 1e5):

        N, M = x.shape[0], y.shape[0]



        cost = cdist(x, y)

        size = max(N, M)
        padded_cost = np.full((size, size), unmatched_cost)
        padded_cost[:N, :M] = cost

        row_ind, col_ind = linear_sum_assignment(padded_cost)

        x_to_y = np.full(N, -1, dtype=int)

        for r, c in zip(row_ind, col_ind):
            if r < N and c < M and padded_cost[r, c] < unmatched_cost:
                x_to_y[r] = c  # 合法匹配

        return x_to_y

    def match_gt(self,gt_box9d, pred_boxes9d):
        new_pred = []

        x_to_y = self.hungarian_match_allow_unmatched(gt_box9d[:,0:3], pred_boxes9d[:,0:3])

        for i, j in enumerate(x_to_y):

            if j==-1:
                new_pred.append(np.zeros(shape=(10)))
            else:
                new_pred.append(pred_boxes9d[j])

        return gt_box9d, np.array(new_pred)

    def eval_box6d_error(self,annos, max_dis = 8):

        all_orin_error = []

        all_pos_error = []

        for i, each_anno in enumerate(annos) :
            gt_box9d = each_anno['gt_box9d']
            pred_boxes9d = each_anno['pred_box9d']

            pred_xyz = pred_boxes9d[:, 0:3]

            pred_xyz[np.abs(pred_xyz)>max_dis] = max_dis

            pred_boxes9d[:, 0:3] = pred_xyz

            if len(gt_box9d)==0 or len(pred_boxes9d)==0:
                continue
            gt_box9d, pred_boxes9d = self.match_gt(gt_box9d, pred_boxes9d)

            pred_xyz = pred_boxes9d[:, 0:3]

            gt_xyz = gt_box9d[:, 0:3]

            pred_angle = pred_boxes9d[:, 6:9]
            gt_angle = gt_box9d[:, 6:9]

            dis_error = np.linalg.norm(pred_xyz-gt_xyz,axis=-1)
            angle_error = self.val_rotation_enler(pred_angle, gt_angle)

            all_orin_error.append(angle_error)
            all_pos_error.append(dis_error)

        if len(all_orin_error)>0 and len(all_pos_error)>0:

            all_orin_error = np.concatenate(all_orin_error)
            all_pos_error = np.concatenate(all_pos_error)
        else:
            all_orin_error=np.array([0])
            all_pos_error=np.array([0])

        return all_orin_error, all_pos_error

    def val_rotation(self,pred_q, gt_q):
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
        # error = 2 * np.arccos(d) * 180 / np.pi0
        # d     = abs(np.dot(groundtruth, predicted))
        # d     = min(1.0, max(-1.0, d))

        d = np.abs(np.dot(groundtruth, predicted))
        d = np.minimum(1.0, np.maximum(-1.0, d))
        error = 2 * np.arccos(d) * 180 / np.pi

        return error

    def val_rotation_enler(self,pred_q, gt_q):
        rotation_pred = R.from_euler('zyx', pred_q, degrees=False)
        rotation_gt = R.from_euler('zyx', gt_q, degrees=False)

        rotation_pred = rotation_pred.as_quat()
        rotation_gt = rotation_gt.as_quat()

        all_error = [self.val_rotation(p,q) for p,q in zip(rotation_pred,rotation_gt)]

        all_error = np.array(all_error)

        all_error[all_error>90] = 180 - all_error[all_error>90]

        return all_error



    def eval_6d_pose(self,annos, max_dis = 8):


        orin_error, pos_error = self.eval_box6d_error(annos, max_dis)


        return orin_error, pos_error



