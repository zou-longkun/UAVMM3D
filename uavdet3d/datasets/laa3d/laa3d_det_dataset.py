from uavdet3d.datasets import DatasetTemplate
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
import pickle as pkl
from .ads_metric import LAA3D_ADS_Metric
from scipy.spatial.transform import Rotation as R

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

try:
    # 尝试用 tkAGG（有界面环境可用）
    matplotlib.use('tkAGG')
except ImportError:
    # 无界面环境 fallback 到 Agg
    matplotlib.use('Agg')

def box9d_to_2d(boxes9d, img_width=1280., img_height=720., intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None):
    import numpy as np
    import cv2
    from scipy.spatial.transform import Rotation as R

    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)

    corners_local = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=np.float32)

    boxes2d = []
    size = []

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box

        corners = corners_local * np.array([l, w, h])
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)
        corners += np.array([x, y, z])

        # ---- 变换到相机坐标系 ----
        corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))
        corners_camera = (extrinsic_mat @ corners_homogeneous.T).T[:, :3]

        # ---- 投影到图像 ----
        rvec, _ = cv2.Rodrigues(extrinsic_mat[:3, :3])
        tvec = extrinsic_mat[:3, 3]

        corners_2d, _ = cv2.projectPoints(corners, rvec, tvec, intrinsic_mat, distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        corners_2d[:, 0] = np.clip(corners_2d[:, 0], a_min=0, a_max=int(img_width)-1)
        corners_2d[:, 1] = np.clip(corners_2d[:, 1], a_min=0, a_max=int(img_height)-1)

        x1, y1 = corners_2d.min(axis=0)
        x2, y2 = corners_2d.max(axis=0)
        boxes2d.append([x1, y1, x2, y2])
        size.append(min(abs(x2 - x1), abs(y2 - y1)))

    return np.array(boxes2d), np.array(size)

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

class LAA3D_Det_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training, root_path, logger):
        super(LAA3D_Det_Dataset, self).__init__(dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger)

        self.dataset_cfg=dataset_cfg
        self.root_path=root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.training=training
        self.logger=logger

        self.raw_im_width = dataset_cfg.IM_SIZE[0]
        self.raw_im_hight = dataset_cfg.IM_SIZE[1]

        self.new_im_width = dataset_cfg.IM_RESIZE[0]
        self.new_im_hight = dataset_cfg.IM_RESIZE[1]

        self.distortion_matrix = np.array([0, 0, 0, 0, 0])

        self.im_num = self.dataset_cfg.IM_NUM

        self.center_rad = self.dataset_cfg.CENTER_RAD

        self.stride = self.dataset_cfg.STRIDE

        self.class_name_dict = self.dataset_cfg.CLASS_NAMES

        self.interval= self.dataset_cfg.SAMPLED_INTERVAL[self.mode]


        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.data_info = None

        with open(os.path.join(self.root_path, self.split+'_info.pkl'), 'rb') as f:
            self.data_info = pkl.load(f)

        data_info = []

        for i in range(0,len(self.data_info), self.interval):
            data_info.append(self.data_info[i])

        self.data_info = data_info

    def set_split(self, split):
        self.split = split

        self.data_info = None

        with open(os.path.join(self.root_path, self.split + '_info.pkl'), 'rb') as f:
            self.data_info = pkl.load(f)

        data_info = []

        for i in range(0, len(self.data_info), self.interval):
            data_info.append(self.data_info[i])

        self.data_info = data_info


    def __len__(self):

        return len(self.data_info)

    def __getitem__(self, item):

        # item = item%10

        info = self.data_info[item]

        setting_id = info['setting_id']
        seq_id = info['seq_id']
        brightness = info['brightness']
        frame_id = info['frame_id']
        relative_im_path = info['relative_im_path']

        annos = info['annos']

        boxes9d = annos['box']  # 3D boxes
        this_boxes_2d = annos['box2d']  # 2D boxes

        diff = annos['diff']  # difficulty level
        this_coarse_class = annos['coarse_class']  # object classes
        this_fine_class = annos['fine_class']  # object classes
        this_id = annos['ob_id']  # object identity

        boxes9d = boxes9d[diff < self.dataset_cfg.DiffLevel]
        this_coarse_class = this_coarse_class[diff < self.dataset_cfg.DiffLevel]
        diff = diff[diff < self.dataset_cfg.DiffLevel]

        image_info = info['image_info']
        intrinsic = image_info['intrinsic']
        extrinsic = image_info['extrinsic']
        
        relative_im_path = relative_im_path.replace('\\', '/')
        image = cv2.imread(os.path.join(self.root_path, relative_im_path)).astype(np.float32)


        resized_img = cv2.resize(
            image,
            (self.new_im_width, self.new_im_hight),  # 目标尺寸 (width, height)
            interpolation=cv2.INTER_AREA  # 缩小推荐使用区域插值
        )
        resized_img = resized_img.transpose(2,0,1)
        C,W,H = resized_img.shape
        resized_img = resized_img.reshape(1,C,W,H)

        boxes9d = np.array(boxes9d)

        if len(boxes9d) ==0:
            boxes9d = np.empty(shape=(0, 9))
            this_coarse_class = np.empty(shape=(0,))
            diff = np.empty(shape=(0,))

        data_dict = {}

        data_dict['intrinsic'] = np.array([intrinsic])
        data_dict['extrinsic'] = np.array([extrinsic])
        data_dict['distortion'] = np.array([self.distortion_matrix])
        data_dict['raw_im_size'] = np.array([self.raw_im_width, self.raw_im_hight])
        data_dict['new_im_size'] = np.array([self.new_im_width, self.new_im_hight])
        data_dict['seq_id'] = seq_id
        data_dict['frame_id'] = frame_id
        data_dict['setting_id'] = setting_id
        data_dict['brightness'] = brightness
        data_dict['relative_im_path'] = relative_im_path
        data_dict['stride'] = self.stride
        data_dict['image'] = resized_img
        data_dict['gt_box9d'] = boxes9d
        data_dict['gt_diff'] = diff
        data_dict['gt_name'] = this_coarse_class


        data_dict = self.data_pre_processor(data_dict)

        return data_dict


    def name_from_code(self, name_indices):

        names = [self.class_name_dict[int(x)] for x in name_indices]

        return np.array(names)


    def generate_prediction_dicts(self, batch_dict, output_path):

        batch_size = batch_dict['batch_size']

        annos = []

        for batch_id in range(batch_size):

            pred_boxes9d = batch_dict['pred_boxes9d'][batch_id] # 1, 1, 4, W, H

            seq_id = batch_dict['seq_id'][batch_id]
            frame_id = batch_dict['frame_id'][batch_id]
            setting_id = batch_dict['setting_id'][batch_id]
            brightness = batch_dict['brightness'][batch_id]
            relative_im_path = batch_dict['relative_im_path'][batch_id]

            intrinsic = batch_dict['intrinsic'][batch_id][0] # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id][0] # 4, 4
            distortion = batch_dict['distortion'][batch_id][0] # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id] # 2,
            new_im_size = batch_dict['new_im_size'][batch_id] # 2,

            # key_points_2d = batch_dict['key_points_2d'][batch_id]
            confidence = batch_dict['confidence'][batch_id]
            im_path = os.path.join(self.root_path, relative_im_path)

            pred_name = self.name_from_code(pred_boxes9d[:,-1])

            frame_dict = {'seq_id': seq_id,
                          'frame_id': frame_id,
                          'setting_id': setting_id,
                          'brightness': brightness,
                          'intrinsic': intrinsic,
                          'extrinsic': extrinsic,
                          'distortion': distortion,
                          'raw_im_size': raw_im_size,
                          'confidence': confidence,
                          'pred_box9d': pred_boxes9d,
                          'pred_name': pred_name,
                          'im_path': im_path
                          }

            if 'gt_box9d' in batch_dict:
                gt_diff = batch_dict['gt_diff'][batch_id]
                frame_dict['gt_diff'] = gt_diff
                gt_box9d = batch_dict['gt_box9d'][batch_id].cpu().numpy() # 1, 9
                frame_dict['gt_box9d'] = gt_box9d
                gt_name = batch_dict['gt_name'][batch_id]
                frame_dict['gt_name'] = gt_name

            annos.append(frame_dict)

            # image = cv2.imread(im_path)
            # pred_boxes9d[:,3:6]*=1.
            #
            # # gt_box9d = gt_box9d[gt_box9d[:,2]<10]
            # # print(gt_box9d)
            # # if len(gt_box9d)<1:
            # #     return annos
            #
            # image = draw_box9d_on_image(pred_boxes9d, image,
            #                             img_width=raw_im_size[0],
            #                             img_height=raw_im_size[1],
            #                             color=(255, 0, 0),
            #                             intrinsic_mat=intrinsic,
            #                             extrinsic_mat=extrinsic,
            #                             distortion_matrix=distortion,
            #                             offset=np.array([0., 0., 0.]))
            #
            # image = draw_box9d_on_image(gt_box9d, image,
            #                             img_width=raw_im_size[0],
            #                             img_height=raw_im_size[1],
            #                             color=(0, 0, 255),
            #                             intrinsic_mat=intrinsic,
            #                             extrinsic_mat=extrinsic,
            #                             distortion_matrix=distortion,
            #                             offset=np.array([0., 0., 0.]))
            #
            # cv2.imshow('im', image)
            # cv2.waitKey(0)

        return annos



    def evaluation_all(self, annos, metric_root_path):

        laa_ads_fun = LAA3D_ADS_Metric(eval_config = self.dataset_cfg.LAA3D_ADS_METRIC, classes = self.class_name_dict, metric_save_path=metric_root_path)

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

            if len(setting_mask)>0:
                setting_flag = False
                for each_name in setting_mask:
                    if each_name in setting_id:
                        setting_flag=True
            else:
                setting_flag = True

            if len(wheather_mask)>0:
                wheather_flag = False
                for each_name in wheather_mask:
                    if each_name in seq_id:
                        wheather_flag=True
            else:
                wheather_flag = True

            if drop_rain and 'rain' in seq_id:
                drop_rain_flag = False
            else:
                drop_rain_flag = True

            if brightness > brightness_mask[0] and brightness < brightness_mask[1]:
                brightness_flag = True
            else:
                brightness_flag = False

            if setting_flag and wheather_flag and drop_rain_flag and brightness_flag:
                new_annos.append(info)

        laa_ads_fun = LAA3D_ADS_Metric(eval_config = self.dataset_cfg.LAA3D_ADS_METRIC, classes = self.class_name_dict, metric_save_path=metric_root_path)

        final_str = laa_ads_fun.eval(new_annos)

        return final_str

    def evaluation(self, annos, metric_root_path):

        print('evaluating !!!')

        i_str = self.evaluation_private(annos,metric_root_path)

        return i_str
