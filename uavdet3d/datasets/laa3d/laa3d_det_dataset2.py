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
import pickle as pkl
from .ap import calculate_object_detection_3dap

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

class LAA3DDataset():
    def __init__(self, root_path, split = 'val2', interval=200):

        self.root_path = root_path
        self.split = split
        self.interval = interval

        self.data_info = None

        with open(os.path.join(self.root_path, self.split+'_info.pkl'), 'rb') as f:
            self.data_info = pkl.load(f)

        data_info = []

        for i in range(0,len(self.data_info), interval):
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

        info = self.data_info[item]

        setting_id = info['setting_id']
        seq_id = info['seq_id']

        frame_id = info['frame_id']
        relative_im_path = info['relative_im_path']

        annos = info['annos']

        this_boxes = annos['box']  # 3D boxes
        this_boxes_2d = annos['box2d']  # 2D boxes

        diff = annos['diff'] # difficulty level
        this_coarse_class = annos['coarse_class'] # object classes
        this_fine_class = annos['fine_class'] # object classes
        this_id = annos['ob_id']  # object identity

        this_boxes = this_boxes[diff<6]

        print(this_id)

        image_info = info['image_info']
        intrinsic = image_info['intrinsic']
        extrinsic = image_info['extrinsic']

        image = cv2.imread(os.path.join(self.root_path, relative_im_path))

        new_im = draw_box9d_on_image(this_boxes, image, intrinsic_mat=intrinsic)

        for bbox in this_boxes_2d:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(new_im, (x1, y1), (x2, y2), (100, 0, 100), 2)

        cv2.imshow('im', new_im)
        cv2.waitKey(0)


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

        frame_id = info['frame_id']
        relative_im_path = info['relative_im_path']

        annos = info['annos']

        boxes9d = annos['box']  # 3D boxes
        this_boxes_2d = annos['box2d']  # 2D boxes

        diff = annos['diff']  # difficulty level
        this_coarse_class = annos['coarse_class']  # object classes
        this_fine_class = annos['fine_class']  # object classes
        this_id = annos['ob_id']  # object identity

        boxes9d = boxes9d[diff < 6]
        this_coarse_class = this_coarse_class[diff < 6]
        diff = diff[diff<6]

        image_info = info['image_info']
        intrinsic = image_info['intrinsic']
        extrinsic = image_info['extrinsic']

        image = cv2.imread(os.path.join(self.root_path, relative_im_path).replace('\\', '/')).astype(np.float32)

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
            relative_im_path = batch_dict['relative_im_path'][batch_id]

            intrinsic = batch_dict['intrinsic'][batch_id] # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id] # 4, 4
            distortion = batch_dict['distortion'][batch_id] # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id] # 2,
            new_im_size = batch_dict['new_im_size'][batch_id] # 2,

            # key_points_2d = batch_dict['key_points_2d'][batch_id]
            confidence = batch_dict['confidence'][batch_id]
            im_path = os.path.join(self.root_path, relative_im_path)

            pred_name = self.name_from_code(pred_boxes9d[:,-1])

            if len(pred_boxes9d)>0:
                boxes2d,_ = box9d_to_2d(boxes9d=pred_boxes9d[:,0:9], intrinsic_mat=intrinsic[0],extrinsic_mat=extrinsic[0])
                nms_mask = nms_numpy(boxes2d,confidence,iou_threshold=0.01)
                pred_boxes9d = pred_boxes9d[nms_mask]
                pred_name = pred_name[nms_mask]
                confidence = confidence[nms_mask]

            frame_dict = {'seq_id': seq_id,
                          'frame_id': frame_id,
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
            # pred_boxes9d[:,3:6]*=1.5
            # # #
            # image = draw_box9d_on_image(pred_boxes9d, image,
            #                             img_width=raw_im_size[0],
            #                             img_height=raw_im_size[1],
            #                             color=(255, 0, 0),
            #                             intrinsic_mat=intrinsic[0],
            #                             extrinsic_mat=extrinsic[0],
            #                             distortion_matrix=distortion[0],
            #                             offset=np.array([0., 0., 0.]))
            #
            #
            #
            # image = draw_box9d_on_image(gt_box9d, image,
            #                             img_width=raw_im_size[0],
            #                             img_height=raw_im_size[1],
            #                             color=(0, 0, 255),
            #                             intrinsic_mat=intrinsic[0],
            #                             extrinsic_mat=extrinsic[0],
            #                             distortion_matrix=distortion[0],
            #                             offset=np.array([0., 0., 0.]))
            #
            # cv2.imshow('im', image)
            # cv2.waitKey(0)

        return annos

    def evaluation_by_name(self, annos, metric_root_path, cls_name):

        orin_error, pos_error, size_error = eval_6d_pose(annos, max_dis=self.dataset_cfg.MAX_DIS)

        plt.scatter(np.arange(0,len(pos_error)), pos_error)
        plt.savefig(os.path.join(metric_root_path, cls_name+'_pos_error.png'))

        print(cls_name+'_orin_error： ', orin_error.mean())
        print(cls_name+'_pose_error: ', pos_error.mean())
        print(cls_name+'_size_error: ', size_error.mean())

        print(cls_name+'_orin_error_median: ', np.median(orin_error))
        print(cls_name+'_pose_error_median: ', np.median(pos_error))
        print(cls_name+'_size_error_median: ', np.median(size_error))

        return_str = '\n' + cls_name + ': \n orin_error： ' + str(orin_error.mean()) + '\n' \
                     + 'pose_error: ' + str(pos_error.mean()) + '\n' \
                     + 'size_error: ' + str(size_error.mean()) + '\n' \
                     + 'orin_error_median: ' + str(np.median(orin_error)) + '\n' \
                     + 'pos_error_median: ' + str(np.median(pos_error)) + '\n' \
                     + 'pos_error_median: ' + str(np.median(size_error)) + '\n' \

        orin_error_out_path = os.path.join(str(metric_root_path), cls_name + '_orin_error.csv')
        orin_error_pd = pd.DataFrame({cls_name + '_orin_error': orin_error.tolist()})
        orin_error_pd.to_csv(orin_error_out_path)

        pos_error_out_path = os.path.join(str(metric_root_path), cls_name + '_pos_error.csv')
        pos_error_pd = pd.DataFrame({cls_name + '_pos_error': pos_error.tolist()})
        pos_error_pd.to_csv(pos_error_out_path)

        size_error_out_path = os.path.join(str(metric_root_path), cls_name + '_size_error.csv')
        size_error_pd = pd.DataFrame({cls_name + '_size_error': size_error.tolist()})
        size_error_pd.to_csv(size_error_out_path)

        mean_error_out_path = os.path.join(str(metric_root_path), cls_name + '_mean_median_error.csv')
        mean_error_pd = pd.DataFrame({cls_name + '_orin_error_mean': [orin_error.mean()],
                                      cls_name + '_pose_error_mean': [pos_error.mean()],
                                      cls_name + '_size_error_mean': [size_error.mean()],
                                      cls_name + '_orin_error_median': [orin_error.median()],
                                      cls_name + '_pose_error_median': [pos_error.median()],
                                      cls_name + '_size_error_median': [size_error.median()], })

        mean_error_pd.to_csv(mean_error_out_path)

        return return_str

    def evaluation_AP_by_name(self, annos, metric_root_path, cls_name):

        pred_boxes = []
        pred_scores = []
        pred_frame_id = []
        gt_boxes = []
        gt_difficulty_level = []
        gt_frame_id = []


        for i, each_anno in enumerate(annos):

            pre_box = each_anno['pred_box9d']
            pre_score = each_anno['confidence']
            gt_box = each_anno['gt_box9d']
            gt_diff = each_anno['gt_diff']

            if len(pre_box)>0:
                pred_frame = np.ones_like(pre_score)*i

                pred_boxes.append(pre_box)
                pred_scores.append(pre_score)
                pred_frame_id.append(pred_frame)

            if len(gt_box)>0:
                gt_frame = np.ones_like(gt_diff)*i

                gt_boxes.append(gt_box)
                gt_difficulty_level.append(gt_diff)
                gt_frame_id.append(gt_frame)


        if len(pred_boxes)<=1 or len(gt_boxes)<=1:

            return ''


        pred_boxes = np.concatenate(pred_boxes)
        pred_scores = np.concatenate(pred_scores)
        pred_frame_id = np.concatenate(pred_frame_id)
        gt_boxes = np.concatenate(gt_boxes)
        gt_difficulty_level = np.concatenate(gt_difficulty_level)
        gt_frame_id = np.concatenate(gt_frame_id)

        AP_results, AP_detailed_results = calculate_object_detection_3dap(pred_boxes,
                                                   pred_scores,
                                                   pred_frame_id,
                                                   gt_boxes,
                                                   gt_frame_id)


        return_str = '\n detection AP:\n' + cls_name

        detection_distance = [20,40,80,160,300]

        for i, dif_ap in enumerate(AP_results):
            print(cls_name + '_AP_R40_'+'detection_distance_'+str(detection_distance[i])+': ', dif_ap)

            return_str+=('\n AP_R40_'+'detection_distance_'+str(detection_distance[i])+': '+str(dif_ap))

            dif_ap_value = AP_detailed_results[i]

            dis_thresholds = [1,2,4,8]

            for j, dis_precision_list in enumerate(dif_ap_value) :

                save_path_csv = os.path.join(metric_root_path, cls_name + '_AP_R40_'+'detection_distance_'+str(detection_distance[i])+'_matching_thresh_'+str(dis_thresholds[j])+'_'+str(np.mean(dis_precision_list))+'.csv')

                if len(dis_precision_list)==0:
                    dis_precision_list = [0]*40

                # print(len(np.arange(0,1, step=1/40.)))
                # print(len(dis_precision_list))
                # input()

                pos_error_pd = pd.DataFrame({ 'recall_position': np.arange(0,1, step=1/40.),
                                              'precision': dis_precision_list})
                pos_error_pd.to_csv(save_path_csv)

        return return_str


    def evaluation(self, annos, metric_root_path):

        all_names = self.class_name_dict

        final_str = ''

        all_annos = []

        for cls_n in all_names:

            this_annos_pred = copy.deepcopy(annos)

            new_annos = []

            for each_anno in this_annos_pred:

                new_anno = {}
                gt_name = each_anno['gt_name']
                gt_box = each_anno['gt_box9d']
                gt_diff = each_anno['gt_diff']
                pred_name = each_anno['pred_name']
                pre_box = each_anno['pred_box9d']
                pre_score = each_anno['confidence']

                k = gt_box.__len__()-1

                while k>0 and gt_box[k].sum()==0:
                    k-=1

                gt_box = gt_box[:k+1]

                name_mask = gt_name==cls_n
                new_anno['gt_box9d'] = gt_box[name_mask]
                new_gt = gt_box[name_mask]
                new_anno['gt_diff'] = gt_diff[name_mask]

                name_mask = pred_name==cls_n

                new_anno['pred_box9d'] = pre_box[name_mask] # new_gt#
                new_anno['confidence'] = pre_score[name_mask]  #np.random.randn(len(new_gt))  #



                if new_anno['gt_box9d'].shape[0]>=1:

                    new_annos.append(new_anno)
                    all_annos.append(copy.deepcopy(new_anno))

            if len(new_annos)>=1:
                this_str = self.evaluation_by_name(new_annos, metric_root_path, cls_name = cls_n)

                final_str+=this_str

                ap_str = self.evaluation_AP_by_name(new_annos, metric_root_path, cls_name = cls_n)

                final_str+=ap_str


        this_str = self.evaluation_by_name(all_annos, metric_root_path, cls_name='all')

        final_str += this_str

        ap_str = self.evaluation_AP_by_name(all_annos, metric_root_path, cls_name = 'all')

        final_str += ap_str

        return final_str



