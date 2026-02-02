from .ap import calculate_object_detection_3dap
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .eval6dof import eval_6d_pose
import copy
from scipy.spatial.transform import Rotation as R
from .ap2d import calculate_object_detection_2dap
import cv2


class LAA3D_ADS_Metric():
    def __init__(self, eval_config=None, classes=['MAV', 'EVTOL', 'Helicopter'], metric_save_path=''):

        self.classes = classes
        if eval_config is None:
            self.dis_max = {'MAV': 100, 'EVTOL': 150, 'Helicopter': 300}
            self.ap2d_iou_thresholds = {'MAV': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                        'EVTOL': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                        'Helicopter': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
            self.ap2d_recall_num = 41
            self.ap3d_dis_threshold = {'MAV': [1, 2, 4, 8], 'EVTOL': [1, 2, 4, 8],
                                       'Helicopter': [1, 2, 4, 8, 16, 32, 64]}
            self.ap3d_recall_num = 41
            self.dof6_dis_norm_max = {'MAV': 8, 'EVTOL': 16, 'Helicopter': 64}
            self.dof6_ori_norm_max = {'MAV': 30, 'EVTOL': 30, 'Helicopter': 30}
            self.dof6_size_norm_max = {'MAV': 1, 'EVTOL': 5, 'Helicopter': 10}
            self.min_pixel = 32

        else:
            self.dis_max = eval_config.DisMax
            self.dis_min = eval_config.DisMin
            self.ap2d_iou_thresholds = eval_config.AP2D.IoUThresh
            self.ap2d_recall_num = eval_config.AP2D.RecallNum
            self.ap3d_dis_threshold = eval_config.AP3D.DisThresh
            self.ap3d_recall_num = eval_config.AP3D.RecallNum
            self.dof6_dis_norm_max = eval_config.Dof6.DisNormMax
            self.dof6_ori_norm_max = eval_config.Dof6.OriNormMax
            self.dof6_size_norm_max = eval_config.Dof6.SizeNormMax
            self.min_pixel = eval_config.MinPixel

        self.metric_save_root_path = metric_save_path

    def box9d_to_2d(self, boxes9d, img_width=1280., img_height=720., intrinsic_mat=None, extrinsic_mat=None,
                    distortion_matrix=None):

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

            corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))
            corners_camera = (extrinsic_mat @ corners_homogeneous.T).T[:, :3]

            rvec, _ = cv2.Rodrigues(extrinsic_mat[:3, :3])
            tvec = extrinsic_mat[:3, 3]

            corners_2d, _ = cv2.projectPoints(corners, rvec, tvec, intrinsic_mat, distortion_matrix)
            corners_2d = corners_2d.reshape(-1, 2).astype(int)

            corners_2d[:, 0] = np.clip(corners_2d[:, 0], a_min=0, a_max=int(img_width) - 1)
            corners_2d[:, 1] = np.clip(corners_2d[:, 1], a_min=0, a_max=int(img_height) - 1)

            x1, y1 = corners_2d.min(axis=0)
            x2, y2 = corners_2d.max(axis=0)
            boxes2d.append([x1, y1, x2, y2])
            size.append(max(abs(x2 - x1), abs(y2 - y1)))

        return np.array(boxes2d), np.array(size)

    def evaluation_6dof_by_name(self, annos,
                                metric_root_path,
                                cls_name,
                                max_ass_dis,
                                dof6_dis_norm_max,
                                dof6_ori_norm_max,
                                dof6_size_norm_max):

        orin_error, pos_error, size_error = eval_6d_pose(annos, max_dis=max_ass_dis)

        plt.scatter(np.arange(0, len(pos_error)), pos_error)
        plt.savefig(os.path.join(metric_root_path, cls_name + '_pos_error.png'))

        orin_error_mean = orin_error.mean()
        pos_error_mean = pos_error.mean()
        size_error_mean = size_error.mean()

        orin_error_median = np.median(orin_error)
        pose_error_median = np.median(pos_error)
        size_error_median = np.median(size_error)

        orin_error_new, pos_error_new, size_error_new = copy.deepcopy(orin_error), copy.deepcopy(
            pos_error), copy.deepcopy(size_error)

        orin_error_new[orin_error_new > dof6_ori_norm_max] = dof6_ori_norm_max
        pos_error_new[pos_error_new > dof6_dis_norm_max] = dof6_dis_norm_max
        size_error_new[size_error_new > dof6_size_norm_max] = dof6_size_norm_max

        orin_acc = np.mean(1 - orin_error_new / dof6_ori_norm_max) * 100
        pos_acc = np.mean(1 - pos_error_new / dof6_dis_norm_max) * 100
        size_acc = np.mean(1 - size_error_new / dof6_size_norm_max) * 100

        return_str = '\n\n' + cls_name + ':\n6dof error:\n    orin_error_mean(degree)： ' + str(orin_error_mean) + '\n' \
                     + '    pos_error_mean(m): ' + str(pos_error_mean) + '\n' \
                     + '    size_error_mean(m): ' + str(size_error_mean) + '\n' \
                     + '    orin_error_median(degree): ' + str(orin_error_median) + '\n' \
                     + '    pos_error_median(m): ' + str(pose_error_median) + '\n' \
                     + '    size_error_median(m): ' + str(size_error_median) + '\n' \
                     + '    orin_accuracy(%): ' + str(orin_acc) + '\n' \
                     + '    pos_accuracy(%): ' + str(pos_acc) + '\n' \
                     + '    size_accuracy(%): ' + str(size_acc)

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
        mean_error_pd = pd.DataFrame({cls_name + '_orin_error_mean': [orin_error_mean],
                                      cls_name + '_pose_error_mean': [pos_error_mean],
                                      cls_name + '_size_error_mean': [size_error_mean],
                                      cls_name + '_orin_error_median': [orin_error_median],
                                      cls_name + '_pose_error_median': [pose_error_median],
                                      cls_name + '_size_error_median': [size_error_median],
                                      cls_name + '_orin_accuracy': [orin_acc],
                                      cls_name + '_pose_accuracy': [pos_acc],
                                      cls_name + '_size_accuracy': [size_acc],
                                      })

        mean_error_pd.to_csv(mean_error_out_path)

        return return_str, orin_acc, pos_acc, size_acc

    def evaluation_3dAP_by_name(self, annos, metric_root_path, cls_name, match_threshold, recall_num=41, max_dis=100):

        cls_pred_boxes = []
        cls_pred_boxes_2d = []
        cls_pred_scores = []
        cls_pred_frame_id = []
        cls_gt_boxes = []
        cls_gt_boxes_2d = []
        cls_gt_frame_id = []

        for k, each_anno in enumerate(annos):
            pred_box2d = each_anno['pred_box2d']
            pred_box9d = each_anno['pred_box9d']
            pred_score = each_anno['pred_score']
            pred_frame_id = each_anno['pred_frame_id']
            gt_box2d = each_anno['gt_box2d']
            gt_box9d = each_anno['gt_box9d']
            gt_frame_id = each_anno['gt_frame_id']

            cls_gt_boxes.append(gt_box9d)
            cls_gt_boxes_2d.append(gt_box2d)
            cls_gt_frame_id.append(gt_frame_id)
            cls_pred_scores.append(pred_score)
            cls_pred_boxes.append(pred_box9d)
            cls_pred_boxes_2d.append(pred_box2d)
            cls_pred_frame_id.append(pred_frame_id)

        cls_pred_boxes = np.concatenate(cls_pred_boxes)
        cls_pred_boxes_2d = np.concatenate(cls_pred_boxes_2d)
        cls_pred_scores = np.concatenate(cls_pred_scores)
        cls_pred_frame_id = np.concatenate(cls_pred_frame_id)
        cls_gt_boxes = np.concatenate(cls_gt_boxes)
        cls_gt_boxes_2d = np.concatenate(cls_gt_boxes_2d)
        cls_gt_frame_id = np.concatenate(cls_gt_frame_id)

        AP3d, AP_under_thresholds, AP_detailed_results = calculate_object_detection_3dap(cls_pred_boxes,
                                                                                         cls_pred_scores,
                                                                                         cls_pred_frame_id,
                                                                                         cls_gt_boxes,
                                                                                         cls_gt_frame_id,
                                                                                         matching_thresholds=match_threshold,
                                                                                         recall_number=recall_num,
                                                                                         detection_distances=[max_dis])

        return_str = '\ndetection 3D AP:\n'

        detection_distance = [max_dis]

        return_str += ('    AP_R40_' + 'eval_distance_' + str(max_dis) + '(%): ' + str(AP3d * 100))

        dif_ap_value = AP_detailed_results

        for j, dis_precision_list in enumerate(dif_ap_value):

            return_str += ('\n    AP_R40_' + 'matching_distance_threshold_' + str(match_threshold[j]) + '(%): ' + str(
                AP_under_thresholds[j] * 100))

            save_path_csv = os.path.join(metric_root_path, cls_name + '_AP_R40_' + 'eval_distance_' + str(
                detection_distance[0]) + '_matching_thresh_' + str(match_threshold[j]) + '.csv')

            if len(dis_precision_list) == 0:
                dis_precision_list = [0] * recall_num

            pos_error_pd = pd.DataFrame({'recall_position': np.arange(0, 1, step=1 / recall_num) * 100,
                                         'precision': np.array(dis_precision_list) * 100})

            pos_error_pd.to_csv(save_path_csv)

        return return_str, AP3d * 100

    def evaluation_2dAP_by_name(self, annos, metric_root_path, cls_name, match_threshold, recall_num=41, max_dis=100):

        cls_pred_boxes = []
        cls_pred_boxes_2d = []
        cls_pred_scores = []
        cls_pred_frame_id = []
        cls_gt_boxes = []
        cls_gt_boxes_2d = []
        cls_gt_frame_id = []

        for k, each_anno in enumerate(annos):
            pred_box2d = each_anno['pred_box2d']
            pred_box9d = each_anno['pred_box9d']
            pred_score = each_anno['pred_score']
            pred_frame_id = each_anno['pred_frame_id']
            gt_box2d = each_anno['gt_box2d']
            gt_box9d = each_anno['gt_box9d']
            gt_frame_id = each_anno['gt_frame_id']

            cls_gt_boxes.append(gt_box9d)
            cls_gt_boxes_2d.append(gt_box2d)
            cls_gt_frame_id.append(gt_frame_id)
            cls_pred_scores.append(pred_score)
            cls_pred_boxes.append(pred_box9d)
            cls_pred_boxes_2d.append(pred_box2d)
            cls_pred_frame_id.append(pred_frame_id)

        cls_pred_boxes = np.concatenate(cls_pred_boxes)
        cls_pred_boxes_2d = np.concatenate(cls_pred_boxes_2d)
        cls_pred_scores = np.concatenate(cls_pred_scores)
        cls_pred_frame_id = np.concatenate(cls_pred_frame_id)
        cls_gt_boxes = np.concatenate(cls_gt_boxes)
        cls_gt_boxes_2d = np.concatenate(cls_gt_boxes_2d)
        cls_gt_frame_id = np.concatenate(cls_gt_frame_id)

        ap2d, AP_under_thresholds, detailed_values = calculate_object_detection_2dap(cls_pred_boxes_2d,
                                                                                     cls_pred_scores,
                                                                                     cls_pred_frame_id,
                                                                                     cls_gt_boxes_2d,
                                                                                     cls_gt_frame_id,
                                                                                     matching_thresholds=match_threshold,
                                                                                     recall_number=recall_num)

        return_str = '\ndetection 2D AP: \n'
        return_str += ('    AP_R40_' + 'eval_distance_' + str(max_dis) + '(%): ' + str(ap2d * 100))

        precs = detailed_values[0]
        overall_precision = 0
        recall = 0
        for p_i in range(len(precs)):
            if precs[len(precs)-p_i-1]>0:
                overall_precision= precs[len(precs)-p_i-1]
                recall = (len(precs)-p_i)*(1/recall_num)
                break

        return_str += ('\n    Precision_' + 'eval_distance_' + str(max_dis) + '(%): ' + str(overall_precision))
        return_str += ('\n    Recall_' + 'eval_distance_' + str(max_dis) + '(%): ' + str(recall))

        for j, iou_precision_list in enumerate(detailed_values):

            return_str += ('\n    AP_R40_' + 'matching_IoU_threshold_' + str(match_threshold[j]) + '(%): ' + str(
                AP_under_thresholds[j] * 100))

            save_path_csv = os.path.join(metric_root_path, cls_name + '_AP_R40_' + 'eval_distance_' + str(
                max_dis) + '_matching_thresh_' + str(match_threshold[j]) + '.csv')

            if len(iou_precision_list) == 0:
                iou_precision_list = np.array([0] * recall_num)

            pos_error_pd = pd.DataFrame({'recall_position': np.arange(0, 1, step=1 / recall_num) * 100,
                                         'precision': iou_precision_list * 100})

            pos_error_pd.to_csv(save_path_csv)

        return return_str, ap2d * 100

    def eval(self, annos):
        """
        annos: list
        """

        path_6dof = os.path.join(self.metric_save_root_path, 'dof6_evaluation_results')
        path_3dap = os.path.join(self.metric_save_root_path, 'ap3d_evaluation_results')
        path_2dap = os.path.join(self.metric_save_root_path, 'ap2d_evaluation_results')

        os.makedirs(path_6dof, exist_ok=True)
        os.makedirs(path_3dap, exist_ok=True)
        os.makedirs(path_2dap, exist_ok=True)

        final_str = ''

        for cls_n in self.classes:
            print('##' * 15)
            print('computing metrics for class ', cls_n, '!!!')
            new_annos = []
            this_annos_pred = copy.deepcopy(annos)

            cls_max_dis = self.dis_max[cls_n]
            cls_min_dis = self.dis_min[cls_n]

            ap2d_iou_thresholds = self.ap2d_iou_thresholds[cls_n]
            ap2d_recall_num = self.ap2d_recall_num
            ap3d_dis_threshold = self.ap3d_dis_threshold[cls_n]
            ap3d_recall_num = self.ap3d_recall_num
            dof6_dis_norm_max = self.dof6_dis_norm_max[cls_n]
            dof6_ori_norm_max = self.dof6_ori_norm_max[cls_n]
            dof6_size_norm_max = self.dof6_size_norm_max[cls_n]

            for k, each_anno in enumerate(this_annos_pred):

                new_anno = {}
                gt_name = each_anno['gt_name']
                gt_box = each_anno['gt_box9d'][:, 0:9]
                gt_diff = each_anno['gt_diff']
                pred_name = each_anno['pred_name']
                pre_box = each_anno['pred_box9d'][:, 0:9]
                pre_score = each_anno['confidence']

                intrinsic = each_anno['intrinsic']
                extrinsic = each_anno['extrinsic']
                distortion = each_anno['distortion']

                z = gt_box.__len__() - 1

                while z > 0 and gt_box[z].sum() == 0:
                    z -= 1

                gt_box = gt_box[:z + 1]


                name_mask = gt_name == cls_n
                dis_mask = np.linalg.norm(gt_box[:, 0:3], axis=-1) < cls_max_dis
                dis_mask_min = cls_min_dis < np.linalg.norm(gt_box[:, 0:3], axis=-1)
                if len(gt_box) > 0:
                    gt_box2d, size = self.box9d_to_2d(gt_box, intrinsic_mat=intrinsic, extrinsic_mat=extrinsic,
                                                      distortion_matrix=distortion)
                    size_m = size >= self.min_pixel
                    mask = name_mask * dis_mask * dis_mask_min * size_m
                    new_gt_box9d = gt_box[mask]
                    gt_box2d = gt_box2d[mask]
                    gt_frame_id = np.ones((gt_box2d.shape[0])) * k
                    if len(new_gt_box9d) == 0:
                        gt_box2d = np.empty((0, 4))
                        new_gt_box9d = np.empty((0, 9))
                        gt_frame_id = np.empty(0)
                else:
                    gt_box2d = np.empty((0, 4))
                    new_gt_box9d = np.empty((0, 9))
                    gt_frame_id = np.empty(0)

                new_anno['gt_box9d'] = new_gt_box9d
                new_anno['gt_frame_id'] = gt_frame_id
                new_anno['gt_box2d'] = gt_box2d

                name_mask2 = pred_name == cls_n
                dis_mask2 = np.linalg.norm(pre_box[:, 0:3], axis=-1) < cls_max_dis
                dis_mask2_min = cls_min_dis < np.linalg.norm(pre_box[:, 0:3], axis=-1)

                if len(pre_box) > 0:
                    pred_box2d, size2 = self.box9d_to_2d(pre_box, intrinsic_mat=intrinsic, extrinsic_mat=extrinsic,
                                                         distortion_matrix=distortion)
                    mask_s2 = size2 >= self.min_pixel

                    mask2 = name_mask2 * dis_mask2 * dis_mask2_min * mask_s2
                    new_pred_box9d = pre_box[mask2]
                    new_pred_score = pre_score[mask2]
                    pred_box2d = pred_box2d[mask2]
                    pred_frame_id = np.ones_like(new_pred_score) * k
                    if len(new_pred_box9d) == 0:
                        pred_box2d = np.empty((0, 4))
                        new_pred_box9d = np.empty((0, 9))
                        new_pred_score = np.empty(0)
                        pred_frame_id = np.empty(0)
                else:
                    pred_box2d = np.empty((0, 4))
                    new_pred_box9d = np.empty((0, 9))
                    new_pred_score = np.empty(0)
                    pred_frame_id = np.empty(0)

                new_anno['pred_box9d'] = new_pred_box9d
                new_anno['pred_score'] = new_pred_score
                new_anno['pred_box2d'] = pred_box2d
                new_anno['pred_frame_id'] = pred_frame_id

                new_annos.append(new_anno)

            print('computing pose metric !!!')
            acc_6dof_str, orin_acc, pos_acc, size_acc = self.evaluation_6dof_by_name(new_annos,
                                                                                     path_6dof,
                                                                                     cls_name=cls_n,
                                                                                     max_ass_dis=cls_max_dis,
                                                                                     dof6_dis_norm_max=dof6_dis_norm_max,
                                                                                     dof6_ori_norm_max=dof6_ori_norm_max,
                                                                                     dof6_size_norm_max=dof6_size_norm_max
                                                                                     )

            print('computing 3D AP metric !!!')
            ap3d_str, ap3d = self.evaluation_3dAP_by_name(new_annos,
                                                          path_3dap,
                                                          cls_name=cls_n,
                                                          match_threshold=ap3d_dis_threshold,
                                                          recall_num=ap3d_recall_num,
                                                          max_dis=cls_max_dis)

            print('computing 2D AP metric !!!')
            ap2d_str, ap2d = self.evaluation_2dAP_by_name(new_annos,
                                                          path_2dap,
                                                          cls_name=cls_n,
                                                          match_threshold=ap2d_iou_thresholds,
                                                          recall_num=ap2d_recall_num,
                                                          max_dis=cls_max_dis)

            final_ads = np.mean([ap3d, ap2d, orin_acc, pos_acc, size_acc])

            final_str = final_str + acc_6dof_str + ap3d_str + ap2d_str

            final_str += '\nfinal_LAA3D_ADS_' + cls_n + '(%): ' + str(final_ads) + '\n'

        return final_str

