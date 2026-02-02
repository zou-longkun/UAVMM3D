import os
import torch
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uavdet3d.datasets.laam6d.eval6dof_laam6d import eval_6d_posi
from uavdet3d.datasets.laam6d.ap2d_laam6d import calculate_object_detection_2dap
from .ap_laam6d import calculate_object_detection_3dap


class LAA3D_ADS_Metric():
    def __init__(self, eval_config=None, classes=None, metric_save_path=''):

        self.classes = classes
        if eval_config is None:
            self.dis_max = 150
            self.ap2d_iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            self.ap2d_recall_num = 41
            self.ap3d_dis_threshold = [1, 2, 4, 8]
            self.ap3d_recall_num = 41
            self.dof6_dis_norm_max = 8
            self.dof6_ori_norm_max = 30
            self.dof6_size_norm_max = 1
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

    def box9d_to_2d(self, boxes9d,
                    img_width=1280.0,
                    img_height=720.0,
                    intrinsic_mat=None,
                    extrinsic_mat=None,
                    distortion_matrix=None,
                    eps_z=1e-6,
                    clip_to_image=True):
        """
        boxes9d: (N, 9, 3) 其中 box[0]是center, box[1:]是8个角点
        返回:
        boxes2d: (N,4) int32, [x1,y1,x2,y2]
        size:    (N,) float32, max(w,h)
        valid:   (N,) bool, 投影有效且形成合理2D框
        """

        if intrinsic_mat is None:
            intrinsic_mat = np.array([
                [img_width, 0, img_width / 2],
                [0, img_height, img_height / 2],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            intrinsic_mat = np.asarray(intrinsic_mat, dtype=np.float64)

        if extrinsic_mat is None:
            extrinsic_mat = np.eye(4, dtype=np.float64)
        else:
            extrinsic_mat = np.asarray(extrinsic_mat, dtype=np.float64)

        if distortion_matrix is None:
            distortion_matrix = np.zeros(5, dtype=np.float64)
        else:
            distortion_matrix = np.asarray(distortion_matrix, dtype=np.float64).reshape(-1)

        # Carla → OpenCV 相机坐标变换（你原来的）
        carla_to_opencv = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        extrinsic_inv = np.linalg.inv(extrinsic_mat)

        boxes9d = np.asarray(boxes9d, dtype=np.float64)
        N = len(boxes9d)

        boxes2d = np.zeros((N, 4), dtype=np.int32)
        size = np.zeros((N,), dtype=np.float32)
        valid = np.ones((N,), dtype=bool)

        # 正确的 rvec/tvec（注意：projectPoints 要 Rodrigues rvec，不要传 3x3 矩阵）
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        for i, box in enumerate(boxes9d):
            # box: (9,3)
            # 转到相机系（OpenCV）
            pts_cv = []
            for pt in box:
                pt_hom = np.append(pt, 1.0)  # (4,)
                pt_cam = extrinsic_inv @ pt_hom
                pt_cv_hom = carla_to_opencv @ pt_cam
                pts_cv.append(pt_cv_hom[:3])

            pts_cv = np.asarray(pts_cv, dtype=np.float64)  # (9,3)
            corners_cv = pts_cv[1:]  # (8,3)

            # 关键：如果有点在相机后方/太接近 z=0，投影会爆（inf/NaN）
            # OpenCV相机系里一般 z>0 才可投影
            if np.any(corners_cv[:, 2] <= eps_z) or np.any(~np.isfinite(corners_cv)):
                valid[i] = False
                continue

            corners_2d, _ = cv2.projectPoints(
                corners_cv, rvec, tvec, intrinsic_mat, distortion_matrix
            )

            if corners_2d is None or (not np.isfinite(corners_2d).all()):
                valid[i] = False
                continue

            corners_2d = corners_2d.reshape(-1, 2)  # float64

            x1, y1 = corners_2d.min(axis=0)
            x2, y2 = corners_2d.max(axis=0)

            # 过滤掉退化框（太小/反向）
            if (not np.isfinite([x1, y1, x2, y2]).all()) or (x2 <= x1) or (y2 <= y1):
                valid[i] = False
                continue

            if clip_to_image:
                x1 = np.clip(x1, 0, img_width - 1)
                x2 = np.clip(x2, 0, img_width - 1)
                y1 = np.clip(y1, 0, img_height - 1)
                y2 = np.clip(y2, 0, img_height - 1)
                if (x2 <= x1) or (y2 <= y1):
                    valid[i] = False
                    continue

            boxes2d[i] = [int(x1), int(y1), int(x2), int(y2)]
            size[i] = float(max(abs(x2 - x1), abs(y2 - y1)))

        return boxes2d, size, valid
    
    def evaluation_6dof_by_name(self, annos,
                                metric_root_path,
                                cls_name,
                                max_ass_dis,
                                dof6_dis_norm_max,
                                dof6_ori_norm_max,
                                dof6_size_norm_max):

        orin_error, posi_error, size_error = eval_6d_posi(annos, max_dis=max_ass_dis)

        plt.scatter(np.arange(0, len(posi_error)), posi_error)
        plt.savefig(os.path.join(metric_root_path, cls_name + '_posi_error.png'))

        orin_error_mean = orin_error.mean()
        posi_error_mean = posi_error.mean()
        size_error_mean = size_error.mean()

        orin_error_median = np.median(orin_error)
        posi_error_median = np.median(posi_error)
        size_error_median = np.median(size_error)

        orin_error_new, posi_error_new, size_error_new = copy.deepcopy(orin_error), copy.deepcopy(posi_error), copy.deepcopy(size_error)

        orin_error_new[orin_error_new > dof6_ori_norm_max] = dof6_ori_norm_max
        posi_error_new[posi_error_new > dof6_dis_norm_max] = dof6_dis_norm_max
        size_error_new[size_error_new > dof6_size_norm_max] = dof6_size_norm_max

        if np.all(orin_error_new == 0):
            orin_acc = 0.0
        else:
            orin_acc = np.mean(1 - orin_error_new / dof6_ori_norm_max) * 100
        if np.all(posi_error_new == 0):
            posi_acc = 0.0
        else:
            posi_acc = np.mean(1 - posi_error_new / dof6_dis_norm_max) * 100
        if np.all(size_error_new == 0):
            size_acc = 0.0
        else:
            size_acc = np.mean(1 - size_error_new / dof6_size_norm_max) * 100

        return_str = '\n\n' + cls_name + ':\n6dof error:\n    orin_error_mean(degree)： ' + str(orin_error_mean) + '\n' \
                     + '    posi_error_mean(m): ' + str(posi_error_mean) + '\n' \
                     + '    size_error_mean(m): ' + str(size_error_mean) + '\n' \
                     + '    orin_error_median(degree): ' + str(orin_error_median) + '\n' \
                     + '    posi_error_median(m): ' + str(posi_error_median) + '\n' \
                     + '    size_error_median(m): ' + str(size_error_median) + '\n' \
                     + '    orin_accuracy(%): ' + str(orin_acc) + '\n' \
                     + '    posi_accuracy(%): ' + str(posi_acc) + '\n' \
                     + '    size_accuracy(%): ' + str(size_acc)

        orin_error_out_path = os.path.join(str(metric_root_path), cls_name + '_orin_error.csv')
        orin_error_pd = pd.DataFrame({cls_name + '_orin_error': orin_error.tolist()})
        orin_error_pd.to_csv(orin_error_out_path)

        posi_error_out_path = os.path.join(str(metric_root_path), cls_name + '_posi_error.csv')
        posi_error_pd = pd.DataFrame({cls_name + '_posi_error': posi_error.tolist()})
        posi_error_pd.to_csv(posi_error_out_path)

        size_error_out_path = os.path.join(str(metric_root_path), cls_name + '_size_error.csv')
        size_error_pd = pd.DataFrame({cls_name + '_size_error': size_error.tolist()})
        size_error_pd.to_csv(size_error_out_path)

        mean_error_out_path = os.path.join(str(metric_root_path), cls_name + '_mean_median_error.csv')
        mean_error_pd = pd.DataFrame({cls_name + '_orin_error_mean': [orin_error_mean],
                                      cls_name + '_posi_error_mean': [posi_error_mean],
                                      cls_name + '_size_error_mean': [size_error_mean],
                                      cls_name + '_orin_error_median': [orin_error_median],
                                      cls_name + '_posi_error_median': [posi_error_median],
                                      cls_name + '_size_error_median': [size_error_median],
                                      cls_name + '_orin_accuracy': [orin_acc],
                                      cls_name + '_posi_accuracy': [posi_acc],
                                      cls_name + '_size_accuracy': [size_acc],
                                      })

        mean_error_pd.to_csv(mean_error_out_path)

        return return_str, orin_acc, posi_acc, size_acc

    def evaluation_3dAP_by_name(self, annos, 
                                metric_root_path, 
                                cls_name, 
                                match_threshold, 
                                recall_num=41, 
                                max_dis=100):

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

            posi_error_pd = pd.DataFrame({'recall_position': np.arange(0, 1, step=1 / recall_num) * 100,
                                         'precision': np.array(dis_precision_list) * 100})

            posi_error_pd.to_csv(save_path_csv)

        return return_str, AP3d * 100

    def evaluation_2dAP_by_name(self, annos, 
                                metric_root_path, 
                                cls_name, 
                                match_threshold, 
                                recall_num=41, 
                                max_dis=100):

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

        # 处理detailed_values为空或不完整的情况
        overall_precision = 0.0
        recall = 0.0
        if detailed_values and len(detailed_values) > 0 and len(detailed_values[0]) > 0:
            precs = detailed_values[0]
            for p_i in range(len(precs)):
                if precs[len(precs) - p_i - 1] > 0:
                    overall_precision = precs[len(precs) - p_i - 1]
                    recall = (len(precs) - p_i) * (1 / recall_num)
                    break

        # 转换为百分比形式以保持一致性
        return_str += ('\n    Precision_' + 'eval_distance_' + str(max_dis) + '(%): ' + str(overall_precision * 100))
        # 转换为百分比形式以保持一致性
        return_str += ('\n    Recall_' + 'eval_distance_' + str(max_dis) + '(%): ' + str(recall * 100))

        for j, iou_precision_list in enumerate(detailed_values):

            return_str += ('\n    AP_R40_' + 'matching_IoU_threshold_' + str(match_threshold[j]) + '(%): ' + str(
                AP_under_thresholds[j] * 100))

            save_path_csv = os.path.join(metric_root_path, cls_name + '_AP_R40_' + 'eval_distance_' + str(
                max_dis) + '_matching_thresh_' + str(match_threshold[j]) + '.csv')

            if len(iou_precision_list) == 0:
                iou_precision_list = np.array([0] * recall_num)

            posi_error_pd = pd.DataFrame({'recall_position': np.arange(0, 1, step=1 / recall_num) * 100,
                                         'precision': iou_precision_list * 100})

            posi_error_pd.to_csv(save_path_csv)

        return return_str, ap2d * 100

    def eval(self, annos):
        """
        评估函数：计算6DoF精度、3D AP、2D AP
        Args:
            annos: list - 样本列表，每个样本含gt_boxes、gt_names、pred_boxes等字段
        Returns:
            final_str: str - 所有类别评估结果的汇总字符串
        """
        # 1. 创建评估结果保存路径（与函数参数名metric_root_path对应）
        path_6dof = os.path.join(self.metric_save_root_path, 'dof6_evaluation_results')
        path_3dap = os.path.join(self.metric_save_root_path, 'ap3d_evaluation_results')
        path_2dap = os.path.join(self.metric_save_root_path, 'ap2d_evaluation_results')
        # 确保路径存在（不存在则创建）
        os.makedirs(path_6dof, exist_ok=True)
        os.makedirs(path_3dap, exist_ok=True)
        os.makedirs(path_2dap, exist_ok=True)

        final_str = ''  # 存储最终评估结果字符串

        # 2. 按类别循环评估（每个类别单独计算指标）
        for cls_n in self.classes:
            print('=' * 30)
            print(f'[EVAL] 开始计算类别 {cls_n} 的评估指标')
            print('=' * 30)
            new_annos = []  # 存储处理后的数据（对应函数的annos参数）
            this_annos_pred = copy.deepcopy(annos)  # 深拷贝避免修改原始数据

            # 3. 加载当前类别的评估参数（与函数参数一一对应）
            cls_max_dis = self.dis_max  # 对应函数的max_ass_dis参数
            ap2d_iou_thresholds = self.ap2d_iou_thresholds  # 2D AP的IoU阈值
            ap2d_recall_num = self.ap2d_recall_num  # 2D AP的召回数
            ap3d_dis_threshold = self.ap3d_dis_threshold  # 3D AP的距离匹配阈值
            ap3d_recall_num = self.ap3d_recall_num  # 3D AP的召回数
            # 6DoF专用参数（与evaluation_6dof_by_name的参数完全对应）
            dof6_dis_norm_max = self.dof6_dis_norm_max
            dof6_ori_norm_max = self.dof6_ori_norm_max
            dof6_size_norm_max = self.dof6_size_norm_max

            # 4. 逐样本处理（核心逻辑：CPU转换 + 数据修复 + 筛选）
            for k, each_anno in enumerate(this_annos_pred):
                new_anno = {}  # 存储当前样本处理后的数据

                # ---------------------- 核心1：统一处理CUDA张量转CPU NumPy ----------------------
                def to_cpu_numpy(data):
                    """辅助函数：将PyTorch张量（CPU/GPU）转为NumPy数组，非张量直接转NumPy"""
                    import torch
                    if isinstance(data, torch.Tensor):
                        return data.cpu().numpy() if data.is_cuda else data.numpy()
                    elif isinstance(data, np.ndarray):
                        return data
                    else:
                        return np.array(data)

                # 提取样本定位信息（用于调试输出）
                seq_id = to_cpu_numpy(each_anno.get('seq_id', 'unknown_seq'))
                frame_id = to_cpu_numpy(each_anno.get('frame_id', 'unknown_frame'))
                # 处理单元素数组的情况，避免打印冗余格式
                if isinstance(seq_id, np.ndarray):
                    seq_id = seq_id.item() if seq_id.size == 1 else seq_id
                if isinstance(frame_id, np.ndarray):
                    frame_id = frame_id.item() if frame_id.size == 1 else frame_id

                # 提取并转换所有关键数据（一次性转为CPU NumPy）
                gt_name = to_cpu_numpy(each_anno['gt_names'])
                gt_box = to_cpu_numpy(each_anno['gt_boxes'])
                pred_name = to_cpu_numpy(each_anno['pred_names'])
                pre_box = to_cpu_numpy(each_anno['pred_boxes'])
                pre_score = to_cpu_numpy(each_anno['confidence'])
                intrinsic = to_cpu_numpy(each_anno['intrinsic'])
                extrinsic = to_cpu_numpy(each_anno['extrinsic'])
                distortion = to_cpu_numpy(each_anno['distortion'])

                # ---------------------- 核心2：修复gt_names与gt_boxes长度不匹配 ----------------------
                orig_name_len = len(gt_name)
                orig_box_len = len(gt_box)
                if orig_name_len != orig_box_len:
                    print(f"\n[WARNING] 样本 seq_id={seq_id}, frame_id={frame_id} 数据不匹配：")
                    print(f"          gt_names长度={orig_name_len}, gt_boxes长度={orig_box_len}，已自动修复")

                    if orig_name_len > orig_box_len:
                        gt_name = gt_name[:orig_box_len]
                        print(f"          → 截断gt_names：从{orig_name_len}个减至{orig_box_len}个")
                    else:
                        gt_box  = gt_box[:orig_name_len]
                        print(f"          → 截断gt_boxes：从{orig_box_len}个减至{orig_name_len}个")
                        # missing_count = orig_box_len - orig_name_len
                        # missing_names = np.full(
                        #     shape=(missing_count,),
                        #     fill_value="Unknown",
                        #     dtype=gt_name.dtype
                        # )
                        # gt_name = np.concatenate([gt_name, missing_names], axis=0)
                        # print(f"          → 补全gt_names：新增{missing_count}个'Unknown'，补全后长度={len(gt_name)}")

                # 强制校验：修复后长度必须一致
                assert len(gt_name) == len(gt_box), \
                    f"样本 seq_id={seq_id}, frame_id={frame_id} 修复失败！" \
                    f"gt_names长度={len(gt_name)}, gt_boxes长度={len(gt_box)}"

                # ---------------------- 核心3：筛选当前类别的GT框 ----------------------
                if len(self.classes) == 1:
                    name_mask = np.ones_like(gt_name, dtype=bool)
                else:
                    name_mask = gt_name == cls_n
                # 计算距离掩码（gt_box已转CPU NumPy，无需.cpu()）
                gt_centers = gt_box[:, 0]
                gt_center_dist = np.linalg.norm(gt_centers, axis=-1)
                dis_mask = gt_center_dist < self.dis_max
                dis_mask_min = gt_center_dist > self.dis_min 

                # 生成2D框并筛选尺寸
                if len(gt_box) > 0:
                    gt_box2d, size, valid2d = self.box9d_to_2d(
                        gt_box,
                        intrinsic_mat=intrinsic,
                        extrinsic_mat=extrinsic,
                        distortion_matrix=distortion
                    )
                    size_m = size >= self.min_pixel

                    # --- 在 GT 部分，mask 计算前加 ---
                    # N = len(gt_box)
                    # if (int((name_mask & dis_mask & dis_mask_min & size_m).sum()) != N):
                    #     print("GT total:", N)
                    #     print("  name_mask true:", int(name_mask.sum()))
                    #     print("  dis_mask true:", int(dis_mask.sum()))
                    #     print("  dis_mask_min true:", int(dis_mask_min.sum()))
                    #     print("  size_m true:", int(size_m.sum()))
                    #     print("  final mask true:", int((name_mask & dis_mask & dis_mask_min & size_m).sum()))

                    mask = name_mask & dis_mask & dis_mask_min & valid2d & size_m
                    new_gt_box9d = gt_box[mask]
                    new_gt_box2d = gt_box2d[mask]
                    gt_frame_id = np.ones((len(new_gt_box9d),)) * k
                else:
                    new_gt_box9d = np.empty((0, 9, 3), dtype=np.float32)
                    new_gt_box2d = np.empty((0, 4), dtype=np.float32)
                    gt_frame_id = np.empty(0, dtype=np.int32)

                # 保存GT数据
                new_anno['gt_box9d'] = new_gt_box9d
                new_anno['gt_box2d'] = new_gt_box2d
                new_anno['gt_frame_id'] = gt_frame_id

                # ---------------------- 核心4：筛选当前类别的预测框 ----------------------
                if len(self.classes) == 1:
                    name_mask2 = np.ones_like(pred_name, dtype=bool)
                else:
                    name_mask2 = pred_name == cls_n
                # 计算预测框距离掩码
                pre_centers = pre_box[:, 0]
                pre_center_dist = np.linalg.norm(pre_centers, axis=-1)
                dis_mask2 = pre_center_dist < cls_max_dis
                dis_mask2_min = pre_center_dist > self.dis_min

                # 生成预测2D框并筛选尺寸
                if len(pre_box) > 0:
                    pred_box2d, size2, valid2d2 = self.box9d_to_2d(
                        pre_box,
                        intrinsic_mat=intrinsic,
                        extrinsic_mat=extrinsic,
                        distortion_matrix=distortion
                    )
                    size_m2 = size2 >= self.min_pixel

                    # N = len(pre_box)
                    # if (int((name_mask2 & dis_mask2 & dis_mask2_min & size_m2).sum()) != N):
                    #     print("Pred total:", N)
                    #     print("  name_mask true:", int(name_mask2.sum()))
                    #     print("  dis_mask true:", int(dis_mask2.sum()))
                    #     print("  dis_mask_min true:", int(dis_mask2_min.sum()))
                    #     print("  size_m true:", int(size_m2.sum()))
                    #     print("  final mask true:", int((name_mask2 & dis_mask2 & dis_mask2_min & size_m2).sum()))

                    mask2 = name_mask2 & dis_mask2 & dis_mask2_min & valid2d2 & size_m2
                    # if mask2.sum() != mask.sum():
                    #     print(mask2.sum(), mask.sum())
                    #     print("  name_mask true:", int(name_mask.sum()))
                    #     print("  dis_mask true:", int(dis_mask.sum()))
                    #     print("  dis_mask_min true:", int(dis_mask_min.sum()))
                    #     print("  valid2d true:", int(valid2d.sum()))
                    #     print("  size_m true:", int(size_m.sum()))
                    #     print("  final mask true:", int((name_mask & dis_mask & dis_mask_min & size_m).sum()))

                    #     print("  name_mask true:", int(name_mask2.sum()))
                    #     print("  dis_mask true:", int(dis_mask2.sum()))
                    #     print("  dis_mask_min true:", int(dis_mask2_min.sum()))
                    #     print("  valid2d2 true:", int(valid2d2.sum()))
                    #     print("  size_m true:", int(size_m2.sum()))
                    #     print("  final mask true:", int((name_mask2 & dis_mask2 & dis_mask2_min & size_m2).sum()))
                    new_pred_box9d = pre_box[mask2]
                    new_pred_score = pre_score[mask2]
                    new_pred_box2d = pred_box2d[mask2]
                    pred_frame_id = np.ones((len(new_pred_box9d),)) * k
                else:
                    new_pred_box9d = np.empty((0, 9, 3), dtype=np.float32)
                    new_pred_score = np.empty(0, dtype=np.float32)
                    new_pred_box2d = np.empty((0, 4), dtype=np.float32)
                    pred_frame_id = np.empty(0, dtype=np.int32)

                # 保存预测数据
                new_anno['pred_box9d'] = new_pred_box9d
                new_anno['pred_score'] = new_pred_score
                new_anno['pred_box2d'] = new_pred_box2d
                new_anno['pred_frame_id'] = pred_frame_id

                new_annos.append(new_anno)

            # ---------------------- 5. 计算各类评估指标（核心：参数名完全匹配函数声明） ----------------------
            # 5.1 计算6DoF姿态精度（参数名与evaluation_6dof_by_name声明完全对应）
            print(f"\n[EVAL] 正在计算类别 {cls_n} 的6DoF姿态精度...")
            acc_6dof_str, orin_acc, posi_acc, size_acc = self.evaluation_6dof_by_name(
                annos=new_annos,  # 对应函数第1个参数annos
                metric_root_path=path_6dof,  # 对应函数第2个参数metric_root_path（修正前为save_path）
                cls_name=cls_n,  # 对应函数第3个参数cls_name
                max_ass_dis=cls_max_dis,  # 对应函数第4个参数max_ass_dis
                dof6_dis_norm_max=dof6_dis_norm_max,  # 对应函数第5个参数
                dof6_ori_norm_max=dof6_ori_norm_max,  # 对应函数第6个参数
                dof6_size_norm_max=dof6_size_norm_max  # 对应函数第7个参数
            )
            print(f"[EVAL] 类别 {cls_n} 6DoF精度计算完成!")

            # 5.2 计算3D AP（假设函数声明参数名与6DoF一致，若不同可替换metric_root_path为实际参数名）
            print(f"[EVAL] 正在计算类别 {cls_n} 的3D AP...")
            ap3d_str, ap3d = self.evaluation_3dAP_by_name(
                annos=new_annos,
                metric_root_path=path_3dap,  # 统一用metric_root_path，与6DoF函数保持一致
                cls_name=cls_n,
                match_threshold=ap3d_dis_threshold,
                recall_num=ap3d_recall_num,
                max_dis=cls_max_dis
            )
            print(f"[EVAL] 类别 {cls_n} 3D AP计算完成!")

            # 5.3 计算2D AP（同上，参数名与3D AP保持一致）
            print(f"[EVAL] 正在计算类别 {cls_n} 的2D AP...")
            ap2d_str, ap2d = self.evaluation_2dAP_by_name(
                annos=new_annos,
                metric_root_path=path_2dap,  # 统一用metric_root_path
                cls_name=cls_n,
                match_threshold=ap2d_iou_thresholds,
                recall_num=ap2d_recall_num,
                max_dis=cls_max_dis
            )
            print(f"[EVAL] 类别 {cls_n} 2D AP计算完成!")

            # 5.4 计算最终ADS指标（保留2位小数）
            final_ads = np.mean([ap3d, ap2d, orin_acc, posi_acc, size_acc])
            final_ads = round(final_ads, 2)

            # 拼接最终结果字符串
            final_str += f"\n{'='*50}\n"
            final_str += f"类别 {cls_n} 评估结果\n"
            final_str += f"{'='*50}\n"
            final_str += acc_6dof_str + "\n"
            final_str += ap3d_str + "\n"
            final_str += ap2d_str + "\n"
            final_str += f"最终 LAA3D_ADS_{cls_n} (%) : {final_ads}\n"
            final_str += f"{'='*50}\n"

        # 打印最终汇总
        print(f"\n{'='*60}")
        print("所有类别评估完成！最终汇总结果：")
        print(f"{'='*60}")
        print(final_str)

        return final_str


