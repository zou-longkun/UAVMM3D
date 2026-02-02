import numpy as np
from uavdet3d.utils.object_encoder import all_object_encoders
from uavdet3d.utils.centernet_utils import draw_gaussian_to_heatmap, draw_res_to_heatmap
import torch
import cv2
from functools import partial
import copy


class DataPreProcessor():
    def __init__(self, dataset_cfg, training):

        self.dataset_cfg, self.training = dataset_cfg, training

        processor_configs = self.dataset_cfg.DATA_PRE_PROCESSOR

        self.data_processor_queue = []

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def convert_box9d_to_heatmap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_heatmap, config=config)

        obj_num = self.dataset_cfg.OBJ_NUM

        im_num = self.dataset_cfg.IM_NUM

        center_rad = self.dataset_cfg.CENTER_RAD

        offset = config.OFFSET
        key_point_encoder = all_object_encoders[config.ENCODER]
        encode_corner = np.array(config.ENCODER_CORNER)

        intrinsic = data_dict['intrinsic']  # 1,3,3
        extrinsic = data_dict['extrinsic']
        distortion = data_dict['distortion']

        if 'gt_box9d' not in data_dict:
            return data_dict

        gt_box9d = data_dict['gt_box9d']  # N,9

        raw_im_width, raw_im_hight = data_dict['raw_im_size'][0], data_dict['raw_im_size'][1]

        new_im_width, new_im_hight = data_dict['new_im_size'][0], data_dict['new_im_size'][1]

        stride = data_dict['stride']

        # N, 4, 2

        gt_heat_map = []

        res_x = []
        res_y = []

        all_pts2d = []

        for im_id in range(im_num):
            all_ob_heat_map = []

            all_ob_res_x = []
            all_ob_res_y = []

            pts3d, pts2d = key_point_encoder(gt_box9d,
                                             encode_corner=encode_corner,
                                             intrinsic_mat=intrinsic[im_id],
                                             extrinsic_mat=extrinsic[im_id],
                                             distortion_matrix=distortion[im_id],
                                             offset=offset)

            key_pts_num = pts2d.shape[1]

            for ob_id in range(obj_num):
                ob_pts = pts2d[ob_id]
                this_heat_map = torch.zeros(key_pts_num, new_im_hight // stride, new_im_width // stride)
                this_residual_x = np.zeros(shape=(key_pts_num, new_im_hight // stride, new_im_width // stride))
                this_residual_y = np.zeros(shape=(key_pts_num, new_im_hight // stride, new_im_width // stride))

                for k_i, center in enumerate(ob_pts):
                    center[0] *= (new_im_width / raw_im_width / stride)
                    center[1] *= (new_im_hight / raw_im_hight / stride)

                    this_heat_map[k_i] = draw_gaussian_to_heatmap(this_heat_map[k_i], center, center_rad)
                    this_residual_x[k_i], this_residual_y[k_i] = draw_res_to_heatmap(this_residual_x[k_i],
                                                                                     this_residual_y[k_i], center)

                all_ob_heat_map.append(this_heat_map)

                all_ob_res_x.append(this_residual_x)
                all_ob_res_y.append(this_residual_y)

            all_ob_heat_map = torch.stack(all_ob_heat_map)
            all_ob_res_x = np.stack(all_ob_res_x)
            all_ob_res_y = np.stack(all_ob_res_y)

            gt_heat_map.append(all_ob_heat_map.cpu().numpy())
            res_x.append(all_ob_res_x)
            res_y.append(all_ob_res_y)
            all_pts2d.append(pts2d)

        gt_heatmap = np.array(gt_heat_map)  # 1,1,4,W,H
        gt_res_x = np.array(res_x)  # 1,1,4,W,H
        gt_res_y = np.array(res_y)  # 1,1,4,W,H
        all_pts2d = np.array(all_pts2d)  # 1,1,4,2
        # print(gt_res_x.max())
        # #
        # cv2.imwrite('im.png', data_dict['image'][0].transpose(1,2,0)*255)
        # cv2.imwrite('center.png', gt_res_x[0,0,0:3,:,:].transpose(1,2,0)*255)
        # input()

        data_dict['gt_heatmap'] = gt_heatmap
        data_dict['gt_res_x'] = gt_res_x
        data_dict['gt_res_y'] = gt_res_y
        data_dict['gt_pts2d'] = all_pts2d.reshape(im_num * obj_num, -1, 2)

        return data_dict

    def map_name_to_index(self, gt_name, class_name_config):

        return np.array([[class_name_config.index(x) for x in gt_name]])

    def convert_box9d_to_centermap(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.convert_box9d_to_centermap, config=config)

        center_rad = self.dataset_cfg.CENTER_RAD
        class_name_config = self.dataset_cfg.CLASS_NAMES
        im_num = self.dataset_cfg.IM_NUM

        offset = config.OFFSET
        center_point_encoder = all_object_encoders[config.ENCODER]

        if 'gt_box9d' not in data_dict:
            return data_dict

        gt_box9d = data_dict['gt_box9d']  # N,9
        intrinsic = data_dict['intrinsic']  # 1, 3,3
        extrinsic = data_dict['extrinsic']  # 1, 4,4
        distortion = data_dict['distortion']  # ,1 5
        stride = data_dict['stride']

        gt_name = data_dict['gt_name']  # 1,

        cls_index = self.map_name_to_index(gt_name, class_name_config).reshape(-1, 1)  # N,1

        gt_box9d_with_cls = np.concatenate([gt_box9d, cls_index], -1)  # N, 10

        raw_im_size = data_dict['raw_im_size']  # 2,
        new_im_size = data_dict['new_im_size']  # 2,

        gt_hm, gt_center_res, gt_center_dis, gt_dim, gt_rot = center_point_encoder(gt_box9d_with_cls,
                                                                                   intrinsic,
                                                                                   extrinsic,
                                                                                   distortion,
                                                                                   new_im_size[0],
                                                                                   new_im_size[1],
                                                                                   raw_im_size[0],
                                                                                   raw_im_size[1],
                                                                                   stride,
                                                                                   im_num,
                                                                                   class_name_config,
                                                                                   center_rad)
        # # 1, Class, W,H
        # # 1, 2, W,H
        # # 1, 1, W,H
        # 'hm': {'out_channels': 1},
        # 'center_res': {'out_channels': 2},
        # 'center_dis': {'out_channels': 1},
        # 'dim': {'out_channels': 3},
        # 'rot': {'out_channels': 6},

        data_dict['hm'] = gt_hm
        data_dict['center_res'] = gt_center_res

        data_dict['center_dis'] = gt_center_dis / self.dataset_cfg.MAX_DIS

        data_dict['dim'] = gt_dim / self.dataset_cfg.MAX_SIZE
        data_dict['rot'] = gt_rot

        # print(gt_hm[gt_hm>0])
        # print(gt_center_res[gt_center_res>0])
        # print(gt_center_dis[gt_center_dis>0])
        # print(gt_dim[gt_dim>0])
        # print(gt_rot[gt_rot>0])
        # input()

        return data_dict

    def image_normalization(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalization, config=config)

        data_dict['image'] /= 255

        return data_dict

    def filter_box_outside(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_box_outside, config=config)

        gt_box9d = data_dict['gt_box9d']  # N,9

        raw_im_width, raw_im_hight = data_dict['raw_im_size'][0], data_dict['raw_im_size'][1]

        intrinsic = data_dict['intrinsic']  # 1,3,3

        im_num = self.dataset_cfg.IM_NUM

        extrinsic = data_dict['extrinsic']
        distortion = data_dict['distortion']
        gt_name = data_dict['gt_name']
        gt_diff = data_dict['gt_diff']

        valid_gt = []
        valid_name = []
        valid_gt_diff = []

        for i in range(im_num):

            if len(gt_box9d) > 0:
                gt_pts = copy.deepcopy(gt_box9d[:, 0:3])

                corners_2d, _ = cv2.projectPoints(gt_pts, extrinsic[i, :3, :3], extrinsic[i, :3, 3], intrinsic[i],
                                                  distortion[i])
                corners_2d = corners_2d.reshape(-1, 2).astype(int)
                mask_x_low = corners_2d[:, 0] > 0
                mask_x_high = corners_2d[:, 0] < raw_im_width
                mask_y_low = corners_2d[:, 1] > 0
                mask_y_high = corners_2d[:, 1] < raw_im_hight

                mask = mask_x_low * mask_x_high * mask_y_low * mask_y_high

                valid_gt.append(gt_box9d[mask])
                valid_name.append(gt_name[mask])
                valid_gt_diff.append(gt_diff[mask])

        if len(valid_gt) > 0:
            valid_gt = np.concatenate(valid_gt)
            valid_name = np.concatenate(valid_name)
            valid_gt_diff = np.concatenate(valid_gt_diff)
        else:
            valid_gt = np.empty(shape=(0, 9))
            valid_name = np.empty(shape=(0))
            valid_gt_diff = np.empty(shape=(0))

        data_dict['gt_box9d'] = valid_gt
        data_dict['gt_name'] = valid_name
        data_dict['gt_diff'] = valid_gt_diff

        return data_dict

    def __call__(self, data_dict):

        for func in self.data_processor_queue:
            data_dict = func(data_dict)

        return data_dict
