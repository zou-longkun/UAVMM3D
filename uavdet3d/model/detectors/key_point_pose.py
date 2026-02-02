import torch
import numpy as np
from uavdet3d.model.detectors.detector_template import DetectorTemplate
from uavdet3d.utils.object_encoder import all_object_encoders

class KeyPoint2Pose(DetectorTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg
        self.dataset = dataset

        self.key_point_decoder = all_object_encoders[self.model_cfg.POST_PROCESSING.DECONDER]

        self.encode_corner = np.array(self.model_cfg.POST_PROCESSING.DECODER_CORNER)
        self.off_set = np.array(self.model_cfg.POST_PROCESSING.OFFSET)

        self.pnp_algo = self.model_cfg.POST_PROCESSING.PnP_ALGO

    def forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:

            loss = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }

            return ret_dict
        else:
            return self.post_processing(batch_dict)

    def get_training_loss(self):

        loss = self.dense_head_2d.get_loss()

        return loss

    def post_processing(self, batch_dict):

        batch_size = batch_dict['batch_size']

        im_num = self.dataset.im_num
        obj_num = self.dataset.obj_num

        all_pred_boxes9d = []
        all_key_points_2d = []
        all_confidence = []

        for batch_id in range(batch_size):
            pred_heat_map = batch_dict['pred_heatmap'][batch_id]  # 1, 1, 4, W, H
            pred_res_x = batch_dict['pred_res_x'][batch_id]  # 1, 1, 4, W, H
            pred_res_y = batch_dict['pred_res_y'][batch_id]  # 1, 1, 4, W, H

            intrinsic = batch_dict['intrinsic'][batch_id]  # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id]  # 4, 4
            distortion = batch_dict['distortion'][batch_id]  # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id]  # 2,
            new_im_size = batch_dict['new_im_size'][batch_id]  # 2,
            obj_size = batch_dict['obj_size'][batch_id]  # 3,

            stride = batch_dict['stride'][batch_id]

            key_points_2d, confidence, pred_boxes9d = self.key_point_decoder(self.encode_corner,
                                                                             self.off_set,
                                                                            pred_heat_map,
                                                                            pred_res_x,
                                                                            pred_res_y,
                                                                            new_im_size[0],
                                                                            new_im_size[1],
                                                                            raw_im_size[0],
                                                                            raw_im_size[1],
                                                                            stride,
                                                                            im_num,
                                                                            obj_num,
                                                                            obj_size,
                                                                            intrinsic,
                                                                            distortion,
                                                                            self.pnp_algo)
            all_pred_boxes9d.append(pred_boxes9d)
            all_key_points_2d.append(key_points_2d)
            all_confidence.append(confidence)

        batch_dict['pred_boxes9d'] = np.array(all_pred_boxes9d)
        batch_dict['key_points_2d'] = np.array(all_key_points_2d)
        batch_dict['confidence'] = np.array(all_confidence)

        return batch_dict