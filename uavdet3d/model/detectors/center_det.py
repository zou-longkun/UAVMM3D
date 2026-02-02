import torch
import numpy as np
from uavdet3d.model.detectors.detector_template import DetectorTemplate
from uavdet3d.utils.object_encoder import all_object_encoders
import torch.nn.functional as F


class CenterDet(DetectorTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.center_decoder = all_object_encoders[self.model_cfg.POST_PROCESSING.DECONDER]

        self.max_num = self.model_cfg.POST_PROCESSING.MAX_OBJ
        self.score_thresh = self.model_cfg.POST_PROCESSING.SCORE_THRESH

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

        all_pred_boxes9d = []

        all_confidence = []

        hm = batch_dict['pred_center_dict']['hm']
        center_res = batch_dict['pred_center_dict']['center_res']
        center_dis = batch_dict['pred_center_dict']['center_dis']
        dim = batch_dict['pred_center_dict']['dim']
        rot = batch_dict['pred_center_dict']['rot']

        def reshape_t(tensor, batch_size):
            BK, C, W, H = tensor.shape
            return tensor.reshape(batch_size, -1, C, W, H)

        hm = reshape_t(hm, batch_size)
        center_res = reshape_t(center_res, batch_size)
        center_dis = reshape_t(center_dis, batch_size)
        dim = reshape_t(dim, batch_size)
        rot = reshape_t(rot, batch_size)

        for batch_id in range(batch_size):
            intrinsic = batch_dict['intrinsic'][batch_id]  # 3, 3
            extrinsic = batch_dict['extrinsic'][batch_id]  # 4, 4
            distortion = batch_dict['distortion'][batch_id]  # 5,
            raw_im_size = batch_dict['raw_im_size'][batch_id]  # 2,
            new_im_size = batch_dict['new_im_size'][batch_id]  # 2,
            # obj_size = batch_dict['obj_size'][batch_id]  # 3,
            print(intrinsic)
            print(extrinsic)
            print(distortion)
            stride = batch_dict['stride'][batch_id]

            this_hm = hm[batch_id]
            this_hm = torch.sigmoid(this_hm)
            this_center_res = center_res[batch_id]
            # this_center_dis = torch.exp(center_dis[batch_id])*self.dataset.dataset_cfg.MAX_DIS
            # this_dim = torch.exp(dim[batch_id]) #*self.dataset.dataset_cfg.MAX_SIZE

            this_center_dis = center_dis[batch_id] * self.dataset.dataset_cfg.MAX_DIS
            this_dim = dim[batch_id] * self.dataset.dataset_cfg.MAX_SIZE
            this_rot = rot[batch_id]

            pred_boxes9d, confidence = self.center_decoder(this_hm,
                                                           this_center_res,
                                                           this_center_dis,
                                                           this_dim,
                                                           this_rot,
                                                           intrinsic,
                                                           extrinsic,
                                                           distortion,
                                                           new_im_size[0],
                                                           new_im_size[1],
                                                           raw_im_size[0],
                                                           raw_im_size[1],
                                                           stride,
                                                           im_num,
                                                           self.max_num)
            # pred_boxes9d[:,0:2]+=0.2

            all_pred_boxes9d.append(pred_boxes9d[confidence > self.score_thresh])
            all_confidence.append(confidence[confidence > self.score_thresh])

        batch_dict['pred_boxes9d'] = all_pred_boxes9d
        batch_dict['confidence'] = all_confidence

        return batch_dict
