import torch
import numpy as np
from uavdet3d.model.detectors.detector_template import DetectorTemplate
from uavdet3d.utils.object_encoder_laam6d import all_object_encoders
import torch.nn.functional as F


class CenterDetLaam6d(DetectorTemplate):
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

        # 用GT替换预测，验证编码解码是否正确
        # hm = batch_dict['hm']
        # center_res = batch_dict['center_res']
        # center_dis = batch_dict['center_dis']
        # dim = batch_dict['dim']
        # rot = batch_dict['rot']

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
            # data_dict = {}
            # for key in batch_dict:
            #     # 跳过不需要处理的键
            #     if key in {'batch_size', 'down_sample_ratio', 'pred_center_dict'}:
            #         continue
            #
            #     if key in {'intrinsic', 'extrinsic', 'distortion'}:
            #         value = [batch_dict[key][batch_id]]
            #     else:
            #         # 提取当前batch_id的样本
            #         value = batch_dict[key][batch_id]
            #
            #     # 转换为numpy数组
            #     if isinstance(value, torch.Tensor):
            #         # 处理PyTorch张量：分离梯度→移到CPU→转numpy
            #         data_dict[key] = value.detach().cpu().numpy()
            #     else:
            #         try:
            #             # 尝试转换其他类型（列表、标量等）
            #             data_dict[key] = np.array(value)
            #         except:
            #             # 无法转换时保留原始值（或根据需求处理）
            #             data_dict[key] = value
            #             print(f"警告：键'{key}'的值无法转换为NumPy数组，已保留原始类型")
            #
            # pred_boxes_params = pred_boxes9d[:, :-1]  # 移除 cls，得到 (N, 9) 的 9 参数框
            # # 2. 转换为 9 点格式的 3D 框 (N, 9, 3)
            # pred_box9d_9points = self.dataset.convert_9params_to_9points(pred_boxes_params)
            # gt_box9d, pred_box9d = self.dataset.match_gt(data_dict['gt_boxes'], pred_box9d_9points)
            #
            # print(gt_box9d)
            # print(pred_box9d)
            #
            # # 可视化对比：使用转换后的 9 点格式框
            # self.dataset.data_pre_processor._visualize_encode_decode(
            #     data_dict,
            #     gt_box9d,  # 原始 9 点格式框
            #     pred_box9d,  # 转换后的 9 点格式框
            #     save_dir="encode_decode_vis_pred",
            #     save_prefix=f"{data_dict.get('seq_id')}_{data_dict.get('frame_id')}"
            # )

            all_pred_boxes9d.append(pred_boxes9d[confidence > self.score_thresh])
            all_confidence.append(confidence[confidence > self.score_thresh])

        batch_dict['pred_boxes9d'] = all_pred_boxes9d
        batch_dict['confidence'] = all_confidence

        return batch_dict
