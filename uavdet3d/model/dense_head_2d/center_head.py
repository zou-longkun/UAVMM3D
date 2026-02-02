import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2


class CenterHead(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterHead, self).__init__()

        self.model_cfg = model_cfg

        self.head_config = self.model_cfg.SEPARATE_HEAD_CFG

        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER

        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT

        self.in_c = self.model_cfg.INPUT_CHANNELS

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            fc = nn.Sequential(
                nn.Conv2d(self.in_c, conv_mid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(conv_mid),
                nn.ReLU(),
                nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.__setattr__(cur_head_name, fc)

        self.forward_loss_dict=dict()

    def get_loss(self):
        loss = 0
        for cur_name in self.head_keys:

            if cur_name == 'hm':

                pred_heatmap = self.forward_loss_dict['pred_center_dict']['hm']
                gt_heatmap = self.forward_loss_dict['gt_center_dict']['hm']

                pred = pred_heatmap.reshape(-1)
                gt = gt_heatmap.reshape(-1)

                l = F.binary_cross_entropy(torch.sigmoid(pred), gt, reduction='none')
                mask = gt > 0
                l_map = l.sum() / (mask.sum() + 1)
                # print(cur_name, l_map)
                loss+=l_map
            else:
                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]

                pred_h = pred_h.reshape(-1)
                gt_h = gt_h.reshape(-1)

                mask_x = torch.abs(gt_h) > 0

                loss_res = torch.abs(gt_h[mask_x] - pred_h[mask_x]).mean()
                # print(cur_name, loss_res)
                loss+=loss_res

        return loss


    def forward(self, batch_dict):

        pred_dict = {}

        gt_dict = {}

        x = batch_dict['features_2d']

        B, K, C, W, H = x.shape

        x = x.reshape(B*K, C, W, H)

        for cur_name in self.head_keys:

            pred_dict[cur_name] = self.__getattr__(cur_name)(x)

            if self.training:
                gt_dict[cur_name] = batch_dict[cur_name]

        batch_dict['pred_center_dict'] = pred_dict

        self.forward_loss_dict['pred_center_dict'] = pred_dict

        if self.training:
            batch_dict['gt_dict'] = gt_dict
            self.forward_loss_dict['gt_center_dict'] = gt_dict

        return batch_dict
