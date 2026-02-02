import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class CenterHeadLaam6d(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterHeadLaam6d, self).__init__()

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

        self.forward_loss_dict = dict()

    def get_loss(self):
        loss = 0
        total_valid_loss_terms = 0
        
        for cur_name in self.head_keys:
            if cur_name in self.forward_loss_dict['pred_center_dict'] and cur_name in self.forward_loss_dict['gt_center_dict']:
                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]

                if cur_name == 'hm':
                    pred = pred_h.reshape(-1)
                    gt = gt_h.reshape(-1)
                    # 使用带权重的二元交叉熵损失
                    l = F.binary_cross_entropy_with_logits(pred, gt, pos_weight=torch.tensor([10.0], device=pred.device), reduction='none')
                    mask = gt > 0
                    mask_sum = mask.sum().item()
                    
                    if mask_sum > 0:
                        # 有正样本时，正常计算损失
                        l_map = l.sum() / mask_sum
                        loss += l_map
                        total_valid_loss_terms += 1
                    else:
                        # 没有正样本时，添加一个小的正则化损失，确保梯度能传播
                        # 对于hm头，我们希望负样本区域的预测值接近0
                        neg_loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='mean') * 0.01
                        loss += neg_loss
                        total_valid_loss_terms += 0.01  # 给负样本损失较低的权重
                else:
                    pred = pred_h.reshape(-1)
                    gt = gt_h.reshape(-1)
                    mask_x = torch.abs(gt) > 0
                    mask_count = mask_x.sum().item()
                    
                    if mask_count > 0:
                        # 有有效GT时，正常计算L1损失
                        loss_res = torch.abs(gt[mask_x] - pred[mask_x]).mean()
                        loss += loss_res
                        total_valid_loss_terms += 1
                    else:
                        # 没有有效GT时，添加一个小的正则化损失，确保梯度能传播
                        # 对于回归头，我们希望预测值接近0（默认值）
                        reg_loss = torch.abs(pred).mean() * 0.001
                        loss += reg_loss
                        total_valid_loss_terms += 0.001  # 给正则化损失较低的权重
        
        # 如果所有损失项都无效，返回一个非常小的可微损失
        if total_valid_loss_terms == 0:
            return torch.tensor(1e-6, requires_grad=True, device=next(self.parameters()).device)
        
        return loss

    def forward(self, batch_dict):
        pred_dict = {}
        gt_dict = {}

        x = batch_dict['features_2d']

        for cur_name in self.head_keys:
            head_module = self.__getattr__(cur_name)
            pred = head_module(x)
            pred_dict[cur_name] = pred

            if self.training:
                if cur_name in batch_dict:
                    gt = batch_dict[cur_name]
                    gt_dict[cur_name] = gt

        batch_dict['pred_center_dict'] = pred_dict
        self.forward_loss_dict['pred_center_dict'] = pred_dict

        if self.training:
            batch_dict['gt_dict'] = gt_dict
            self.forward_loss_dict['gt_center_dict'] = gt_dict

        return batch_dict