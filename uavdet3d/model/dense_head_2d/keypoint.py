import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2

class KeyPoint(nn.Module):
    def __init__(self, model_cfg ):
        super(KeyPoint, self).__init__()

        self.model_cfg = model_cfg

        in_channels, obj_num, out_channels, radius = self.model_cfg.INPUT_CHANNELS,self.model_cfg.OBJ_NUM,self.model_cfg.OUTPUT_CHANNELS,self.model_cfg.RADIUS,

        self.conv_block = nn.Conv2d(in_channels, obj_num*out_channels, kernel_size=1)

        self.conv_block_res_x = nn.Conv2d(in_channels, obj_num * out_channels, kernel_size=1)

        self.conv_block_res_y = nn.Conv2d(in_channels, obj_num * out_channels, kernel_size=1)

        self.obj_num = obj_num
        self.out_channels = out_channels

        self.forward_loss_dict=dict()

    def get_loss(self):

        pred_heatmap = self.forward_loss_dict['pred_heatmap'] # B,K,N,4,W,H
        gt_heatmap = self.forward_loss_dict['gt_heatmap'] # B,K,N,4,W,H

        pred = pred_heatmap.reshape(-1)
        gt = gt_heatmap.reshape(-1)

        l = F.binary_cross_entropy(pred, gt, reduction='none')
        mask = gt>0
        l_map = l.sum()/(mask.sum()+1)

        pred_res_x = self.forward_loss_dict['pred_res_x']
        gt_res_x = self.forward_loss_dict['gt_res_x']

        mask_x = torch.abs(gt_res_x)>0

        loss_res_x = torch.abs(gt_res_x[mask_x]-pred_res_x[mask_x]).mean()

        pred_res_y = self.forward_loss_dict['pred_res_y']
        gt_res_y = self.forward_loss_dict['gt_res_y']

        mask_y = torch.abs(gt_res_y)>0

        loss_res_y = torch.abs(gt_res_y[mask_y]-pred_res_y[mask_y]).mean()
        #
        # print('lmap ', l_map)
        # print('lresx ', loss_res_x)
        # print('lresy ', loss_res_y)

        return l_map+loss_res_y+loss_res_x


    def forward(self, batch_dict):

        x = batch_dict['features_2d']

        B, K, C, W, H = x.shape

        x = x.reshape(B*K, C, W, H)

        y = self.conv_block(x) # BK, O4, W, H

        res_x = self.conv_block_res_x(x) # BK, O4, W, H
        res_y = self.conv_block_res_y(x) # BK, O4, W, H

        pred_heat_map = y.reshape(B, K, self.obj_num,self.out_channels, W, H)

        pred_res_x = res_x.reshape(B, K, self.obj_num, self.out_channels, W, H)
        pred_res_y = res_y.reshape(B, K, self.obj_num, self.out_channels, W, H)

        pred_heat_map = torch.sigmoid(pred_heat_map)

        batch_dict['pred_heatmap'] = pred_heat_map
        batch_dict['pred_res_x'] = pred_res_x
        batch_dict['pred_res_y'] = pred_res_y
        # cv2.imwrite('center.png', pred_heat_map[0,0,0,0:3,:,:].cpu().detach().numpy().transpose(1,2,0)*255)
        # input()

        self.forward_loss_dict['pred_heatmap'] = batch_dict['pred_heatmap']


        self.forward_loss_dict['pred_res_x'] = batch_dict['pred_res_x']


        self.forward_loss_dict['pred_res_y'] = batch_dict['pred_res_y']


        if self.training:
            self.forward_loss_dict['gt_heatmap'] = batch_dict['gt_heatmap']
            self.forward_loss_dict['gt_res_x'] = batch_dict['gt_res_x']
            self.forward_loss_dict['gt_res_y'] = batch_dict['gt_res_y']

        return batch_dict
