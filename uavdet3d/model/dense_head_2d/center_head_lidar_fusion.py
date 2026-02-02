import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def centernet_hm_loss(pred_logits, gt, alpha=2.0, beta=4.0, eps=1e-6):
    """
    pred_logits: (B,C,H,W) or (B,H,W) logits
    gt: same shape, gaussian heatmap in [0,1]
    """
    pred = torch.sigmoid(pred_logits).clamp(eps, 1 - eps)
    pos_thr = 0.9

    pos_mask = (gt > pos_thr).float()
    neg_mask = (gt <= pos_thr).float()

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_weight = torch.pow(1 - gt, beta)
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg_mask

    num_pos = pos_mask.sum().clamp(min=1.0)
    loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


class DepthPredictionBlock(nn.Module):
    """专门的深度预测增强模块，针对LiDAR融合优化（新增抗过拟合机制）"""
    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.1):
        super(DepthPredictionBlock, self).__init__()
        # 深度特征增强网络，保留残差连接核心
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)  # 新增dropout，缓解过拟合（适配多场景数据分布差异）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # 残差连接（确保深度特征不丢失）
        self.shortcut = nn.Conv2d(in_channels, mid_channels, kernel_size=1) if in_channels != mid_channels else nn.Identity()
        # 瓶颈层（优化特征维度，提升融合效率）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels//2, kernel_size=1),
            nn.BatchNorm2d(mid_channels//2),
            nn.ReLU(inplace=True)
        )
        # 输出层（保持无偏置设计，避免训练震荡）
        self.out_conv = nn.Conv2d(mid_channels//2, out_channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out) + residual  # 残差融合，保留LiDAR原始深度特征
        out = self.bottleneck(out)
        out = self.out_conv(out)
        return out


class CenterHeadLidarFusion(nn.Module):
    def __init__(self, model_cfg ):
        super(CenterHeadLidarFusion, self).__init__()

        self.model_cfg = model_cfg
        self.head_config = self.model_cfg.SEPARATE_HEAD_CFG
        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT
        self.in_c = self.model_cfg.INPUT_CHANNELS

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            # 深度预测任务（center_dis）使用增强模块
            if cur_head_name == 'center_dis':
                # 适配无人机场景：dropout_rate=0.1（平衡特征保留与抗过拟合）
                fc = DepthPredictionBlock(self.in_c, conv_mid, out_c, dropout_rate=0.1)
            elif cur_head_name == 'hm':
                # HM独立特征分支：增加通道数和深度，减少梯度竞争
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid * 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(conv_mid * 2, conv_mid, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=True)
                )
                # 初始化偏置以匹配目标分布（mean=0.095）
                nn.init.constant_(fc[-1].bias, -2.25)
            else:
                # 其他任务：新增dropout和weight decay友好设计
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(conv_mid),
                    nn.ReLU(),
                    nn.Dropout2d(0.1),  # 轻微dropout，防止其他头过拟合
                    nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=False)
                )
            self.__setattr__(cur_head_name, fc)

        self.forward_loss_dict = dict()


    def get_loss(self):
        loss = 0
        head_losses = {}
        loss_weight_hm = 10.0 
        loss_weight_dim = 10.0
        loss_weight_dis = 20.0
        loss_weight_rot = 1.0
        loss_weight_res = 1.0

        for cur_name in self.head_keys:
            if cur_name in self.forward_loss_dict['pred_center_dict'] and cur_name in self.forward_loss_dict['gt_center_dict']:
                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]
                current_loss = 0

                pred = pred_h.reshape(-1)
                gt = gt_h.reshape(-1)
                mask_x = torch.abs(gt) > 0
                mask_count = mask_x.sum().item()
                
                if cur_name == 'hm':
                    print("\n ++++++++++++++++++++++++++++++++++++++")
                    # print('hm_gt', gt[mask_x][:])
                    # print('hm_pred', torch.sigmoid(pred)[mask_x][:])
                    l = F.binary_cross_entropy(torch.sigmoid(pred), gt, reduction='none')
                    l_map = l.sum() / (mask_count + 1)
                    weighted_loss = l_map * loss_weight_hm
                    current_loss = weighted_loss.item()
                    loss += weighted_loss
                    # weighted_loss = centernet_hm_loss(pred_h, gt_h) * loss_weight_hm
                    # current_loss = weighted_loss.item()
                    # loss += weighted_loss


                    # 可视化 gt_h 的热图
                    import matplotlib.pyplot as plt
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    gt_img = gt_h[0].detach().cpu().numpy().squeeze()
                    im1 = ax1.imshow(gt_img, cmap='hot', interpolation='nearest')
                    ax1.set_title("Ground Truth Heatmap")
                    fig.colorbar(im1, ax=ax1)
                    pred_img = torch.sigmoid(pred_h)[0].detach().cpu().numpy().squeeze()
                    im2 = ax2.imshow(pred_img, cmap='hot', interpolation='nearest')
                    ax2.set_title("Predicted Heatmap")
                    fig.colorbar(im2, ax=ax2)
                    plt.tight_layout()
                    plt.show()


                if mask_count > 0:
                    if cur_name == 'center_dis':
                        print('center_dis_gt:',gt[mask_x])
                        print('center_dis_pred:',pred[mask_x])
                        mse_loss = F.l1_loss(pred[mask_x], gt[mask_x], reduction='mean')
                        weighted_loss = mse_loss * loss_weight_dis
                        current_loss = weighted_loss.item()
                        loss += weighted_loss
                    if cur_name == 'rot':
                        mse_loss = F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean')
                        weighted_loss = mse_loss * loss_weight_rot
                        current_loss = weighted_loss.item()
                        loss += weighted_loss
                        # diff = gt[mask_x] - pred[mask_x]
                        # abs_diff = torch.abs(diff)
                        # loss_res = abs_diff.mean() * loss_weight_rot
                    if cur_name == 'center_res':
                        mse_loss = F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean')
                        weighted_loss = mse_loss * loss_weight_res
                        current_loss = weighted_loss.item()
                        loss += weighted_loss
                    if cur_name == 'dim':
                        print('dim_gt:',gt[mask_x])
                        print('dim_pred:',(pred[mask_x]))
                        mse_loss = F.l1_loss((pred[mask_x]), gt[mask_x], reduction='mean')
                        weighted_loss = mse_loss * loss_weight_dim
                        current_loss = weighted_loss.item()
                        loss += weighted_loss
                
                # 存储当前头的loss值
                head_losses[cur_name] = current_loss
        
        # 打印所有预测头的loss值，并找出最差的预测
        if head_losses:
            print("\n各预测头loss值:")
            for head_name, head_loss in head_losses.items():
                print(f"{head_name}: {head_loss:.6f}")
            # 找出loss最大的预测头
            worst_head = max(head_losses, key=head_losses.get)
            print(f"最差的预测是: {worst_head}, loss值: {head_losses[worst_head]:.6f}")

        return loss

    def forward(self, batch_dict):
        pred_dict = {}
        gt_dict = {}

        x = batch_dict['features_2d']  # 兼容多模态特征输入（RGB/IR/LiDAR融合后特征）

        for cur_name in self.head_keys:
            head_module = self.__getattr__(cur_name)
            pred = head_module(x)

            if cur_name == 'hm' and not self.training:
                pred = torch.sigmoid(pred)
                gt = batch_dict[cur_name]

                # import matplotlib.pyplot as plt
                # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                # gt_img = gt[0].detach().cpu().numpy().squeeze()
                # im1 = ax1.imshow(gt_img, cmap='hot', interpolation='nearest')
                # ax1.set_title("Ground Truth Heatmap")
                # fig.colorbar(im1, ax=ax1)
                # pred_img = pred[0].detach().cpu().numpy().squeeze()
                # im2 = ax2.imshow(pred_img, cmap='hot', interpolation='nearest')
                # ax2.set_title("Predicted Heatmap")
                # fig.colorbar(im2, ax=ax2)
                # plt.tight_layout()
                # plt.show()
                
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