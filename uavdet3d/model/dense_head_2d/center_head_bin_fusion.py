import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthPredictionBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Conv2d(in_channels, mid_channels, kernel_size=1) if in_channels != mid_channels else nn.Identity()
        half = max(1, mid_channels // 2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_channels, half, kernel_size=1),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(half, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out) + res
        out = self.bottleneck(out)
        out = self.out_conv(out)
        return out


def decode_center_dis_from_bins(bin_logits, res_map, w_norm: float):
    """
    bin_logits: (B,N,H,W)
    res_map:    (B,N,H,W) residual in [-inf, inf], we clamp to [-1,1]
    return:
      z_hat: (B,1,H,W) in [0,1] (approx)
    """
    B, N, H, W = bin_logits.shape
    b_hat = torch.argmax(bin_logits, dim=1)  # (B,H,W)

    # gather residual of predicted bin
    idx = b_hat.unsqueeze(1)  # (B,1,H,W)
    r_hat = torch.gather(res_map, dim=1, index=idx).squeeze(1)  # (B,H,W)
    r_hat = r_hat.clamp(-1.0, 1.0)

    z_hat = ((b_hat.to(bin_logits.dtype) + 0.5) * w_norm) + r_hat * (0.5 * w_norm)
    z_hat = z_hat.clamp(0.0, 1.0)
    return z_hat.unsqueeze(1)  # (B,1,H,W)


class CenterHeadLidarFusion(nn.Module):
    """
    center_dis 改为：bin分类(30类, 5m/bin) + class-conditioned residual
    - 输入/GT/输出仍然是 0~1 归一化距离
    - 下游继续使用 pred_center_dict['center_dis'] (B,1,H,W) 连续值
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT
        self.in_c = self.model_cfg.INPUT_CHANNELS

        # ---- distance bin config (normalized) ----
        self.D_MAX = float(getattr(self.model_cfg, "D_MAX", 150.0))
        self.BIN_W_M = float(getattr(self.model_cfg, "BIN_W_M", 5.0))
        self.bin_w_norm = self.BIN_W_M / self.D_MAX  # 5/150
        self.num_bins = int((self.D_MAX + self.BIN_W_M - 1e-6) // self.BIN_W_M)  # 30

        # ---- lidar loss params (keep your old ones if you use them) ----
        self.lidar_w = float(getattr(self.model_cfg, "LIDAR_LOSS_WEIGHT", 0.0))  # default off here
        self.lidar_K = int(getattr(self.model_cfg, "LIDAR_MIN_COUNT", 5))
        self.lidar_max_spread = float(getattr(self.model_cfg, "LIDAR_MAX_SPREAD", 3.0 / 150.0))
        self.lidar_max_delta_to_gt = float(getattr(self.model_cfg, "LIDAR_MAX_DELTA_TO_GT", 10.0 / 150.0))

        # ---- build heads ----
        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            if cur_head_name == 'center_dis':
                # ✅ replace: bin logits + residual map (N channels)
                self.center_bin_head = DepthPredictionBlock(self.in_c, conv_mid, self.num_bins, dropout_rate=0.1)
                self.center_resbin_head = DepthPredictionBlock(self.in_c, conv_mid, self.num_bins, dropout_rate=0.1)

                # keep an attribute named 'center_dis' for compatibility (not used as module)
                setattr(self, 'center_dis', nn.Identity())

            elif cur_head_name == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid * 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_mid * 2, conv_mid, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_mid, out_c, kernel_size=3, padding=1, bias=True),
                )
                nn.init.constant_(fc[-1].bias, -2.25)
                setattr(self, cur_head_name, fc)

            else:
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid, kernel_size=3, padding=1),
                    nn.BatchNorm2d(conv_mid),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1),
                    nn.Conv2d(conv_mid, out_c, kernel_size=3, padding=1, bias=False),
                )
                setattr(self, cur_head_name, fc)

        self.forward_loss_dict = {}

        # loss weights (you can tune)
        self.loss_weight_hm = 10.0
        self.loss_weight_dim = 10.0
        self.loss_weight_rot = 1.0
        self.loss_weight_res = 1.0
        self.loss_weight_bincls = 1.0      # ✅ bin CE
        self.loss_weight_binres = 1.0      # ✅ residual SmoothL1

    def get_loss(self):
        loss = 0.0

        pred_dict = self.forward_loss_dict.get('pred_center_dict', {})
        gt_dict = self.forward_loss_dict.get('gt_center_dict', {})

        for cur_name in self.head_keys:
            if cur_name not in pred_dict or cur_name not in gt_dict:
                continue

            pred_h = pred_dict[cur_name]
            gt_h = gt_dict[cur_name]

            # ---- hm ----
            if cur_name == 'hm':
                pred = pred_h.reshape(-1)
                gt = gt_h.reshape(-1)
                l = F.binary_cross_entropy(torch.sigmoid(pred), gt, reduction='mean')
                loss = loss + l * self.loss_weight_hm
                continue

            # ---- center_dis (bin+res) ----
            if cur_name == 'center_dis':
                # retrieve logits/res maps stored in forward
                bin_logits = pred_dict['center_bin_logits']   # (B,N,H,W)
                res_map = pred_dict['center_resbin_map']      # (B,N,H,W)

                # mask: only GT center points
                gt_z = gt_h[:, 0]  # (B,H,W)
                mask = gt_z > 0

                if mask.sum() == 0:
                    continue

                # GT bin + residual (all in normalized 0~1)
                w = self.bin_w_norm
                b_gt = torch.floor(gt_z / w).to(torch.long).clamp(0, self.num_bins - 1)  # (B,H,W)
                center_norm = (b_gt.to(gt_z.dtype) + 0.5) * w
                r_gt = (gt_z - center_norm) / (0.5 * w)
                r_gt = r_gt.clamp(-1.0, 1.0)

                # CE on bins
                # reshape to (num_valid, N)
                logits_flat = bin_logits.permute(0, 2, 3, 1)[mask]  # (K,N)
                b_flat = b_gt[mask]                                 # (K,)
                l_cls = F.cross_entropy(logits_flat, b_flat)

                # residual gather at GT bin
                # gather -> (B,1,H,W) then mask
                idx = b_gt.unsqueeze(1)  # (B,1,H,W)
                r_pred = torch.gather(res_map, dim=1, index=idx).squeeze(1)  # (B,H,W)
                l_res = F.smooth_l1_loss(r_pred[mask], r_gt[mask], reduction='mean')

                loss = loss + l_cls * self.loss_weight_bincls + l_res * self.loss_weight_binres

                # (optional) LiDAR constraint on decoded z
                if self.lidar_w > 0 and ('lidar_depth_ds' in self.forward_loss_dict):
                    # you can plug your old lidar constraint here if you want,
                    # using decoded z_hat (B,1,H,W) and the same mask logic.
                    pass

                continue

            # ---- other regression heads (keep your old masking style if GT is sparse) ----
            pred = pred_h.reshape(-1)
            gt = gt_h.reshape(-1)
            mask_x = torch.abs(gt) > 0
            if mask_x.sum() == 0:
                continue

            if cur_name == 'dim':
                loss = loss + F.l1_loss(pred[mask_x], gt[mask_x], reduction='mean') * self.loss_weight_dim
            elif cur_name == 'rot':
                loss = loss + F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean') * self.loss_weight_rot
            elif cur_name == 'center_res':
                loss = loss + F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean') * self.loss_weight_res

        return loss

    def forward(self, batch_dict):
        x = batch_dict['features_2d']
        pred_dict, gt_dict = {}, {}

        for cur_name in self.head_keys:
            if cur_name == 'center_dis':
                # bin + residual
                bin_logits = self.center_bin_head(x)     # (B,N,H,W)
                res_map = self.center_resbin_head(x)     # (B,N,H,W)

                # decode to continuous distance (B,1,H,W) for downstream compatibility
                center_dis = decode_center_dis_from_bins(bin_logits, res_map, self.bin_w_norm)

                pred_dict['center_bin_logits'] = bin_logits
                pred_dict['center_resbin_map'] = res_map
                pred_dict['center_dis'] = center_dis

            else:
                head = getattr(self, cur_name)
                pred = head(x)
                if cur_name == 'hm' and not self.training:
                    pred = torch.sigmoid(pred)
                pred_dict[cur_name] = pred

            if self.training and cur_name in batch_dict:
                gt_dict[cur_name] = batch_dict[cur_name]

        batch_dict['pred_center_dict'] = pred_dict
        self.forward_loss_dict['pred_center_dict'] = pred_dict

        if self.training:
            batch_dict['gt_dict'] = gt_dict
            self.forward_loss_dict['gt_center_dict'] = gt_dict

        return batch_dict
