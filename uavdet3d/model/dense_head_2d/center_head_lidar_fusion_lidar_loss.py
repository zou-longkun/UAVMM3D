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


# ----------------------------
# LiDAR sparse constraint utils
# ----------------------------

def masked_median(x: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    """
    x: (..., N)
    mask: same shape bool, True for valid
    returns:
      median over valid elements (invalid -> 0),
      valid_count
    """
    big = torch.finfo(x.dtype).max
    x2 = x.masked_fill(~mask, big)
    x_sorted, _ = torch.sort(x2, dim=dim)

    valid_count = mask.sum(dim=dim)  # (...,)

    mid_idx = torch.clamp((valid_count - 1) // 2, min=0)
    med = torch.gather(x_sorted, dim, mid_idx.unsqueeze(dim)).squeeze(dim)

    med = torch.where(valid_count > 0, med, torch.zeros_like(med))
    return med, valid_count


def robust_downsample_depth(depth_map: torch.Tensor,
                            down_factor: int = 4,
                            invalid_val: float = 0.0):
    """
    Downsample sparse depth map (B,1,H,W) -> (B,1,H//f,W//f)
    by taking median over valid depths in each fxf patch.

    Returns:
      depth_ds:  (B,1,Hf,Wf) median depth per patch (0 if no valid)
      count_ds:  (B,1,Hf,Wf) number of valid pixels per patch
      spread_ds: (B,1,Hf,Wf) (p90 - p10) over valid pixels (0 if insufficient)
    """
    assert depth_map.dim() == 4 and depth_map.size(1) == 1, "depth_map must be (B,1,H,W)"
    B, _, H, W = depth_map.shape
    f = down_factor
    Hf, Wf = H // f, W // f
    if Hf <= 0 or Wf <= 0:
        raise ValueError(f"Input depth size too small for down_factor={f}: H={H}, W={W}")

    depth = depth_map[:, :, :Hf * f, :Wf * f]

    # (B, f*f, Hf*Wf)
    patches = F.unfold(depth, kernel_size=f, stride=f)  # (B, f*f, Hf*Wf)
    patches = patches.transpose(1, 2)                  # (B, Hf*Wf, f*f)

    valid = (patches != invalid_val)

    med, cnt = masked_median(patches, valid, dim=-1)  # (B, Hf*Wf)

    # robust spread: p90 - p10 over valid pixels
    big = torch.finfo(patches.dtype).max
    p2 = patches.masked_fill(~valid, big)
    p_sorted, _ = torch.sort(p2, dim=-1)  # (B, Hf*Wf, f*f)

    k = cnt  # (B, Hf*Wf)
    idx10 = torch.clamp((0.1 * (k - 1).to(torch.float32)).floor().to(torch.long), min=0)
    idx90 = torch.clamp((0.9 * (k - 1).to(torch.float32)).floor().to(torch.long), min=0)

    p10 = torch.gather(p_sorted, -1, idx10.view(B, -1, 1)).squeeze(-1)
    p90 = torch.gather(p_sorted, -1, idx90.view(B, -1, 1)).squeeze(-1)
    spread = p90 - p10
    spread = torch.where(k > 0, spread, torch.zeros_like(spread))

    depth_ds = med.view(B, 1, Hf, Wf)
    count_ds = cnt.view(B, 1, Hf, Wf).to(depth_map.dtype)
    spread_ds = spread.view(B, 1, Hf, Wf)
    return depth_ds, count_ds, spread_ds


def compute_lidar_depth_constraint(
    pred_center_dis: torch.Tensor,   # (B,1,Hf,Wf) or (B,Hf,Wf)
    gt_center_dis: torch.Tensor,     # same shape
    lidar_depth_ds: torch.Tensor,    # (B,1,Hf,Wf), 0 means invalid
    lidar_count_ds: torch.Tensor,    # (B,1,Hf,Wf)
    lidar_spread_ds: torch.Tensor,   # (B,1,Hf,Wf)
    hm_gt: torch.Tensor = None,      # (B,1,Hf,Wf) or (B,C,Hf,Wf) or (B,Hf,Wf)
    hm_thr: float = 0.3,             # use hm region instead of single center pixel
    K: int = 1,                      # minimum valid pixels per patch
    max_spread: float = 0.05,        # in SAME SCALE as depth (if normalized, also normalized)
    weight: float = 5.0,
    debug: bool = False,
    debug_prefix: str = "[LiDAR gate]",
):
    """
    Sparse LiDAR constraint with HM-based mask (recommended for sparse UAV LiDAR):
      mask = (hm_gt > hm_thr) AND lidar_valid
    lidar_valid = depth>0 AND count>=K AND spread<=max_spread

    Returns 0 if mask is empty.
    """

    pred = pred_center_dis[:, 0] if pred_center_dis.dim() == 4 else pred_center_dis
    gt = gt_center_dis[:, 0] if gt_center_dis.dim() == 4 else gt_center_dis

    # ---- mask from HM (preferred); fallback to gt_center_dis!=0 ----
    if hm_gt is not None:
        hm = hm_gt
        if hm.dim() == 4:
            # If multi-class HM, use max across classes so any class triggers
            if hm.size(1) > 1:
                hm = hm.max(dim=1).values
            else:
                hm = hm[:, 0]
        gt_mask = hm > hm_thr
    else:
        gt_mask = (torch.abs(gt) > 0)

    z_lidar = lidar_depth_ds[:, 0]
    cnt = lidar_count_ds[:, 0]
    spr = lidar_spread_ds[:, 0]

    lidar_has_depth = (z_lidar > 0)
    lidar_enough = (cnt >= K)
    lidar_tight = (spr <= max_spread)

    lidar_valid = lidar_has_depth & lidar_enough & lidar_tight
    mask = gt_mask & lidar_valid

    if debug:
        with torch.no_grad():
            total_roi = gt_mask.sum().item()
            has_depth_roi = (gt_mask & lidar_has_depth).sum().item()
            enough_roi = (gt_mask & lidar_has_depth & lidar_enough).sum().item()
            tight_roi = (gt_mask & lidar_has_depth & lidar_enough & lidar_tight).sum().item()
            valid_roi = mask.sum().item()

            # global stats
            total_cells = gt_mask.numel()
            frac_roi = total_roi / max(1, total_cells)

            print(
                f"{debug_prefix} roi={total_roi}({frac_roi*100:.3f}%) "
                f"has_depth={has_depth_roi} enough={enough_roi} tight={tight_roi} valid={valid_roi} "
                f"K={K} hm_thr={hm_thr} max_spread={max_spread}"
            )

            if total_roi > 0:
                # show average count/spread on ROI for diagnosing gate strictness
                roi_cnt = cnt[gt_mask]
                roi_spr = spr[gt_mask]
                if roi_cnt.numel() > 0:
                    print(
                        f"{debug_prefix} ROI cnt: mean={roi_cnt.float().mean().item():.3f}, "
                        f"p50={roi_cnt.float().median().item():.3f}, "
                        f"max={roi_cnt.float().max().item():.3f}"
                    )
                if roi_spr.numel() > 0:
                    # median on float tensors is ok
                    print(
                        f"{debug_prefix} ROI spr: mean={roi_spr.float().mean().item():.4f}, "
                        f"p50={roi_spr.float().median().item():.4f}, "
                        f"max={roi_spr.float().max().item():.4f}"
                    )

    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    return F.smooth_l1_loss(pred[mask], z_lidar[mask], reduction='mean') * float(weight)


# ----------------------------
# Model blocks
# ----------------------------

class DepthPredictionBlock(nn.Module):
    """专门的深度预测增强模块，针对LiDAR融合优化（新增抗过拟合机制）"""
    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.1):
        super(DepthPredictionBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channels, mid_channels, kernel_size=1) if in_channels != mid_channels else nn.Identity()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_channels, max(1, mid_channels // 2), kernel_size=1),
            nn.BatchNorm2d(max(1, mid_channels // 2)),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(max(1, mid_channels // 2), out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out) + residual
        out = self.bottleneck(out)
        out = self.out_conv(out)
        return out


class CenterHeadLidarFusion(nn.Module):
    """
    Full version with:
      - robust_downsample_depth for LiDAR depth map
      - HM-based LiDAR sparse constraint for center_dis
      - Debug prints for gate statistics and lidar_loss value

    Expected in batch_dict during training:
      - features_2d: (B, C, Hf, Wf)
      - lidar_depth_map: (B, 1, H_in, W_in) (should already be aligned to image plane)
      - hm: (B, 1 or C, Hf, Wf) GT heatmap at feature resolution
      - center_dis: (B, 1, Hf, Wf) GT distance (normalized or not, but must match lidar map scale)
    """
    def __init__(self, model_cfg):
        super(CenterHeadLidarFusion, self).__init__()

        self.model_cfg = model_cfg
        self.head_config = self.model_cfg.SEPARATE_HEAD_CFG
        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT
        self.in_c = self.model_cfg.INPUT_CHANNELS

        # --- LiDAR constraint hyperparams ---
        self.lidar_down_factor = getattr(self.model_cfg, "LIDAR_DOWN_FACTOR", 4)
        self.lidar_invalid_val = float(getattr(self.model_cfg, "LIDAR_INVALID_VAL", 0.0))
        self.lidar_min_count = int(getattr(self.model_cfg, "LIDAR_MIN_COUNT", 1))
        self.lidar_max_spread = float(getattr(self.model_cfg, "LIDAR_MAX_SPREAD", 0.05))
        self.lidar_loss_weight = float(getattr(self.model_cfg, "LIDAR_LOSS_WEIGHT", 5.0))
        self.lidar_hm_thr = float(getattr(self.model_cfg, "LIDAR_HM_THR", 0.3))

        # Debug controls
        self.lidar_debug = bool(getattr(self.model_cfg, "LIDAR_DEBUG", True))
        self.lidar_debug_every = int(getattr(self.model_cfg, "LIDAR_DEBUG_EVERY", 50))
        self._debug_step = 0

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            if cur_head_name == 'center_dis':
                fc = DepthPredictionBlock(self.in_c, conv_mid, out_c, dropout_rate=0.1)
            elif cur_head_name == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid * 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(conv_mid * 2, conv_mid, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(conv_mid, out_c, kernel_size=3, stride=1, padding=1, bias=True)
                )
                nn.init.constant_(fc[-1].bias, -2.25)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(conv_mid),
                    nn.ReLU(),
                    nn.Dropout2d(0.1),
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
            if cur_name in self.forward_loss_dict.get('pred_center_dict', {}) and cur_name in self.forward_loss_dict.get('gt_center_dict', {}):
                pred_h = self.forward_loss_dict['pred_center_dict'][cur_name]
                gt_h = self.forward_loss_dict['gt_center_dict'][cur_name]
                current_loss = 0

                pred = pred_h.reshape(-1)
                gt = gt_h.reshape(-1)
                mask_x = torch.abs(gt) > 0
                mask_count = mask_x.sum().item()

                if cur_name == 'hm':
                    l = F.binary_cross_entropy(torch.sigmoid(pred), gt, reduction='none')
                    l_map = l.sum() / (mask_count + 1)
                    weighted_loss = l_map * loss_weight_hm
                    current_loss = float(weighted_loss.item())
                    loss += weighted_loss

                if mask_count > 0:
                    if cur_name == 'center_dis':
                        base_loss = F.l1_loss(pred[mask_x], gt[mask_x], reduction='mean')
                        weighted_loss = base_loss * loss_weight_dis

                        # ---- LiDAR sparse constraint (HM-gated) ----
                        if ('lidar_depth_ds' in self.forward_loss_dict and
                            'lidar_count_ds' in self.forward_loss_dict and
                            'lidar_spread_ds' in self.forward_loss_dict):

                            # debug printing every N steps to avoid spamming
                            do_debug = self.lidar_debug and (self._debug_step % self.lidar_debug_every == 0)

                            hm_gt = self.forward_loss_dict['gt_center_dict'].get('hm', None)
                            lidar_loss = compute_lidar_depth_constraint(
                                pred_center_dis=pred_h,
                                gt_center_dis=gt_h,
                                lidar_depth_ds=self.forward_loss_dict['lidar_depth_ds'],
                                lidar_count_ds=self.forward_loss_dict['lidar_count_ds'],
                                lidar_spread_ds=self.forward_loss_dict['lidar_spread_ds'],
                                hm_gt=hm_gt,
                                hm_thr=self.lidar_hm_thr,
                                K=self.lidar_min_count,
                                max_spread=self.lidar_max_spread,
                                weight=self.lidar_loss_weight,
                                debug=do_debug,
                                debug_prefix="[LiDAR gate]",
                            )

                            # if do_debug:
                            #     print(f"[LiDAR loss] {lidar_loss.item():.6f} (w={self.lidar_loss_weight})")
                            print(f"[LiDAR loss] {lidar_loss.item():.6f} (w={self.lidar_loss_weight})")

                            weighted_loss = weighted_loss + lidar_loss

                        current_loss = float(weighted_loss.item())
                        loss += weighted_loss

                    if cur_name == 'rot':
                        mse_loss = F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean')
                        weighted_loss = mse_loss * loss_weight_rot
                        current_loss = float(weighted_loss.item())
                        loss += weighted_loss

                    if cur_name == 'center_res':
                        mse_loss = F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean')
                        weighted_loss = mse_loss * loss_weight_res
                        current_loss = float(weighted_loss.item())
                        loss += weighted_loss

                    if cur_name == 'dim':
                        mse_loss = F.l1_loss(pred[mask_x], gt[mask_x], reduction='mean')
                        weighted_loss = mse_loss * loss_weight_dim
                        current_loss = float(weighted_loss.item())
                        loss += weighted_loss

                head_losses[cur_name] = current_loss

        if head_losses:
            worst_head = max(head_losses, key=head_losses.get)
            print("\n各预测头loss值:")
            for head_name, head_loss in head_losses.items():
                print(f"{head_name}: {head_loss:.6f}")
            print(f"最差的预测是: {worst_head}, loss值: {head_losses[worst_head]:.6f}")

        self._debug_step += 1
        return loss

    def forward(self, batch_dict):
        pred_dict = {}
        gt_dict = {}

        x = batch_dict['features_2d']

        # ---- Build LiDAR robust depth map at feature resolution ----
        if 'lidar_depth_map' in batch_dict:
            lidar_depth_map = batch_dict['lidar_depth_map']
            depth_ds, count_ds, spread_ds = robust_downsample_depth(
                lidar_depth_map,
                down_factor=self.lidar_down_factor,
                invalid_val=self.lidar_invalid_val
            )
            batch_dict['lidar_depth_ds'] = depth_ds
            batch_dict['lidar_count_ds'] = count_ds
            batch_dict['lidar_spread_ds'] = spread_ds

        for cur_name in self.head_keys:
            head_module = self.__getattr__(cur_name)
            pred = head_module(x)

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

            # store lidar maps for loss
            if 'lidar_depth_ds' in batch_dict:
                self.forward_loss_dict['lidar_depth_ds'] = batch_dict['lidar_depth_ds']
                self.forward_loss_dict['lidar_count_ds'] = batch_dict['lidar_count_ds']
                self.forward_loss_dict['lidar_spread_ds'] = batch_dict['lidar_spread_ds']

        return batch_dict
