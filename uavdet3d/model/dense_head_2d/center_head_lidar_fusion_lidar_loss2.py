import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# utils: robust downsample lidar depth (0~1 scale)
# ----------------------------

def masked_median(x: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    big = torch.finfo(x.dtype).max
    x2 = x.masked_fill(~mask, big)
    x_sorted, _ = torch.sort(x2, dim=dim)

    valid_count = mask.sum(dim=dim)
    mid_idx = torch.clamp((valid_count - 1) // 2, min=0)
    med = torch.gather(x_sorted, dim, mid_idx.unsqueeze(dim)).squeeze(dim)
    med = torch.where(valid_count > 0, med, torch.zeros_like(med))
    return med, valid_count


def robust_downsample_depth(depth_map: torch.Tensor, down_factor: int = 4, invalid_val: float = 0.0):
    """
    depth_map: (B,1,H,W)  (IMPORTANT: expected 0~1 normalized if you train center_dis in 0~1)
    returns:
      depth_ds  (B,1,H//f,W//f) median per patch
      count_ds  (B,1,H//f,W//f) valid pixel count per patch
      spread_ds (B,1,H//f,W//f) p90-p10 per patch  (same scale as depth_map)
    """
    assert depth_map.dim() == 4 and depth_map.size(1) == 1, "depth_map must be (B,1,H,W)"
    B, _, H, W = depth_map.shape
    f = int(down_factor)
    Hf, Wf = H // f, W // f
    if Hf <= 0 or Wf <= 0:
        raise ValueError(f"Input depth too small for down_factor={f}: H={H}, W={W}")

    depth = depth_map[:, :, :Hf * f, :Wf * f]
    patches = F.unfold(depth, kernel_size=f, stride=f).transpose(1, 2)  # (B, Hf*Wf, f*f)
    valid = (patches != invalid_val)

    med, cnt = masked_median(patches, valid, dim=-1)  # (B, Hf*Wf)

    # p90 - p10
    big = torch.finfo(patches.dtype).max
    p2 = patches.masked_fill(~valid, big)
    p_sorted, _ = torch.sort(p2, dim=-1)  # (B, Hf*Wf, f*f)

    k = cnt
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


def compute_lidar_depth_constraint_gtpoints(
    pred_center_dis: torch.Tensor,   # (B,1,Hf,Wf) 0~1
    gt_center_dis: torch.Tensor,     # (B,1,Hf,Wf) 0~1, only centers are >0
    lidar_depth_ds: torch.Tensor,    # (B,1,Hf,Wf) 0~1
    lidar_count_ds: torch.Tensor,    # (B,1,Hf,Wf)
    lidar_spread_ds: torch.Tensor,   # (B,1,Hf,Wf) 0~1
    K: int = 5,
    max_spread: float = 0.02,        # 0~1 ; 0.02 ~= 3m when MAX_DIS=150
    max_delta_to_gt: float = 0.067,  # 0~1 ; 0.067 ~= 10m when MAX_DIS=150
    weight: float = 0.5,
    debug: bool = False,
):
    pred = pred_center_dis[:, 0]
    gt = gt_center_dis[:, 0]
    z_lidar = lidar_depth_ds[:, 0]
    cnt = lidar_count_ds[:, 0]
    spr = lidar_spread_ds[:, 0]

    gt_mask = gt > 0
    lidar_mask = (z_lidar > 0) & (cnt >= K)
    if max_spread is not None and max_spread > 0:
        lidar_mask = lidar_mask & (spr <= max_spread)
    if max_delta_to_gt is not None and max_delta_to_gt > 0:
        lidar_mask = lidar_mask & (torch.abs(z_lidar - gt) <= max_delta_to_gt)

    mask = gt_mask & lidar_mask

    if debug:
        total = int(gt_mask.sum().item())
        used = int(mask.sum().item())
        print(f"[LiDAR train@GT] gt_pts={total} used={used} "
              f"K={K} spr_thr={max_spread} d2gt_thr={max_delta_to_gt} w={weight}")
        if used > 0:
            d = torch.abs(z_lidar[mask] - gt[mask])
            print(f"  |lidar-gt| mean={float(d.mean().item()):.4f} p50={float(d.median().item()):.4f} max={float(d.max().item()):.4f}")
            print(f"  cnt mean={float(cnt[mask].float().mean().item()):.2f} spr mean={float(spr[mask].mean().item()):.4f}")

    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    return F.smooth_l1_loss(pred[mask], z_lidar[mask], reduction="mean") * float(weight)


# ----------------------------
# head blocks
# ----------------------------

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


class CenterHeadLidarFusion(nn.Module):
    """
    ✅ Training head only:
      - build lidar_depth_ds/count_ds/spread_ds at feature resolution
      - add LiDAR loss only on GT center_dis points (gt_center_dis>0)
      - NO inference-time replacement here (avoid double-calib)

    Assumptions:
      - center_dis GT/pred are normalized 0~1 (Z/MAX_DIS)
      - batch_dict['lidar_depth_map'] is also normalized 0~1 (invalid=0)
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT
        self.in_c = self.model_cfg.INPUT_CHANNELS

        # lidar downsample
        self.lidar_down_factor = int(getattr(self.model_cfg, "LIDAR_DOWN_FACTOR", 4))
        self.lidar_invalid_val = float(getattr(self.model_cfg, "LIDAR_INVALID_VAL", 0.0))

        # lidar train constraint (normalized scale!)
        self.lidar_w = float(getattr(self.model_cfg, "LIDAR_LOSS_WEIGHT", 0.5))
        self.lidar_K = int(getattr(self.model_cfg, "LIDAR_MIN_COUNT", 5))
        self.lidar_max_spread = float(getattr(self.model_cfg, "LIDAR_MAX_SPREAD", 3.0 / 150.0))      # default 3m/150
        self.lidar_max_delta_to_gt = float(getattr(self.model_cfg, "LIDAR_MAX_DELTA_TO_GT", 10.0 / 150.0))  # default 10m/150

        # debug
        self.lidar_debug = bool(getattr(self.model_cfg, "LIDAR_DEBUG", True))
        self.lidar_debug_every = int(getattr(self.model_cfg, "LIDAR_DEBUG_EVERY", 50))
        self._step = 0

        # heads
        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]['out_channels']
            conv_mid = self.net_config[cur_head_name]['conv_dim']

            if cur_head_name == 'center_dis':
                fc = DepthPredictionBlock(self.in_c, conv_mid, out_c, dropout_rate=0.1)
            elif cur_head_name == 'hm':
                fc = nn.Sequential(
                    nn.Conv2d(self.in_c, conv_mid * 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_mid * 2, conv_mid, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_mid, out_c, kernel_size=3, padding=1, bias=True),
                )
                nn.init.constant_(fc[-1].bias, -2.25)
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

    def get_loss(self):
        loss = 0.0
        head_losses = {}

        # your weights
        loss_weight_hm = 10.0
        loss_weight_dim = 10.0
        loss_weight_dis = 20.0
        loss_weight_rot = 1.0
        loss_weight_res = 1.0

        do_debug = self.lidar_debug and (self._step % self.lidar_debug_every == 0)

        pred_dict = self.forward_loss_dict.get('pred_center_dict', {})
        gt_dict = self.forward_loss_dict.get('gt_center_dict', {})

        for cur_name in self.head_keys:
            if cur_name not in pred_dict or cur_name not in gt_dict:
                continue

            pred_h = pred_dict[cur_name]
            gt_h = gt_dict[cur_name]
            current_loss = 0.0

            pred = pred_h.reshape(-1)
            gt = gt_h.reshape(-1)
            mask_x = torch.abs(gt) > 0
            mask_count = mask_x.sum().item()

            if cur_name == 'hm':
                l = F.binary_cross_entropy(torch.sigmoid(pred), gt, reduction='none')
                l_map = l.sum() / (mask_count + 1)
                weighted = l_map * loss_weight_hm
                loss = loss + weighted
                current_loss = float(weighted.item())

            if mask_count > 0:
                if cur_name == 'center_dis':
                    base = F.l1_loss(pred[mask_x], gt[mask_x], reduction='mean')
                    weighted = base * loss_weight_dis

                    # ✅ lidar loss on GT center points only
                    if ('lidar_depth_ds' in self.forward_loss_dict and
                        'lidar_count_ds' in self.forward_loss_dict and
                        'lidar_spread_ds' in self.forward_loss_dict):

                        lidar_loss = compute_lidar_depth_constraint_gtpoints(
                            pred_center_dis=pred_h,
                            gt_center_dis=gt_h,
                            lidar_depth_ds=self.forward_loss_dict['lidar_depth_ds'],
                            lidar_count_ds=self.forward_loss_dict['lidar_count_ds'],
                            lidar_spread_ds=self.forward_loss_dict['lidar_spread_ds'],
                            K=self.lidar_K,
                            max_spread=self.lidar_max_spread,
                            max_delta_to_gt=self.lidar_max_delta_to_gt,
                            weight=self.lidar_w,
                            debug=do_debug,
                        )
                        if do_debug:
                            print(f"[LiDAR loss] {lidar_loss.item():.6f}")

                        weighted = weighted + lidar_loss

                    loss = loss + weighted
                    current_loss = float(weighted.item())

                elif cur_name == 'rot':
                    weighted = F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean') * loss_weight_rot
                    loss = loss + weighted
                    current_loss = float(weighted.item())

                elif cur_name == 'center_res':
                    weighted = F.mse_loss(pred[mask_x], gt[mask_x], reduction='mean') * loss_weight_res
                    loss = loss + weighted
                    current_loss = float(weighted.item())

                elif cur_name == 'dim':
                    weighted = F.l1_loss(pred[mask_x], gt[mask_x], reduction='mean') * loss_weight_dim
                    loss = loss + weighted
                    current_loss = float(weighted.item())

            head_losses[cur_name] = current_loss

        if do_debug and head_losses:
            worst = max(head_losses, key=head_losses.get)
            print("\n[Head losses]")
            for k, v in head_losses.items():
                print(f"  {k}: {v:.6f}")
            print(f"  worst: {worst} ({head_losses[worst]:.6f})")

        self._step += 1
        return loss

    def forward(self, batch_dict):
        x = batch_dict['features_2d']
        pred_dict, gt_dict = {}, {}

        # build lidar ds maps (still normalized 0~1)
        if 'lidar_depth_map' in batch_dict and batch_dict['lidar_depth_map'] is not None:
            depth_ds, count_ds, spread_ds = robust_downsample_depth(
                batch_dict['lidar_depth_map'],
                down_factor=self.lidar_down_factor,
                invalid_val=self.lidar_invalid_val,
            )
            batch_dict['lidar_depth_ds'] = depth_ds
            batch_dict['lidar_count_ds'] = count_ds
            batch_dict['lidar_spread_ds'] = spread_ds

            if self.lidar_debug and (self._step % self.lidar_debug_every == 0):
                valid = (depth_ds > 0)
                ratio = float(valid.float().mean().item())
                print(f"[LiDAR ds] {tuple(depth_ds.shape)} valid_ratio={ratio*100:.3f}%")

        # run heads
        for cur_name in self.head_keys:
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

            if 'lidar_depth_ds' in batch_dict:
                self.forward_loss_dict['lidar_depth_ds'] = batch_dict['lidar_depth_ds']
                self.forward_loss_dict['lidar_count_ds'] = batch_dict['lidar_count_ds']
                self.forward_loss_dict['lidar_spread_ds'] = batch_dict['lidar_spread_ds']

        return batch_dict
