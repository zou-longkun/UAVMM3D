# center_head_2d_consistency.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vis_utils import visualize_2d_boxes_on_images


# ------------------------ basic blocks ------------------------
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
    return normalized depth in [0,1], shape (B,1,H,W)
    """
    b_hat = torch.argmax(bin_logits, dim=1)  # (B,H,W)
    idx = b_hat.unsqueeze(1)                 # (B,1,H,W)
    r_hat = torch.gather(res_map, dim=1, index=idx).squeeze(1).clamp(-1.0, 1.0)
    z_hat = ((b_hat.to(bin_logits.dtype) + 0.5) * w_norm) + r_hat * (0.5 * w_norm)
    return z_hat.clamp(0.0, 1.0).unsqueeze(1)


def _to_scalar(x, name="value") -> float:
    if x is None:
        raise RuntimeError(f"{name} is None")
    if torch.is_tensor(x):
        if x.numel() == 0:
            raise RuntimeError(f"{name} tensor is empty")
        return float(x.reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        if x.size == 0:
            raise RuntimeError(f"{name} ndarray is empty")
        return float(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise RuntimeError(f"{name} list/tuple is empty")
        return _to_scalar(x[0], name=name)
    return float(x)


# ------------------------ projection utils ------------------------
def _as_torch(x, device, dtype, name: str):
    if x is None:
        raise RuntimeError(f"Missing {name} in batch_dict")
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _ensure_batch(M, B: int, name: str):
    if M.ndim == 2:
        return M.unsqueeze(0).expand(B, -1, -1)
    if M.ndim == 3:
        if M.shape[0] == B:
            return M
        if M.shape[0] == 1:
            return M.expand(B, -1, -1)
        raise RuntimeError(f"{name} batch mismatch: got {M.shape[0]} vs B={B}")
    raise RuntimeError(f"{name} must be 2D or 3D, got {tuple(M.shape)}")


def _carla_to_opencv_R(device, dtype):
    return torch.tensor(
        [[0, 1, 0],
         [0, 0, -1],
         [1, 0, 0]],
        device=device, dtype=dtype
    )


def project_world_to_image_carla(world_pts, K, E_wc, eps=1e-6):
    """
    world_pts: (B,N,3) world
    K: (B,3,3)
    E_wc: (B,4,4) camera->world (c2w)  ✅你说的默认就是这个
    """
    B, N, _ = world_pts.shape
    device, dtype = world_pts.device, world_pts.dtype

    ones = torch.ones((B, N, 1), device=device, dtype=dtype)
    pts_h = torch.cat([world_pts, ones], dim=-1)  # (B,N,4)

    # world -> carla_cam : inverse(E_wc)
    E_cw = torch.inverse(E_wc)  # world->carla_cam
    pts_carla = torch.matmul(pts_h, E_cw.transpose(1, 2))[..., :3]  # (B,N,3)

    # carla_cam -> opencv_cam
    R = _carla_to_opencv_R(device, dtype)
    pts_cv = torch.matmul(pts_carla, R.transpose(0, 1))  # (B,N,3)

    X = pts_cv[..., 0]
    Y = pts_cv[..., 1]
    Z = pts_cv[..., 2].clamp_min(eps)

    fx = K[:, 0, 0].view(B, 1)
    fy = K[:, 1, 1].view(B, 1)
    cx = K[:, 0, 2].view(B, 1)
    cy = K[:, 1, 2].view(B, 1)

    u = (X / Z) * fx + cx
    v = (Y / Z) * fy + cy
    return u, v, pts_cv[..., 2]


def unproject_image_to_world_carla(u, v, Z, K, E_wc, eps=1e-6):
    """
    u,v,Z: (B,N) pixels, pixels, meters (OpenCV cam)
    Return world points (B,N,3)
    """
    B, N = u.shape
    device, dtype = u.device, u.dtype

    fx = K[:, 0, 0].view(B, 1)
    fy = K[:, 1, 1].view(B, 1)
    cx = K[:, 0, 2].view(B, 1)
    cy = K[:, 1, 2].view(B, 1)

    Zc = Z.clamp_min(eps)
    X = (u - cx) * Zc / fx
    Y = (v - cy) * Zc / fy
    pts_cv = torch.stack([X, Y, Zc], dim=-1)  # (B,N,3)

    # OpenCV -> Carla cam
    R_c2v = _carla_to_opencv_R(device, dtype)  # carla->opencv
    R_v2c = R_c2v.transpose(0, 1)              # opencv->carla
    pts_carla = torch.matmul(pts_cv, R_v2c)

    ones = torch.ones((B, N, 1), device=device, dtype=dtype)
    pts_carla_h = torch.cat([pts_carla, ones], dim=-1)  # (B,N,4)

    # Carla cam -> world using E_wc (c2w)
    pts_w = torch.matmul(pts_carla_h, E_wc.transpose(1, 2))[..., :3]
    return pts_w


def corners_3d_from_center_dim_yaw(center_w, dim_lwh, yaw, z_centered=True):
    """
    center_w: (B,N,3)
    dim_lwh:  (B,N,3) [l,w,h]
    yaw:      (B,N) (around +Z)
    Return:   (B,N,8,3)
    """
    l = dim_lwh[..., 0].clamp_min(1e-3)
    w = dim_lwh[..., 1].clamp_min(1e-3)
    h = dim_lwh[..., 2].clamp_min(1e-3)

    x = 0.5 * l
    y = 0.5 * w
    if z_centered:
        z1 = 0.5 * h
        z0 = -0.5 * h
    else:
        z0 = torch.zeros_like(h)
        z1 = h

    offsets = torch.stack([
        torch.stack([ x,  y, z0], dim=-1),
        torch.stack([ x, -y, z0], dim=-1),
        torch.stack([-x, -y, z0], dim=-1),
        torch.stack([-x,  y, z0], dim=-1),
        torch.stack([ x,  y, z1], dim=-1),
        torch.stack([ x, -y, z1], dim=-1),
        torch.stack([-x, -y, z1], dim=-1),
        torch.stack([-x,  y, z1], dim=-1),
    ], dim=-2)  # (B,N,8,3)

    cy = torch.cos(yaw).unsqueeze(-1).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1).unsqueeze(-1)

    ox = offsets[..., 0:1]
    oy = offsets[..., 1:2]
    oz = offsets[..., 2:3]

    rx = cy * ox - sy * oy
    ry = sy * ox + cy * oy
    rz = oz
    rot_offsets = torch.cat([rx, ry, rz], dim=-1)
    return rot_offsets + center_w.unsqueeze(-2)


def decode_angles_a1_a2_a3_at_points(rot_map, ys, xs):
    """
    ✅ Match your decoder exactly:
      a1 = atan2(rot[1], rot[0])
      a2 = atan2(rot[3], rot[2])
      a3 = atan2(rot[5], rot[4])
    rot is tanh'ed => normalize each (sin,cos) pair before atan2.
    Return: a1,a2,a3 each (B,K)
    """
    B, R, H, W = rot_map.shape
    if R < 6:
        raise RuntimeError(f"rot must have >=6 channels to match your decoder, got R={R}")

    Kp = ys.shape[1]
    ind = (ys * W + xs).long()
    indR = ind.unsqueeze(1).expand(B, R, Kp)
    rot_flat = rot_map.view(B, R, -1)
    rot_k = torch.gather(rot_flat, dim=2, index=indR)  # (B,R,K)

    def _atan2_norm(s, c):
        n = torch.sqrt(s * s + c * c).clamp_min(1e-6)
        return torch.atan2(s / n, c / n)

    a1 = _atan2_norm(rot_k[:, 1], rot_k[:, 0])
    a2 = _atan2_norm(rot_k[:, 3], rot_k[:, 2])
    a3 = _atan2_norm(rot_k[:, 5], rot_k[:, 4])
    return a1, a2, a3


# ------------------------ main head ------------------------
class CenterHeadLidarFusion(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.head_keys = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.net_config = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT
        self.in_c = self.model_cfg.INPUT_CHANNELS

        self.D_MAX = float(getattr(self.model_cfg, "D_MAX", 150.0))
        self.BIN_W_M = float(getattr(self.model_cfg, "BIN_W_M", 5.0))
        self.bin_w_norm = self.BIN_W_M / self.D_MAX
        self.num_bins = int((self.D_MAX + self.BIN_W_M - 1e-6) // self.BIN_W_M)

        for cur_head_name in self.head_keys:
            out_c = self.net_config[cur_head_name]["out_channels"]
            conv_mid = self.net_config[cur_head_name]["conv_dim"]

            if cur_head_name == "center_dis":
                self.center_bin_head = DepthPredictionBlock(self.in_c, conv_mid, self.num_bins, dropout_rate=0.1)
                self.center_resbin_head = DepthPredictionBlock(self.in_c, conv_mid, self.num_bins, dropout_rate=0.1)
                setattr(self, "center_dis", nn.Identity())
            elif cur_head_name == "hm":
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

        self.loss_weight_hm = 10.0
        self.loss_weight_dim = 10.0
        self.loss_weight_rot = 1.0
        self.loss_weight_res = 1.0
        self.loss_weight_bincls = 1.0
        self.loss_weight_binres = 1.0

        # ✅ 2D consistency
        self.loss_weight_2d_cons = float(getattr(self.model_cfg, "LOSS_WEIGHT_2D_CONS", 0.5))
        self.cons_topk = int(getattr(self.model_cfg, "CONS_TOPK", 10))
        self.cons_hm_thr = float(getattr(self.model_cfg, "CONS_HM_THR", 0.2))
        self.eps = 1e-6

        # if your 3D center is bottom-centered, set False in cfg
        self.cons_z_centered = bool(getattr(self.model_cfg, "CONS_Z_CENTERED", True))
        self.cons_pixel_clip = float(getattr(self.model_cfg, "CONS_PIXEL_CLIP", 1e6))

    # -------------------- GT helper --------------------
    @staticmethod
    def _local_max_mask(hm, k=3):
        pad = (k - 1) // 2
        hmax = F.max_pool2d(hm.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)
        return hm.eq(hmax)

    def _select_cons_points(self, hm_gt):
        B, C, H, W = hm_gt.shape
        hm_w = hm_gt.amax(dim=1)  # (B,H,W)
        peak = self._local_max_mask(hm_w, k=3) & (hm_w > self.cons_hm_thr)

        flat = hm_w.view(B, -1)
        peak_flat = peak.view(B, -1)
        score = torch.where(peak_flat, flat, torch.full_like(flat, -1e9))

        K = min(self.cons_topk, score.shape[1])
        val, ind = torch.topk(score, k=K, dim=1)

        ys = (ind // W).long()
        xs = (ind % W).long()

        valid = val > -1e8
        w = torch.where(valid, val.clamp_min(1e-4), torch.zeros_like(val))
        return ys, xs, w, valid

    def _gt_box2d_at_points(self, gt_boxes9d, ys, xs, stride, K, E_wc):
        device = ys.device
        dtype = K.dtype
        B, Kp = ys.shape

        if gt_boxes9d is None:
            return (
                torch.zeros((B, Kp, 4), device=device, dtype=dtype),
                torch.zeros((B, Kp), device=device, dtype=torch.bool),
            )

        gt_boxes9d = _as_torch(gt_boxes9d, device, dtype, "gt_boxes9d")
        if gt_boxes9d.ndim != 4 or gt_boxes9d.shape[-2:] != (9, 3):
            raise RuntimeError(f"gt_boxes9d must be (B,M,9,3), got {tuple(gt_boxes9d.shape)}")

        with torch.no_grad():
            gt_boxes9d = gt_boxes9d.detach()
            Kd = K.detach()
            Ed = E_wc.detach()

            B2, M, _, _ = gt_boxes9d.shape

            centers_w = gt_boxes9d[:, :, 0, :]  # (B,M,3)
            uc, vc, zc = project_world_to_image_carla(centers_w, Kd, Ed, eps=self.eps)
            cx_f = uc / float(stride)
            cy_f = vc / float(stride)
            gt_center_valid = zc > 0

            corners_w = gt_boxes9d[:, :, 1:, :].reshape(B2, M * 8, 3)
            uu, vv, zz = project_world_to_image_carla(corners_w, Kd, Ed, eps=self.eps)
            uu = uu.view(B2, M, 8)
            vv = vv.view(B2, M, 8)
            zz = zz.view(B2, M, 8)
            box_valid = (zz > 0).all(dim=-1) & gt_center_valid

            x1 = uu.min(dim=-1).values
            y1 = vv.min(dim=-1).values
            x2 = uu.max(dim=-1).values
            y2 = vv.max(dim=-1).values
            gt_box2d_all = torch.stack([x1, y1, x2, y2], dim=-1)

            px = xs.float()
            py = ys.float()
            dx = px.unsqueeze(-1) - cx_f.unsqueeze(1)
            dy = py.unsqueeze(-1) - cy_f.unsqueeze(1)
            dist2 = dx * dx + dy * dy
            dist2 = torch.where(box_valid.unsqueeze(1), dist2, torch.full_like(dist2, 1e18))

            nn_idx = torch.argmin(dist2, dim=-1)
            min_d = torch.gather(dist2, dim=-1, index=nn_idx.unsqueeze(-1)).squeeze(-1)
            gt_valid = min_d < 1e17

            idx4 = nn_idx.unsqueeze(-1).expand(B2, Kp, 4)
            gt_box2d = torch.gather(gt_box2d_all, dim=1, index=idx4)
            gt_box2d = torch.where(gt_valid.unsqueeze(-1), gt_box2d, torch.zeros_like(gt_box2d))

        return gt_box2d, gt_valid

    def _pred_box2d_at_points(self, pred_dict, ys, xs, stride, K, E_wc):
        """
        ✅ Aligned with your eval: project predicted 3D corners -> 2D box.
        Return:
          box:    (B,K,4)
          pred_ok:(B,K) whether all corners are in front & finite
        Also fallback invalid ones to the stable approx box (u,v,Z,w,h).
        """
        B, Kp = ys.shape
        dtype = pred_dict["center_res"].dtype

        res = pred_dict["center_res"]  # (B,2,H,W)
        dim = pred_dict["dim"]         # (B,3,H,W)
        z01 = pred_dict["center_dis"]  # (B,1,H,W)
        rot = pred_dict["rot"]         # (B,6,H,W) in your setup

        _, _, H, W = res.shape
        ind = (ys * W + xs).long()
        ind2 = ind.unsqueeze(1).expand(B, 2, Kp)
        ind3 = ind.unsqueeze(1).expand(B, 3, Kp)

        res_flat = res.view(B, 2, -1)
        dim_flat = dim.view(B, 3, -1)
        z_flat = z01.view(B, 1, -1)

        res_k = torch.gather(res_flat, dim=2, index=ind2)
        dim_k = torch.gather(dim_flat, dim=2, index=ind3)
        z_k = torch.gather(z_flat, dim=2, index=ind.unsqueeze(1))

        u = (xs.to(dtype) + res_k[:, 0]) * float(stride)
        v = (ys.to(dtype) + res_k[:, 1]) * float(stride)
        Z = (z_k[:, 0] * self.D_MAX).clamp_min(1.0)  # meters

        # decode angles like your decoder; use a1 as yaw for corner gen
        a1, a2, a3 = decode_angles_a1_a2_a3_at_points(rot, ys, xs)
        yaw = a1

        dim_lwh = dim_k.permute(0, 2, 1).contiguous()  # (B,K,3) [l,w,h]
        center_w = unproject_image_to_world_carla(u, v, Z, K, E_wc, eps=self.eps)

        corners_w = corners_3d_from_center_dim_yaw(
            center_w=center_w,
            dim_lwh=dim_lwh,
            yaw=yaw,
            z_centered=self.cons_z_centered
        )  # (B,K,8,3)

        corners_w_flat = corners_w.view(B, Kp * 8, 3)
        uu, vv, zz = project_world_to_image_carla(corners_w_flat, K, E_wc, eps=self.eps)
        uu = uu.view(B, Kp, 8)
        vv = vv.view(B, Kp, 8)
        zz = zz.view(B, Kp, 8)

        pred_ok = (zz > 0).all(dim=-1) & torch.isfinite(uu).all(dim=-1) & torch.isfinite(vv).all(dim=-1)

        x1 = uu.min(dim=-1).values
        y1 = vv.min(dim=-1).values
        x2 = uu.max(dim=-1).values
        y2 = vv.max(dim=-1).values
        box = torch.stack([x1, y1, x2, y2], dim=-1).clamp(-self.cons_pixel_clip, self.cons_pixel_clip)

        # fallback for invalid ones (stable approx)
        fx = K[:, 0, 0].view(B, 1)
        fy = K[:, 1, 1].view(B, 1)
        w_m = dim_lwh[..., 1].clamp_min(0.1)
        h_m = dim_lwh[..., 2].clamp_min(0.1)
        w_px = fx * w_m / (Z + self.eps)
        h_px = fy * h_m / (Z + self.eps)
        box_fb = torch.stack([u - 0.5 * w_px, v - 0.5 * h_px, u + 0.5 * w_px, v + 0.5 * h_px], dim=-1)
        box_fb = box_fb.clamp(-self.cons_pixel_clip, self.cons_pixel_clip)

        box = torch.where(pred_ok.unsqueeze(-1), box, box_fb)
        pred_ok = torch.isfinite(box).all(dim=-1)
        return box, pred_ok

    # -------------------- loss --------------------
    def get_loss(self):
        loss = 0.0
        pred_dict = self.forward_loss_dict.get("pred_center_dict", {})
        gt_dict = self.forward_loss_dict.get("gt_center_dict", {})

        # ---------- original losses ----------
        for cur_name in self.head_keys:
            if cur_name not in pred_dict or cur_name not in gt_dict:
                continue

            pred_h = pred_dict[cur_name]
            gt_h = gt_dict[cur_name]
            # gt_h = gt_h.squeeze(1)

            if cur_name == "hm":
                l = F.binary_cross_entropy(torch.sigmoid(pred_h), gt_h, reduction="mean")
                loss = loss + l * self.loss_weight_hm
                continue

            if cur_name == "center_dis":
                bin_logits = pred_dict["center_bin_logits"]
                res_map = pred_dict["center_resbin_map"]

                gt_z = gt_h[:, 0]  # normalized [0,1]
                mask = gt_z > 0
                if mask.sum() == 0:
                    continue

                w = self.bin_w_norm
                b_gt = torch.floor(gt_z / w).to(torch.long).clamp(0, self.num_bins - 1)
                center_norm = (b_gt.to(gt_z.dtype) + 0.5) * w
                r_gt = ((gt_z - center_norm) / (0.5 * w)).clamp(-1.0, 1.0)

                logits_flat = bin_logits.permute(0, 2, 3, 1)[mask]
                b_flat = b_gt[mask]
                l_cls = F.cross_entropy(logits_flat, b_flat)

                idx = b_gt.unsqueeze(1)
                r_pred = torch.gather(res_map, dim=1, index=idx).squeeze(1)
                l_res = F.smooth_l1_loss(r_pred[mask], r_gt[mask], reduction="mean")

                loss = loss + l_cls * self.loss_weight_bincls + l_res * self.loss_weight_binres
                continue

            pred = pred_h.reshape(-1)
            gt = gt_h.reshape(-1)
            mask_x = torch.abs(gt) > 0
            if mask_x.sum() == 0:
                continue

            if cur_name == "dim":
                loss = loss + F.l1_loss(pred[mask_x], gt[mask_x], reduction="mean") * self.loss_weight_dim
            elif cur_name == "center_res":
                loss = loss + F.mse_loss(pred[mask_x], gt[mask_x], reduction="mean") * self.loss_weight_res
            elif cur_name == "rot":
                pass

        # ---------- 2D consistency (aligned with eval, normalized, stable) ----------
        if self.loss_weight_2d_cons > 0:
            if ("hm" in gt_dict) and ("center_res" in pred_dict) and ("center_dis" in pred_dict) and ("dim" in pred_dict) and ("rot" in pred_dict):
                hm_gt = gt_dict["hm"]
                # hm_gt = hm_gt.squeeze(1)
                B = hm_gt.shape[0]
                device = hm_gt.device
                dtype = hm_gt.dtype

                stride = _to_scalar(gt_dict.get("stride", None), name="stride")

                K = gt_dict.get("K", None)
                E_wc = gt_dict.get("E", None)
                if K is None or E_wc is None:
                    return loss

                K = _ensure_batch(K.to(device=device, dtype=dtype), B, "intrinsic")
                # ✅ NO protection, NO auto inverse: you said extrinsic is c2w
                E_wc = _ensure_batch(E_wc.to(device=device, dtype=dtype), B, "extrinsic")

                ys, xs, w, vmask = self._select_cons_points(hm_gt)
                if vmask.any():
                    gt_boxes9d = gt_dict.get("gt_boxes9d", None)

                    gt_box2d, gt_ok = self._gt_box2d_at_points(
                        gt_boxes9d=gt_boxes9d, ys=ys, xs=xs,
                        stride=stride, K=K, E_wc=E_wc
                    )
                    pred_box2d, pred_ok = self._pred_box2d_at_points(pred_dict, ys, xs, stride, K, E_wc)

                    ok = vmask & gt_ok & pred_ok
                    if ok.any():
                        # normalize by approx image size (Wf*stride, Hf*stride)
                        Hf, Wf = hm_gt.shape[2], hm_gt.shape[3]
                        img_w = float(Wf) * float(stride)
                        img_h = float(Hf) * float(stride)

                        pred_n = pred_box2d.clone()
                        gt_n = gt_box2d.clone()
                        pred_n[..., 0::2] /= (img_w + 1e-6)
                        pred_n[..., 1::2] /= (img_h + 1e-6)
                        gt_n[..., 0::2] /= (img_w + 1e-6)
                        gt_n[..., 1::2] /= (img_h + 1e-6)

                        ww = w[ok].clamp_min(1e-4)
                        l = F.smooth_l1_loss(pred_n[ok], gt_n[ok], reduction="none").sum(dim=-1)
                        loss = loss + (l * ww).sum() / (ww.sum() + 1e-6) * self.loss_weight_2d_cons

        return loss

    # -------------------- forward --------------------
    def forward(self, batch_dict):
        x = batch_dict["features_2d"]
        pred_dict, gt_dict = {}, {}

        for cur_name in self.head_keys:
            if cur_name == "center_dis":
                bin_logits = self.center_bin_head(x)
                res_map = self.center_resbin_head(x)
                center_dis = decode_center_dis_from_bins(bin_logits, res_map, self.bin_w_norm)
                pred_dict["center_bin_logits"] = bin_logits
                pred_dict["center_resbin_map"] = res_map
                pred_dict["center_dis"] = center_dis
            else:
                head = getattr(self, cur_name)
                pred = head(x)
                if cur_name == "rot":
                    pred = torch.tanh(pred)
                if cur_name == "hm" and not self.training:
                    pred = torch.sigmoid(pred)
                pred_dict[cur_name] = pred

            if self.training and cur_name in batch_dict:
                gt_dict[cur_name] = batch_dict[cur_name]

        if self.training and self.loss_weight_2d_cons > 0:
            device = x.device
            dtype = x.dtype
            B = x.shape[0]

            gt_dict["stride"] = batch_dict.get("stride", float(getattr(self.model_cfg, "STRIDE", 8.0)))

            if "gt_boxes9d" in batch_dict:
                gt_dict["gt_boxes9d"] = batch_dict["gt_boxes9d"]
            elif "gt_boxes" in batch_dict:
                gt_dict["gt_boxes9d"] = batch_dict["gt_boxes"]
            else:
                gt_dict["gt_boxes9d"] = None

            if "intrinsic" in batch_dict and "extrinsic" in batch_dict:
                K = _as_torch(batch_dict["intrinsic"], device, dtype, "intrinsic")
                E = _as_torch(batch_dict["extrinsic"], device, dtype, "extrinsic")

                while K.ndim > 3 and K.shape[1] == 1:
                    K = K.squeeze(1)
                while E.ndim > 3 and E.shape[1] == 1:
                    E = E.squeeze(1)

                gt_dict["K"] = _ensure_batch(K, B, "intrinsic")
                gt_dict["E"] = _ensure_batch(E, B, "extrinsic")

        batch_dict["pred_center_dict"] = pred_dict
        self.forward_loss_dict["pred_center_dict"] = pred_dict

        if self.training:
            batch_dict["gt_dict"] = gt_dict
            self.forward_loss_dict["gt_center_dict"] = gt_dict
        
        # visualize_2d_boxes_on_images(
        #     batch_dict=batch_dict,
        #     save_dir="./debug_vis_2d_consistency",
        #     max_vis=2,
        #     topk=10,
        #     hm_thr=0.2,
        #     stride_default=8.0,
        #     D_MAX_default=150.0,
        # )

        return batch_dict
