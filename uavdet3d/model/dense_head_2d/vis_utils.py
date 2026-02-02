import os
from typing import Optional
import torch
import numpy as np
import time
import torch.nn.functional as F


def _carla_to_opencv_R(device, dtype):
    return torch.tensor(
        [[0, 1, 0],
         [0, 0, -1],
         [1, 0, 0]],
        device=device, dtype=dtype
    )

def project_world_to_image_carla(world_pts, K, E_wc, eps=1e-6):
    """
    world_pts: (B,N,3) in world
    K: (B,3,3)
    E_wc: (B,4,4) camera->world
    Return u,v,z_cv: (B,N)
    """
    B, N, _ = world_pts.shape
    device, dtype = world_pts.device, world_pts.dtype

    ones = torch.ones((B, N, 1), device=device, dtype=dtype)
    pts_h = torch.cat([world_pts, ones], dim=-1)  # (B,N,4)

    E_cw = torch.inverse(E_wc)  # (B,4,4) world->carla_cam
    pts_carla = torch.matmul(pts_h, E_cw.transpose(1, 2))[..., :3]  # (B,N,3)

    R = _carla_to_opencv_R(device, dtype)  # (3,3)
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


def _to_scalar(x, default):
    if x is None:
        return float(default)
    if torch.is_tensor(x):
        return float(x.reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        return float(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)):
        return float(np.asarray(x).reshape(-1)[0])
    return float(x)


def _to_numpy_img(img_t: torch.Tensor) -> np.ndarray:
    """
    Accept:
      - (3,H,W) CHW
      - (H,W,3) HWC
    Return:
      - uint8 HWC BGR contiguous
    """
    x = img_t.detach().cpu()

    # squeeze possible batch dim = 1
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 3:
        raise ValueError(f"[vis] image must be 3D, got shape={tuple(x.shape)}")

    # CHW -> HWC
    if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = x.permute(1, 2, 0)

    if x.shape[-1] not in (1, 3):
        raise ValueError(f"[vis] image last dim must be 1 or 3, got shape={tuple(x.shape)}")

    x = x.numpy()

    # convert to uint8
    if x.dtype != np.uint8:
        # common normalize cases
        if np.isfinite(x).all():
            vmin, vmax = float(x.min()), float(x.max())
        else:
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            vmin, vmax = float(x.min()), float(x.max())

        # assume [0,1] or [-1,1]
        if vmax <= 1.5 and vmin >= -0.5:
            if vmin < 0:
                x = (x + 1.0) * 0.5
            x = np.clip(x, 0.0, 1.0) * 255.0
        else:
            # fallback min-max
            x = (x - vmin) / (vmax - vmin + 1e-6)
            x = np.clip(x, 0.0, 1.0) * 255.0

        x = x.astype(np.uint8)

    # ensure 3 channels
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    # RGB -> BGR
    x = x[..., ::-1]

    # contiguous
    x = np.ascontiguousarray(x)
    return x

def _draw_box_cv2(
    img_bgr: np.ndarray,
    box_xyxy,
    color,
    text: str = None,
    box_thickness: int = 1,
    font_scale: float = 0.2,
    text_thickness: int = 1,
    text_bg: bool = False,
):
    import cv2
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    # box
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, int(box_thickness))

    if text:
        org = (x1, max(0, y1 - 5))
        font = cv2.FONT_HERSHEY_PLAIN

        if text_bg:
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            # background rectangle (black)
            x_bg1 = x1
            y_bg1 = max(0, y1 - th - baseline - 8)
            x_bg2 = min(w - 1, x1 + tw + 4)
            y_bg2 = min(h - 1, y1)
            cv2.rectangle(img_bgr, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
            org = (x1 + 2, y1 - 4)

        cv2.putText(img_bgr, text, org, font, float(font_scale), 
                    color, int(text_thickness), cv2.LINE_8,)

@torch.no_grad()
def visualize_2d_boxes_on_images(
    batch_dict: dict,
    save_dir: str,
    max_vis: int = 2,
    topk: int = 10,
    hm_thr: float = 0.2,
    stride_default: float = 8.0,
    D_MAX_default: float = 150.0,
):
    """
    保存若干张 debug 图：
      - 绿色：GT 2D box（gt_boxes9d 投影）
      - 红色：Pred 2D box（用 pred center_res+dim+center_dis 近似）
    要求 batch_dict 至少包含：
      - image tensor: images / image / img (B,3,H,W) 或 (B,H,W,3)
      - intrinsic (B,3,3), extrinsic (B,4,4)
      - gt_boxes9d (B,M,9,3) 或 gt_boxes
      - pred_center_dict (from forward)
      - hm GT: batch_dict["hm"]（用它选点）
    """
    os.makedirs(save_dir, exist_ok=True)

    # -------- locate images --------
    img_key = None
    for k in ["images", "image", "img", "camera_images", "rgb"]:
        if k in batch_dict:
            img_key = k
            break
    if img_key is None:
        print("[vis] No image key found in batch_dict (tried images/image/img/camera_images/rgb). Skip.")
        return

    imgs = batch_dict[img_key]
    if not torch.is_tensor(imgs):
        imgs = torch.as_tensor(np.asarray(imgs))
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    B = imgs.shape[0]

    # -------- camera params --------
    if "intrinsic" not in batch_dict or "extrinsic" not in batch_dict:
        print("[vis] Missing intrinsic/extrinsic in batch_dict. Skip.")
        return

    K = batch_dict["intrinsic"]
    E = batch_dict["extrinsic"]
    if not torch.is_tensor(K):
        K = torch.as_tensor(np.asarray(K))
    if not torch.is_tensor(E):
        E = torch.as_tensor(np.asarray(E))

    # 强制 float32（投影更稳）
    K = K.detach().float().cpu()
    E = E.detach().float().cpu()

    # -------- gt boxes --------
    gt_boxes9d = None
    if "gt_boxes9d" in batch_dict:
        gt_boxes9d = batch_dict["gt_boxes9d"]
    elif "gt_boxes" in batch_dict:
        gt_boxes9d = batch_dict["gt_boxes"]
    if gt_boxes9d is None:
        print("[vis] Missing gt_boxes9d/gt_boxes. Skip GT draw.")
    else:
        if not torch.is_tensor(gt_boxes9d):
            gt_boxes9d = torch.as_tensor(np.asarray(gt_boxes9d))
        gt_boxes9d = gt_boxes9d.detach().float().cpu()

    # -------- pred dict --------
    if "pred_center_dict" not in batch_dict:
        print("[vis] Missing pred_center_dict. Skip pred draw.")
        pred_dict = None
    else:
        pred_dict = batch_dict["pred_center_dict"]

    # -------- stride/D_MAX --------
    stride = _to_scalar(batch_dict.get("stride", None), stride_default)
    D_MAX = float(getattr(getattr(batch_dict.get("model_cfg", None), "D_MAX", None), "__float__", lambda: D_MAX_default)()) \
        if "model_cfg" in batch_dict else D_MAX_default
    # 你这里 head 内部 D_MAX 可能不同；如果 batch_dict 没塞 model_cfg，就用默认

    # -------- select points from GT hm peaks --------
    if "hm" not in batch_dict:
        print("[vis] Missing hm GT in batch_dict. Skip (need it to pick points).")
        return
    hm_gt = batch_dict["hm"]
    if not torch.is_tensor(hm_gt):
        hm_gt = torch.as_tensor(np.asarray(hm_gt))
    hm_gt = hm_gt.detach().float().cpu()  # (B,C,Hf,Wf)

    B2, C, Hf, Wf = hm_gt.shape
    assert B2 == B, f"hm batch {B2} != img batch {B}"

    hm_w = hm_gt.amax(dim=1)  # (B,Hf,Wf)
    # simple peak mask (same as your logic, simplified)
    pad = 1
    hmax = F.max_pool2d(hm_w.unsqueeze(1), kernel_size=3, stride=1, padding=pad).squeeze(1)
    peak = (hm_w == hmax) & (hm_w > hm_thr)

    flat = hm_w.view(B, -1)
    peak_flat = peak.view(B, -1)
    score = torch.where(peak_flat, flat, torch.full_like(flat, -1e9))
    Kk = min(topk, score.shape[1])
    val, ind = torch.topk(score, k=Kk, dim=1)
    ys = (ind // Wf).long()
    xs = (ind % Wf).long()
    valid = val > -1e8  # (B,K)

    # -------- helpers: project GT boxes to 2D --------
    def project_gt_boxes_one(batch_i: int):
        if gt_boxes9d is None:
            return []
        g = gt_boxes9d[batch_i]  # (M,9,3)
        if g.numel() == 0:
            return []
        M = g.shape[0]
        corners = g[:, 1:, :].reshape(1, M * 8, 3)  # (1,M*8,3)
        Ki = K[batch_i].unsqueeze(0)
        Ei = E[batch_i].unsqueeze(0)
        uu, vv, zz = project_world_to_image_carla(corners, Ki, Ei, eps=1e-6)
        uu = uu.view(1, M, 8)[0]
        vv = vv.view(1, M, 8)[0]
        zz = zz.view(1, M, 8)[0]
        ok = (zz > 0).all(dim=-1)
        x1 = uu.min(dim=-1).values
        y1 = vv.min(dim=-1).values
        x2 = uu.max(dim=-1).values
        y2 = vv.max(dim=-1).values
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        out = []
        for m in range(M):
            if bool(ok[m].item()):
                out.append(boxes[m].tolist())
        return out

    # -------- helpers: pred boxes at selected points --------
    def pred_boxes_at_points(batch_i: int):
        if pred_dict is None:
            return []
        if not all(k in pred_dict for k in ["center_res", "dim", "center_dis"]):
            return []

        res = pred_dict["center_res"].detach().float().cpu()[batch_i:batch_i+1]  # (1,2,Hf,Wf)
        dim = pred_dict["dim"].detach().float().cpu()[batch_i:batch_i+1]         # (1,3,Hf,Wf)
        z01 = pred_dict["center_dis"].detach().float().cpu()[batch_i:batch_i+1]  # (1,1,Hf,Wf)
        Ki = K[batch_i].unsqueeze(0)                                             # (1,3,3)

        y = ys[batch_i:batch_i+1]
        x = xs[batch_i:batch_i+1]
        v = valid[batch_i:batch_i+1]
        if not v.any():
            return []

        # flatten gather
        ind_ = (y * Wf + x).long()  # (1,K)
        ind2 = ind_.unsqueeze(1).expand(1, 2, Kk)
        ind3 = ind_.unsqueeze(1).expand(1, 3, Kk)

        res_k = torch.gather(res.view(1, 2, -1), dim=2, index=ind2)  # (1,2,K)
        dim_k = torch.gather(dim.view(1, 3, -1), dim=2, index=ind3)  # (1,3,K)
        z_k   = torch.gather(z01.view(1, 1, -1), dim=2, index=ind_.unsqueeze(1))  # (1,1,K)

        u = (x.float() + res_k[:, 0]) * stride
        v_ = (y.float() + res_k[:, 1]) * stride

        Z = (z_k[:, 0] * D_MAX_default).clamp_min(1.0)

        fx = Ki[:, 0, 0].view(1, 1)
        fy = Ki[:, 1, 1].view(1, 1)

        # 注意：这里假设 dim channel: [?, w, h]（和你代码一致）
        w_m = dim_k[:, 1].clamp_min(0.1)
        h_m = dim_k[:, 2].clamp_min(0.1)

        w_px = fx * w_m / (Z + 1e-6)
        h_px = fy * h_m / (Z + 1e-6)

        x1 = u - 0.5 * w_px
        y1 = v_ - 0.5 * h_px
        x2 = u + 0.5 * w_px
        y2 = v_ + 0.5 * h_px

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)[0]  # (K,4)

        out = []
        for i in range(Kk):
            if bool(v[0, i].item()):
                out.append(boxes[i].tolist())
        return out

    # -------- draw --------
    for b in range(min(B, max_vis)):
        img_bgr = _to_numpy_img(imgs[b, 0])

        gt_boxes = project_gt_boxes_one(b)
        for j, box in enumerate(gt_boxes):
            _draw_box_cv2(
                img_bgr, box, (0, 255, 0),
                text=f"GT{j}",
                box_thickness=2,
                font_scale=0.7,
                text_thickness=2
            )

        pred_boxes = pred_boxes_at_points(b)
        for j, box in enumerate(pred_boxes):
            _draw_box_cv2(
                img_bgr, box, (0, 0, 255),
                text=f"Pred{j}",
                box_thickness=2,
                font_scale=0.7,
                text_thickness=2
            )

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"dbg_2dbox_{ts}_b{b}.png")
        import cv2
        cv2.imwrite(out_path, img_bgr)

    print(f"[vis] saved to {save_dir}")
