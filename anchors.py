import torch
import numpy as np
import math

def _meshgrid(feature_h: int, feature_w: int, stride: int, device: torch.device, add_half_stride=True):
    """Return (N,2) tensor of (x,y) centers for one feature level."""
    if add_half_stride:
        shifts_x = torch.arange(0, feature_w * stride, step=stride, dtype=torch.float32, device=device) + stride / 2
        shifts_y = torch.arange(0, feature_h * stride, step=stride, dtype=torch.float32, device=device) + stride / 2
    else:
        shifts_x = torch.arange(0, feature_w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, feature_h * stride, step=stride, dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    points = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=-1)  # (N,2)
    return points


def generate_candidate_points(image_size: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32), 
                              device: str | torch.device = "cpu",
                              add_half_stride = True):
    """Generate anchor‑free candidate locations for ATSS / simOTA.

    Args:
        image_size: (height, width) of input image.
        strides:     strides for each pyramid level (p2‑p4 by default).
        device:      torch device (string or torch.device).

    Returns:
        points_per_lvl: list with one (N_i,2) tensor per feature level containing (x_center, y_center).
        strides_per_lvl: list of same length with scalar stride for each level.
    """
    H, W = image_size
    points_per_lvl = []
    strides_per_lvl = []
    for s in strides:
        feat_h = math.ceil(H / s)
        feat_w = math.ceil(W / s)
        pts = _meshgrid(feat_h, feat_w, s, device=torch.device(device), add_half_stride=add_half_stride)
        points_per_lvl.append(pts)
        strides_per_lvl.extend([s] * len(pts))
    return torch.cat(points_per_lvl,dim=0), torch.from_numpy(np.array(strides_per_lvl)).to(device=device)
