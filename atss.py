import torch
import cv2
import numpy as np
from bbox import BBoxBatch

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) boxes in ``x1, y1, x2, y2`` format.
        boxes2: (M, 4) boxes in ``x1, y1, x2, y2`` format.
        eps:    Small value added to the union to avoid division by zero.

    Returns:
        Tensor of shape ``(N, M)`` containing pairwise IoU values.
    """
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)
    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)
