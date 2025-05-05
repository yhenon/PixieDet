import torch
import torch.nn.functional as F
from typing import Tuple
from bbox import BBoxBatch, _bbox_decode

def quality_focal_loss(
    pred: torch.Tensor,
    target: Tuple[torch.Tensor, torch.Tensor],
    beta: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Quality Focal Loss (QFL).

    Args:
        pred:        (N,) logits 
        target:      tuple of
                         labels: (N,) category indices in [0, 1]
                         quality: (N,) float IoU scores in [0, 1] for positive samples
        beta:        focusing parameter (2.0 in the paper)
        reduction:   'none' | 'mean' | 'sum'

    Returns:
        loss:        scalar if reduction!='none', else (N,) per-sample loss
    """
    labels, scores = target
    N = pred.size(0)
    # sigmoid predictions / scale factor
    pred_sigmoid = pred.sigmoid()
    # ----------------------------------
    # 1) negatives: supervised by quality=0
    # ----------------------------------
    zero_target = torch.zeros_like(pred)
    # BCE w/ logits against zero, modulated by p^β
    loss = F.binary_cross_entropy_with_logits(
        pred, zero_target, reduction="none"
    ) * pred_sigmoid.pow(beta)

    # ----------------------------------
    # 2) positives: supervised by their IoU quality
    # ----------------------------------
    # find valid (pos) indices
    valid = (labels == 1)
    if valid.any():
        pos_idx = valid.nonzero(as_tuple=False).squeeze(1)
        #pos_labels = labels[pos_idx].long()
        pos_scores = scores[pos_idx]

        # modulating factor = |q - p|^β
        scale_pos = (pos_scores - pred_sigmoid[pos_idx]).abs().pow(beta)

        # BCE w/ logits against target=q
        loss_pos = F.binary_cross_entropy_with_logits(
            pred[pos_idx],
            pos_scores,
            reduction="none",
        ) * scale_pos

        # replace the loss on the true class dim
        loss[pos_idx] = loss_pos

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # 'none'
        return loss


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'sum',
    num_pos_norm: float = 1.0, # The normalization factor (like your original code)
    eps: float = 1e-8        # Small epsilon for numerical stability
):
    targets = targets.float().to(logits.device)
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.where(targets == 1, p, 1 - p)
    modulating_factor = (1.0 - p_t).pow(gamma)
    alpha_factor = torch.where(targets == 1, alpha, 1.0 - alpha)
    focal_loss = alpha_factor * modulating_factor * ce_loss

    # Apply reduction
    if reduction == 'mean':
        loss = focal_loss.mean()
    elif reduction == 'sum':
        # Normalize by num_pos_norm as in the original code
        # Add epsilon to num_pos_norm to prevent division by zero
        loss = focal_loss.sum() / max(num_pos_norm, eps)
    elif reduction == 'none':
        loss = focal_loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Supported types: 'none', 'mean', 'sum'")

    return loss

def bbox_diou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Distance‑IoU loss between two sets of boxes in (x, y, w, h) format.

    Args:
        pred: (..., 4) boxes
        target: (..., 4) boxes
    Returns:
        diou loss averaged over last dim‑1
    """
    pred_xyxy = pred
    tgt_xyxy = target

    assert pred_xyxy.shape[-1] == 4, "Expected 4 values in last dim"
    assert tgt_xyxy.shape[-1] == 4, "Expected 4 values in last dim"
    assert pred_xyxy.shape[-2] == tgt_xyxy.shape[-2], "Expected same number of boxes in last dim"

    # check we are in xyxy format
    w_tgt = tgt_xyxy[..., 2] - tgt_xyxy[..., 0]
    h_tgt = tgt_xyxy[..., 3] - tgt_xyxy[..., 1]
    w_pred = pred_xyxy[..., 2] - pred_xyxy[..., 0]
    h_pred = pred_xyxy[..., 3] - pred_xyxy[..., 1]
    assert (w_tgt >= 0).all(), "Expected x1 < x2 in last dim"
    assert (h_tgt >= 0).all(), "Expected y1 < y2 in last dim"
    assert (w_pred >= 0).all(), "Expected x1 < x2 in last dim"
    assert (h_pred >= 0).all(), "Expected y1 < y2 in last dim"

    # Intersection
    inter_x1 = torch.max(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], tgt_xyxy[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    area_pred = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
    area_gt = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]) * (tgt_xyxy[..., 3] - tgt_xyxy[..., 1])

    union = area_pred + area_gt - inter_area + eps
    iou = inter_area / union

    # center distance
    cx_pred = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
    cy_pred = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
    cx_gt = (tgt_xyxy[..., 0] + tgt_xyxy[..., 2]) / 2
    cy_gt = (tgt_xyxy[..., 1] + tgt_xyxy[..., 3]) / 2
    center_dist_sq = (cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2

    # enclosing box diag
    enclose_x1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
    enclose_y1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
    enclose_x2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
    enclose_y2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])
    diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    diou = iou -  center_dist_sq / diag_sq
    loss = 1 - diou
    return loss.mean()

def bbox_eiou(pred, target, smooth_point=0.1, eps=1e-7):
    r"""Implementation of paper 'Extended-IoU Loss: A Systematic IoU-Related
     Method: Beyond Simplified Regression for Better Localization,

     <https://ieeexplore.ieee.org/abstract/document/9429909> '.

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    pred_xyxy = pred
    tgt_xyxy = target

    assert pred_xyxy.shape[-1] == 4, "Expected 4 values in last dim"
    assert tgt_xyxy.shape[-1] == 4, "Expected 4 values in last dim"
    assert pred_xyxy.shape[-2] == tgt_xyxy.shape[-2], "Expected same number of boxes in last dim"

    # check we are in xyxy format
    w_tgt = tgt_xyxy[..., 2] - tgt_xyxy[..., 0]
    h_tgt = tgt_xyxy[..., 3] - tgt_xyxy[..., 1]
    w_pred = pred_xyxy[..., 2] - pred_xyxy[..., 0]
    h_pred = pred_xyxy[..., 3] - pred_xyxy[..., 1]
    assert (w_tgt >= 0).all(), "Expected x1 < x2 in last dim"
    assert (h_tgt >= 0).all(), "Expected y1 < y2 in last dim"
    assert (w_pred >= 0).all(), "Expected x1 < x2 in last dim"
    assert (h_pred >= 0).all(), "Expected y1 < y2 in last dim"

    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # extent top left
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)

    # intersection coordinates
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    # extra
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)

    # Intersection
    intersection = (ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1) - (
        ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (
            iy1 - ey1)
    # Union
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (
        ty2 - ty1) - intersection + eps
    # IoU
    ious = 1 - (intersection / union)

    # Smooth-EIoU
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * (ious**2) / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point)

    return loss.mean()

def compute_losses_per_image(
    iou_pred: torch.Tensor,       # (N,) unnormalized
    obj_pred: torch.Tensor,       # (N,) unnormalized
    bbox_pred: BBoxBatch,         # (1, N, 4) predicted boxes (xyxy)
    labels: torch.Tensor,         # (N,) in [0, M-1] or -1 if no assigned bbox
    gt_boxes: BBoxBatch           # (1, M, 4) ground-truth (xyxy)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = obj_pred.device
    N = obj_pred.size(0)
    # convert to xyxy and pick positives
    b_pred = bbox_pred._ensure_xyxy()[0]           # (N,4)
    b_gt   = gt_boxes._ensure_xyxy()[0]            # (M,4)
    pos_mask = labels >= 0                         # (N,)

    # compute IoU quality targets (0 for negatives, IoU for positives)
    quality_targets = torch.zeros((N,), device=device)
    if pos_mask.any():
        pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
        # gather matched GT for each positive
        matched_gt = b_gt[labels[pos_idx]]

        # compute per-box IoU
        x1 = torch.max(b_pred[pos_idx, 0], matched_gt[:, 0])
        y1 = torch.max(b_pred[pos_idx, 1], matched_gt[:, 1])
        x2 = torch.min(b_pred[pos_idx, 2], matched_gt[:, 2])
        y2 = torch.min(b_pred[pos_idx, 3], matched_gt[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_pred = (b_pred[pos_idx, 2] - b_pred[pos_idx, 0]) * (b_pred[pos_idx, 3] - b_pred[pos_idx, 1])
        area_gt   = (matched_gt[:, 2] - matched_gt[:, 0]) * (matched_gt[:, 3] - matched_gt[:, 1])
        union = area_pred + area_gt - inter + 1e-7
        ious  = inter / union

        quality_targets[pos_idx] = ious.detach()

    num_pos = pos_mask.sum()
    num_pos_norm = torch.clamp(num_pos.float(), min=1.0)

    loss_iou = quality_focal_loss(iou_pred.squeeze(1), (pos_mask.long(), quality_targets), beta=2.0, reduction="sum") / num_pos_norm
    loss_obj = binary_focal_loss_with_logits(obj_pred, pos_mask.float().to(device), reduction='sum') / num_pos_norm

    if pos_mask.any():
        loss_bbox = bbox_diou(
            b_pred[pos_mask],
            matched_gt
        )
    else:
        loss_bbox = torch.tensor(0.0, device=device)

    return loss_obj, loss_iou, loss_bbox

