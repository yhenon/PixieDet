import torch
from typing import Optional


def _bbox_decode( priors, bbox_preds):
    xys = (bbox_preds[..., :2] * priors[..., 2:]) + priors[..., :2]
    whs = bbox_preds[..., 2:].exp() * priors[..., 2:]

    tl_x = (xys[..., 0] - whs[..., 0] / 2)
    tl_y = (xys[..., 1] - whs[..., 1] / 2)
    br_x = (xys[..., 0] + whs[..., 0] / 2)
    br_y = (xys[..., 1] + whs[..., 1] / 2)

    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes

class BBoxBatch:
    """
    A wrapper around a [B, N, 4] tensor of boxes.
    Internally stores in 'xyxy' format by default, but can convert on the fly.
    """
    def __init__(
        self,
        boxes: torch.Tensor,
        fmt: str = "xyxy",
        image_sizes: Optional[torch.Tensor] = None,  # shape [B, 2] as (height, width)
    ):
        """
        Args:
            boxes: Tensor of shape [B, N, 4].
            fmt: one of 'xyxy', 'xywh', 'cxcywh'.
            image_sizes: optional Tensor [B, 2] storing (H, W) for each batch entry.
        """
        assert boxes.ndim == 3 and boxes.size(-1) == 4, "Expected [B, N, 4]"
        self.boxes = boxes.float()
        self.format = fmt
        self.image_sizes = image_sizes

    def get_b(self, b: int):
        return BBoxBatch(self.boxes[b:b + 1], fmt=self.format, image_sizes=None if self.image_sizes is None else self.image_sizes[b:b + 1])

    def to(self, device: torch.device):
        self.boxes = self.boxes.to(device)
        if self.image_sizes is not None:
            self.image_sizes = self.image_sizes.to(device)
        return self

    def _ensure_xyxy(self) -> torch.Tensor:
        b = self.boxes
        if self.format == "xyxy":
            return b
        elif self.format == "xywh":
            x, y, w, h = b.unbind(-1)
            return torch.stack([x, y, x + w, y + h], dim=-1)
        elif self.format == "cxcywh":
            cx, cy, w, h = b.unbind(-1)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return torch.stack([x1, y1, x2, y2], dim=-1)
        else:
            raise ValueError(f"Unknown format {self.format}")

    def convert(self, new_fmt: str) -> "BBoxBatch":
        """Return a new BBoxBatch in the requested format."""
        xyxy = self._ensure_xyxy()
        if new_fmt == "xyxy":
            out = xyxy
        elif new_fmt == "xywh":
            x1, y1, x2, y2 = xyxy.unbind(-1)
            out = torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
        elif new_fmt == "cxcywh":
            x1, y1, x2, y2 = xyxy.unbind(-1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            out = torch.stack([cx, cy, x2 - x1, y2 - y1], dim=-1)
        else:
            raise ValueError(f"Unknown target format {new_fmt}")
        return BBoxBatch(out, fmt=new_fmt, image_sizes=self.image_sizes)

    @property
    def xyxy(self) -> torch.Tensor:
        return self._ensure_xyxy()

    @property
    def xywh(self) -> torch.Tensor:
        return self.convert("xywh").boxes

    @property
    def cxcywh(self) -> torch.Tensor:
        return self.convert("cxcywh").boxes

    def clip_to_image(self) -> "BBoxBatch":
        """
        Clip all boxes to lie within [0, W] × [0, H], per batch entry.
        Requires self.image_sizes to be set.
        """
        assert self.image_sizes is not None, "Need image_sizes to clip."
        xyxy = self._ensure_xyxy()
        H, W = self.image_sizes.unbind(-1)  # each is [B]
        x1, y1, x2, y2 = xyxy.unbind(-1)
        x1 = x1.clamp(0, W[:, None])
        y1 = y1.clamp(0, H[:, None])
        x2 = x2.clamp(0, W[:, None])
        y2 = y2.clamp(0, H[:, None])
        clipped = torch.stack([x1, y1, x2, y2], dim=-1)
        return BBoxBatch(clipped, fmt="xyxy", image_sizes=self.image_sizes)

    def area(self) -> torch.Tensor:
        """
        Compute the area of each box: returns [B, N].
        Negative widths/heights get zeroed.
        """
        x1, y1, x2, y2 = self._ensure_xyxy().unbind(-1)
        return ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0))

    def iou(self, other: "BBoxBatch") -> torch.Tensor:
        """
        Compute pairwise IoU between self and other.
        Returns a [B, N, M] tensor, where self has N boxes, other has M.
        """
        # vectorized batch‐wise box IoU
        # (for brevity, here’s the standard corner‐based formula)
        a = self._ensure_xyxy()
        b = other._ensure_xyxy()
        # shapes [B, N, 1, 4] and [B, 1, M, 4]
        a = a[:, :, None]
        b = b[:, None, :]
        max_xy = torch.min(a[..., 2:], b[..., 2:])
        min_xy = torch.max(a[..., :2], b[..., :2])
        inter = (max_xy - min_xy).clamp(min=0)
        inter_area = inter[..., 0] * inter[..., 1]
        area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
        area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
        union = area_a + area_b - inter_area
        return inter_area / union.clamp(min=1e-6)

    def __repr__(self):
        B, N, _ = self.boxes.shape
        return f"<BBoxBatch boxes=[{B}×{N}×4] fmt={self.format}>"
