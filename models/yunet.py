from models.config import get
from models.backbone import YuNetBackbone
from models.tfpn import TFPN
from models.head import YuNet_Head
from torch import nn
import torch


class Yunet(nn.Module):
    def __init__(self, num_kpts=5):
        super(Yunet, self).__init__()
        mm = get()
        self.backbone = YuNetBackbone(stage_channels=mm['backbone']['stage_channels'], downsample_idx=mm['backbone']['downsample_idx'], out_idx=mm['backbone']['out_idx'])
        self.neck = TFPN(in_channels=mm['neck']['in_channels'], out_idx=mm['neck']['out_idx'])

        self.bbox_head = YuNet_Head(num_classes=1, 
                                    in_channels=mm['bbox_head']['in_channels'], 
                                    feat_channels=mm['bbox_head']['feat_channels'],
                                    shared_stacked_convs=mm['bbox_head']['shared_stacked_convs'],
                                    stacked_convs=mm['bbox_head']['stacked_convs'],
                                    use_kps=True,
                                    kps_num=num_kpts)
        self.num_kpts=num_kpts

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        obj, bbox, cls, kpts = self.bbox_head(out)
        B = x.size(0)

        cls = torch.cat([cls[0].reshape(B, 1, -1), cls[1].reshape(B, 1, -1), cls[2].reshape(B, 1, -1)], dim=2).permute(0, 2, 1)
        bbox = torch.cat([bbox[0].reshape(B, 4, -1), bbox[1].reshape(B, 4, -1), bbox[2].reshape(B, 4, -1)], dim=2).permute(0, 2, 1)
        kpts = torch.cat([kpts[0].reshape(B, 2*self.num_kpts, -1), kpts[1].reshape(B, 2*self.num_kpts, -1), kpts[2].reshape(B, 2*self.num_kpts, -1)], dim=2).permute(0, 2, 1)
        obj = torch.cat([obj[0].reshape(B, 1, -1), obj[1].reshape(B, 1, -1), obj[2].reshape(B, 1, -1)], dim=2).permute(0, 2, 1)

        return cls, bbox, obj, kpts