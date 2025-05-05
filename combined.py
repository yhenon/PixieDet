
# ===== file: models/backbone.py =====

import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from models import ConvDPUnit


class Conv_head(nn.Module):

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
    ):
        super(Conv_head, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels, True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class Conv4layerBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        withBNRelu=True,
    ):
        super(Conv4layerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class YuNetBackbone(nn.Module):

    def __init__(self, stage_channels, downsample_idx, out_idx):
        super().__init__()
        self.layer_num = len(stage_channels)
        self.downsample_idx = downsample_idx
        self.out_idx = out_idx
        self.model0 = Conv_head(*stage_channels[0])
        for i in range(1, self.layer_num):
            self.add_module(f'model{i}', Conv4layerBlock(*stage_channels[i]))
        self.init_weights()

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = []
        for i in range(self.layer_num):
            x = self.__getattr__(f'model{i}')(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out
# ===== file: models/config.py =====

def get():
    optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer_config = dict(grad_clip=None)

    lr_mult = 8
    lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=1500,
        warmup_ratio=0.001,
        step=[50 * lr_mult, 68 * lr_mult])
    runner = dict(type='EpochBasedRunner', max_epochs=80 * lr_mult)

    checkpoint_config = dict(interval=80)
    log_config = dict(
        interval=50,
        hooks=[dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')])
    dist_params = dict(backend='nccl')
    log_level = 'INFO'
    load_from = None
    resume_from = None
    workflow = [('train', 1)]
    dataset_type = 'RetinaFaceDataset'
    data_root = 'data/widerface/'
    train_root = 'data/widerface/'
    val_root = 'data/widerface/'
    img_norm_cfg = dict(mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=False)

    data = dict(
        samples_per_gpu=16,
        workers_per_gpu=4,
        train=dict(
            type='RetinaFaceDataset',
            ann_file='data/widerface/labelv2/train/labelv2.txt',
            img_prefix='data/widerface/WIDER_train/images/',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
                dict(
                    type='RandomSquareCrop',
                    crop_choice=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
                dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[0., 0., 0.],
                    std=[1., 1., 1.],
                    to_rgb=False),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                        'gt_keypointss'
                    ])
            ]),
        val=dict(
            type='RetinaFaceDataset',
            ann_file='data/widerface/labelv2/val/labelv2.txt',
            img_prefix='data/widerface/WIDER_val/images/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 640),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip', flip_ratio=0.0),
                        dict(
                            type='Normalize',
                            mean=[0., 0., 0.],
                            std=[1., 1., 1.],
                            to_rgb=False),
                        dict(type='Pad', size=(640, 640), pad_val=0),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]),
        test=dict(
            type='RetinaFaceDataset',
            ann_file='data/widerface/labelv2/val/labelv2.txt',
            img_prefix='data/widerface/WIDER_val/images/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 640),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip', flip_ratio=0.0),
                        dict(
                            type='Normalize',
                            mean=[0., 0., 0.],
                            std=[1., 1., 1.],
                            to_rgb=False),
                        dict(type='Pad', size=(640, 640), pad_val=0),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]))

    model = dict(
        type='YuNet',
        backbone=dict(
            type='YuNetBackbone',
            stage_channels=[[3, 16, 16], [16, 64], [64, 64], [64, 64], [64, 64],
                            [64, 64]],
            downsample_idx=[0, 2, 3, 4],
            out_idx=[3, 4, 5]),
        neck=dict(type='TFPN', in_channels=[64, 64, 64], out_idx=[0, 1, 2]),
        bbox_head=dict(
            type='YuNet_Head',
            num_classes=1,
            in_channels=64,
            shared_stacked_convs=1,
            stacked_convs=0,
            feat_channels=64,
            prior_generator=dict(
                type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            loss_bbox=dict(type='EIoULoss', loss_weight=5.0, reduction='sum'),
            use_kps=True,
            kps_num=5,
            loss_kps=dict(
                type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
            loss_obj=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
        ),
        train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            nms_pre=-1,
            min_bbox_size=0,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=-1,
        ))
    evaluation = dict(interval=1001, metric='mAP') 
    return model
# ===== file: models/head.py =====

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ConvDPUnit
from torch.nn.modules.utils import _pair


class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self, strides, offset=0.5):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes,
                    dtype=torch.float32,
                    device='cuda',
                    with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str): The device where the anchors will be put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda',
                                 with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) +
                   self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) +
                   self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                         stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                         stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                 arrange as (h, w).
            device (str): The device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 device='cuda'):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str, optional): The device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris
    
class YuNet_Head(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        shared_stacked_convs=2,
        stacked_convs=2,
        use_kps=False,
        kps_num=5,

    ):

        super().__init__()
        self.num_classes = num_classes
        self.NK = kps_num
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid_cls = True
        self.use_kps = use_kps
        self.shared_stack_convs = shared_stacked_convs
        prior_generators = {'type': 'MlvlPointGenerator', 'offset': 0, 'strides': [8, 16, 32]}
        self.prior_generator = MlvlPointGenerator(strides=prior_generators['strides'], offset=prior_generators['offset'])
        self.strides = self.prior_generator.strides
        self.strides_num = len(self.strides)

        self.fp16_enabled = False

        if self.shared_stack_convs > 0:
            self.multi_level_share_convs = nn.ModuleList()
        if self.stacked_convs > 0:
            self.multi_level_cls_convs = nn.ModuleList()
            self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_cls = nn.ModuleList()
        self.multi_level_bbox = nn.ModuleList()
        self.multi_level_obj = nn.ModuleList()
        if self.use_kps:
            self.multi_level_kps = nn.ModuleList()
        for _ in self.strides:
            if self.shared_stack_convs > 0:
                single_level_share_convs = []
                for i in range(self.shared_stack_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    single_level_share_convs.append(
                        ConvDPUnit(chn, self.feat_channels))
                self.multi_level_share_convs.append(
                    nn.Sequential(*single_level_share_convs))

            if self.stacked_convs > 0:
                single_level_cls_convs = []
                single_level_reg_convs = []
                for i in range(self.stacked_convs):
                    chn = self.in_channels if i == 0 and \
                        self.shared_stack_convs == 0 else self.feat_channels
                    single_level_cls_convs.append(
                        ConvDPUnit(chn, self.feat_channels))
                    single_level_reg_convs.append(
                        ConvDPUnit(chn, self.feat_channels))
                self.multi_level_reg_convs.append(
                    nn.Sequential(*single_level_reg_convs))
                self.multi_level_cls_convs.append(
                    nn.Sequential(*single_level_cls_convs))

            chn = self.in_channels if self.stacked_convs == 0 and \
                self.shared_stack_convs == 0 else self.feat_channels
            self.multi_level_cls.append(
                ConvDPUnit(chn, self.num_classes, False))
            self.multi_level_bbox.append(ConvDPUnit(chn, 4, False))
            if self.use_kps:
                self.multi_level_kps.append(
                    ConvDPUnit(chn, self.NK * 2, False))
            self.multi_level_obj.append(ConvDPUnit(chn, 1, False))
        self.init_weights()

    def init_weights(self):
        bias_cls = -4.59
        for i in [0, 1, 2]:
            self.multi_level_obj[i].conv2.bias.data.fill_(bias_cls)
            self.multi_level_obj[i].conv2.weight.data.fill_(0)
            self.multi_level_cls[i].conv2.bias.data.fill_(bias_cls)
            self.multi_level_cls[i].conv2.weight.data.fill_(0)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        if self.shared_stack_convs > 0:
            feats = [
                convs(feat)
                for feat, convs in zip(feats, self.multi_level_share_convs)
            ]

        if self.stacked_convs > 0:
            feats_cls, feats_reg = [], []
            for i in range(self.strides_num):
                feats_cls.append(self.multi_level_cls_convs[i](feats[i]))
                feats_reg.append(self.multi_level_reg_convs[i](feats[i]))
            cls_preds = [
                convs(feat)
                for feat, convs in zip(feats_cls, self.multi_level_cls)
            ]
            bbox_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_bbox)
            ]
            obj_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_obj)
            ]
            kps_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_kps)
            ]
        else:
            cls_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_cls)
            ]
            bbox_preds = [
                convs(feat)
                for feat, convs in zip(feats, self.multi_level_bbox)
            ]
            obj_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_obj)
            ]
            kps_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_kps)
            ]

        return cls_preds, bbox_preds, obj_preds, kps_preds

# ===== file: models/__init__.py =====

from torch import nn

class ConvDPUnit(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        withBNRelu=True,
    ):
        super(ConvDPUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1,
            bias=True,
            groups=out_channels)
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = self.relu(x)
        return x


# ===== file: models/tfpn.py =====

import torch.nn as nn
import torch.nn.functional as F
from models import ConvDPUnit

class TFPN(nn.Module):

    def __init__(self, in_channels, out_idx):
        super().__init__()
        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(
                ConvDPUnit(in_channels[i], in_channels[i], True))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats):
        num_feats = len(feats)

        # top-down flow
        for i in range(num_feats - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(
                feats[i], scale_factor=2., mode='nearest')

        feats[0] = self.lateral_convs[0](feats[0])
        outs = [feats[i] for i in self.out_idx]
        return outs
# ===== file: models/yunet.py =====

from models.config import get
from models.backbone import YuNetBackbone
from models.tfpn import TFPN
from models.head import YuNet_Head
from torch import nn
import torch


class Yunet(nn.Module):
    def __init__(self):
        super(Yunet, self).__init__()
        mm = get()
        self.backbone = YuNetBackbone(stage_channels=mm['backbone']['stage_channels'], downsample_idx=mm['backbone']['downsample_idx'], out_idx=mm['backbone']['out_idx'])
        self.neck = TFPN(in_channels=mm['neck']['in_channels'], out_idx=mm['neck']['out_idx'])

        self.bbox_head = YuNet_Head(num_classes=1, 
                                    in_channels=mm['bbox_head']['in_channels'], 
                                    feat_channels=mm['bbox_head']['feat_channels'],
                                    shared_stacked_convs=mm['bbox_head']['shared_stacked_convs'],
                                    stacked_convs=mm['bbox_head']['stacked_convs'],
                                    use_kps=True)
        self.num_kpts=5

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