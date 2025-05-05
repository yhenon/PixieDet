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