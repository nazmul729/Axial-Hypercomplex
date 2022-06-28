import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from lib.models.quaternionconv.quaternion_layers import QuaternionConv
from lib.models.vectormapconv.vectormap_layers import VectorMapConv, VectorMapBatchNorm2d
from lib.models.quaternionconv.QuatPHM import QPHMLayer
from lib.models.quaternionconv.quaternion_QLinearlayers import QLinear

__all__ = ['axial26s', 'axial50s', 'axial50m', 'axial35m', 'axial26m']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, bias=False, width=False):
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes  
        self.kernel_size = kernel_size
        self.stride = stride 
        self.width = width

        # Multi-head self attention
        self.quat = VectorMapConv(3, self.in_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_output = nn.BatchNorm1d(out_planes)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        #self.reset_parameters()

    def forward(self, x): 
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        stacked_output = self.quat(x)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, H)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=3):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) 
        self.conv_down = QuaternionConv(inplanes, planes, kernel_size=1, stride=1, bias=False)
        
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(planes, planes, kernel_size=kernel_size, stride=1)
        self.width_block = AxialAttention(planes, planes, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = QuaternionConv(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)  
	
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out) 

        out = self.conv_up(out)
        out = self.bn2(out) 
        if self.downsample is not None:
            identity = self.downsample(x) 
        out += identity 
        out = self.relu(out) 
        return out


class AxialAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, norm_layer=None, s=0.5):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(120 * s)
        self.conv1 = QuaternionConv(4, self.inplanes, kernel_size=3, stride=1, padding=1,  bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, int(120 * s), layers[0], kernel_size=32)
        self.layer2 = self._make_layer(block, int(240 * s), layers[1], stride=2, kernel_size=32)
        self.layer3 = self._make_layer(block, int(480 * s), layers[2], stride=2, kernel_size=16)
        self.layer4 = self._make_layer(block, int(960 * s), layers[3], stride=2, kernel_size=8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QLinear(5, int(960*block.expansion*s), num_classes)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QuaternionConv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
	
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
	
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        return x

    def forward(self, x):
        return self._forward_impl(x)


def axial26s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [1, 2, 4, 1], s=0.5, **kwargs)
    return model

def axial26m(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [1, 2, 4, 1], s=1, **kwargs)
    return model


def axial50s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    return model


def axial50m(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=1, **kwargs)
    return model


def axial35m(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [2, 3, 4, 2], s=1, **kwargs)
    return model
