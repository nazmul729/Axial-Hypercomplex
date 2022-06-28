'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import os
from torchvision import io
import torch.nn as nn
import torch.nn.functional as F
#from work_dirs import *
from lib.models.quaternionconv.quaternion_layers import QuaternionConv
from lib.models.quaternionconv.QuatPHM import QPHMLayer
from lib.models.quaternionconv.quaternion_QLinearlayers import QLinear
from .utils import get_gaussian_filter


    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, std, stride=1):
        super(BasicBlock, self).__init__()
        
        self.planes = planes
        self.conv1 = QuaternionConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )   
    
    def forward(self, x):
        
        out = F.relu(self.bn1(self.kernel1(self.conv1(x))))
        out = self.bn2(self.kernel2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, width=False):
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes  
        self.width = width
        self.quat = QPHMLayer(4, in_planes, out_planes * 2)
        self.bn_output = nn.BatchNorm2d(out_planes * 2)

        #if stride > 1:
        #    self.pooling = nn.AvgPool2d(stride, stride=stride)

        #self.reset_parameters()

    def forward(self, x): 
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H) 
        stacked_output = self.quat(x) 
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, std, stride=1):
        super(Bottleneck, self).__init__()
        self.planes = planes
        self.conv1 = QuaternionConv(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuaternionConv(planes, self.expansion*planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
           
    def forward(self, x):        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.hight_block(out) #print("After Height-axis: ",out.shape)  #10 64 56 56
        out = self.width_block(out)
        
        out = F.relu(self.bn2(out))
        
        out = self.bn3(self.conv3(out))         #print("Input: ", x.shape, "Out: ",out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ResNet(nn.Module):
    expansion = 4
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 120
        self.conv1 = QuaternionConv(4, 120, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(120) 
        self.layer1 = self._make_layer(block, 120, num_blocks[0], self.std, stride=1)
        self.layer2 = self._make_layer(block, 240, num_blocks[1], self.std, stride=2)
        self.layer3 = self._make_layer(block, 480, num_blocks[2], self.std, stride=2)
        self.layer4 = self._make_layer(block, 960, num_blocks[3], self.std, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #self.linear = nn.Linear(896*block.expansion, num_classes)#
        self.linear = QLinear(4, 960*block.expansion, num_classes)
        #self.reset_parameters()
       
    def _make_layer(self, block, planes, num_blocks, std, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, std, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def Layer_Norm(self, x, num_planes=128):
        mean = x.sum(axis = 0)/(x.shape[0])
        std = ((((x - mean)**2).sum()/(x.shape[0]))+0.00001).sqrt()
        return (x - mean)/std
    
    def forward(self, x):
        #ss=update_Info()
        #print("Before stem: ",self.currentStd)
        out = F.relu(self.bn1(self.conv1(x)))   #out = self.maxpool(out) 
        out = self.layer1(out)    #  print("Layer1: ",out.shape) 
        out = self.layer2(out)    #  print("Layer2: ",out.shape)
        out = self.layer3(out)    #  print("Layer3: ",out.shape)
        out = self.layer4(out)    #  print("Layer4: ",out.shape)
        
        out = self.avgpool(out)              #print(out.shape) [500, 57344]
        out = torch.flatten(out, 1)
        out = self.linear(out)                  #print(out.shape)
        
        return out
    
    #def reset_parameters(self):
    #    self.CurrentStd = epoch_adjust(self.path)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet26(num_classes):
    return ResNet(BasicBlock, [2, 4, 4, 2], num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == '__main__':
    net = ResNet18(1000)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    y = net(torch.randn(128, 3, 256, 256))
    print(y.size())
