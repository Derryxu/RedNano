import torch
from torch import nn
import torch.nn.functional as F
from .util import FlattenLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,  kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels,  kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            #print(1,", ",in_channels,", ", out_channels)
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=1))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class GlobalAvgPool1d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()
    def forward(self, x):
        # print("XX",x.size()[2:])=88 #torch.Size([18, 512, 88])
        return F.avg_pool1d(x, kernel_size=x.size()[2:])


def resnet18(in_channels=5, out_channels=512):
    net = nn.Sequential(
        nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("dropout1",nn.Dropout(p=0.2)) 
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("dropout2",nn.Dropout(p=0.2)) 
    net.add_module("resnet_block4", resnet_block(256, out_channels, 2))
    net.add_module("global_avg_pool", GlobalAvgPool1d())
    #net.add_module("fc", FlattenLayer())
    #net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, output)))
    return net


def resnet18_2(in_channels=5, out_channels=256):
    net = nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(32, 32, 2, first_block=True))
    net.add_module("dropout1",nn.Dropout(p=0.2)) 
    net.add_module("resnet_block2", resnet_block(32, 64, 2))
    net.add_module("resnet_block3", resnet_block(64, 128, 2))
    net.add_module("dropout2",nn.Dropout(p=0.2)) 
    net.add_module("resnet_block4", resnet_block(128, out_channels, 2))
    net.add_module("global_avg_pool", GlobalAvgPool1d())
    return net


def resnet18_3(in_channels=5, out_channels=256):
    net = nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(32, 32, 2, first_block=True))
    net.add_module("dropout1",nn.Dropout(p=0.2)) 
    net.add_module("resnet_block2", resnet_block(32, 64, 2))
    net.add_module("resnet_block3", resnet_block(64, 128, 2))
    net.add_module("dropout2",nn.Dropout(p=0.2)) 
    net.add_module("resnet_block4", resnet_block(128, out_channels, 2))
    net.add_module("global_avg_pool", GlobalAvgPool1d())
    return net


class Resnet(nn.Module):
    def __init__(self, in_channels=5, out_channels=512):
        super(Resnet, self).__init__()
        self.resnet = resnet18(in_channels=in_channels, out_channels=out_channels)
    def forward(self,x):
        x = self.resnet(x).view(x.shape[0], -1)
        return x


class Resnet2(nn.Module):
    def __init__(self, in_channels=5, out_channels=256):
        super(Resnet2, self).__init__()
        self.resnet = resnet18_2(in_channels=in_channels, out_channels=out_channels)
    def forward(self,x):
        x = self.resnet(x).view(x.shape[0], -1)
        return x


class Resnet3(nn.Module):
    def __init__(self, in_channels=5, out_channels=256):
        super(Resnet3, self).__init__()
        self.resnet = resnet18_3(in_channels=in_channels, out_channels=out_channels)
    def forward(self,x):
        x = self.resnet(x).view(x.shape[0], -1)
        return x


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, out_channels,  kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool1d(x, kernel_size=x.size()[2:])
        return x