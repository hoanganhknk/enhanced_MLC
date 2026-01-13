# resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, 1)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            # Option A-like for CIFAR: 1x1 projection
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class CifarResNet(nn.Module):
    """
    ResNet for CIFAR. ResNet32 => n=5 blocks per stage (3 stages) => 6n+2=32.
    Returns hidden feature of dim=64 (after global avgpool) when return_h=True.
    """
    def __init__(self, block, n, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = _conv3x3(3, 16, 1)
        self.bn1   = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, n, stride=1)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_h: bool = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.adaptive_avg_pool2d(out, 1)
        feat = torch.flatten(out, 1)          # (B, 64)
        logits = self.fc(feat)

        if return_h:
            return logits, feat
        return logits

def resnet32(num_classes=10):
    return CifarResNet(BasicBlock, n=5, num_classes=num_classes)
