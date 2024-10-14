import torch
import torch.nn as nn
from typing import Optional, List, Type, Union, Any

class BasicBlock(nn.Module):
    """building block used for resnet with less than 50 layers i.e. resnet 18,34"""
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, v: float =1.0,
                  shortcut: Optional[nn.Module] = None, kernel_size: int = 3) -> None:
        super().__init__()
        if v>1.5:
            self.bn_pre = nn.BatchNorm2d(num_features=inplanes)
            self.relu_pre = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes*self.expansion, kernel_size, stride=1, padding=1, bias=False)
        if v<2.0:
            self.bn2 = nn.BatchNorm2d(num_features=planes*self.expansion)
        self.shortcut = shortcut
        self.v = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v >1.5: # case version 2
            out = self.bn_pre(x)
            preact = self.relu_pre(out)
            out = self.conv1(preact)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.v <2.0: # case version 1
            out = self.bn2(out)

        if self.shortcut is None:
            shortcut = x    # identity
        else:
            if self.v>1.5 and isinstance(self.shortcut, nn.Conv2d): # conv when v==2.0
                shortcut = self.shortcut(preact)
            else:
                shortcut = self.shortcut(x) # conv when v==1.0 or maxpool when v==2.0

        out += shortcut
        if self.v<2.0: # case version 1
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """building block used for resnet with less than 50 layers i.e. resnet 18,34"""
    expansion: int = 4
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                  shortcut: Optional[nn.Module] = None, v:float=1.5, kernel_size: int = 3)->None:
        super().__init__()
        self.v=v
        if v<1.5: # case version 1
            s1, s2 = stride, 1
        else: # case version 2
            s1, s2 = 1, stride
        if v>1.5:
            self.bn_pre = nn.BatchNorm2d(num_features=inplanes)
            self.relu_pre = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=s1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size, stride=s2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False)
        if v<2.0:
            self.bn3 = nn.BatchNorm2d(num_features=planes*self.expansion)
        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v >1.5: # case version 2
            out = self.bn_pre(x)
            preact = self.relu_pre(out)
            out = self.conv1(preact)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.v <2.0: # case version 1 1.5
            out = self.bn3(out)
        if self.shortcut is None:
            shortcut = x    # identity
        else:
            if self.v>1.5 and isinstance(self.shortcut, nn.Conv2d): # conv when v==2.0
                shortcut = self.shortcut(preact)
            else:
                shortcut = self.shortcut(x) # conv when v==1.0 or maxpool when v==2.0

        out += shortcut
        if self.v<2.0: # case version 1
            out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000,
        v:float=1.5, zero_init_residual: bool = False,) -> None:
        super().__init__()
        self.group_count = 1
        self.v = v
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if v<2.0:
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)   
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if v>1.5:
            s1=2;s4=1 
        else:
            s1=1;s4=2
        self.group1 = self._make_group(block, 64, layers[0],stride=s1)
        self.group2 = self._make_group(block, 128, layers[1], stride=2)
        self.group3 = self._make_group(block, 256, layers[2], stride=2)
        self.group4 = self._make_group(block, 512, layers[3], stride=s4)
        if v>1.5:
            self.post_bn = nn.BatchNorm2d(512*block.expansion)
            self.post_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_group(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, num_blocks: int, 
                    stride: int = 1, dilate: bool = False,) -> nn.Sequential:
        shortcut = None
        v = self.v
        if v<2.0:
            b1,b3 = stride, 1
        else:
            b1, b3= 1, stride

        if v<2.0:
            shortcut = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        else:
            shortcut = nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=1, bias=False)
        if block == BasicBlock and self.group_count==1:
            shortcut = None
        
        layers = [ block(self.inplanes, planes, stride=b1, shortcut=shortcut, v=v)]
        self.inplanes = planes*block.expansion
        for _ in range(1, num_blocks-1):
            layers.append( block(self.inplanes, planes, v=v) )
        
        if v<2.0:
            shortcut = None
        else:
            if stride == 1:
                shortcut = None
            else:
                shortcut = nn.MaxPool2d(kernel_size=1, stride=stride)
            
        self.inplanes = planes*block.expansion
        layers.append( block(self.inplanes, planes, stride=b3, shortcut=shortcut,v=v) )
        self.group_count += 1

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.v<2.0:
            x=self.bn1(x)
            x=self.relu(x)   
        x = self.maxpool(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        if self.v>1.5:
            x = self.post_bn(x)
            x = self.post_relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




def ResNet18(v:float=1.0, num_classes:int=1000, zero_init_residual: bool = False) -> ResNet:
    assert v in [1.0], f"(ResNet50 function) v={v} is not in the allowed list [1.0]"
    return ResNet(BasicBlock, [2, 2, 2, 2], v=v, num_classes=num_classes, zero_init_residual=zero_init_residual)

def ResNet34(v:float=1.0, num_classes:int=1000, zero_init_residual: bool = False) -> ResNet:
    assert v in [1.0,2.0], f"(ResNet50 function) v={v} is not in the allowed list [1.0,2.0]"
    return ResNet(BasicBlock, [3, 4, 6, 3], v=v, num_classes=num_classes, zero_init_residual=zero_init_residual)

def ResNet50(v:float=1.0, num_classes:int=1000, zero_init_residual: bool = False) -> ResNet:
    assert v in [1.0, 1.5, 2.0], f"(ResNet50 function) v={v} is not in the allowed list [1.0, 1.5, 2.0]"
    return ResNet(Bottleneck, [3, 4, 6, 3], v=v, num_classes=num_classes, zero_init_residual=zero_init_residual)




