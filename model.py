import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torchsummary import summary
import torch.nn.functional as F
import numpy as np



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes= 10

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



#TODO ResNet18

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = num_classes,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
























#TODO Teacher
#input (64, 8, 8)

class ResNet_t(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = num_classes,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet_t, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 128)
        self.fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        x = self.layer2(x)  #(64, 8, 8)
        feature = x               #(128,4,4)
        x = self.layer3(x)
        x = self.layer4(x)   #(256,2,2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        pr_head = self.fc(x)
        y = nn.ReLU(inplace=True)(pr_head)
        out = self.fc2(y)


        return feature, pr_head, out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_t(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet_t:
    model = ResNet_t(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def teacher_model(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet_t:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_t('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

















#TODO Student
class ResNet_s(nn.Module):

    def __init__(
        self,
        Temperature: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = num_classes,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet_s, self).__init__()
        self.Temperature = Temperature
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_ct = nn.Linear(64 * block.expansion, 256)  #512
        self.fc_ct2= nn.Linear(256, 128)
        self.fc= nn.Linear(128 , num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        y = self.avgpool(feature)
        y = torch.flatten(y, 1)
        y = self.fc_ct(y)
        y = nn.ReLU(inplace=True)(y)
        y = self.fc_ct2(y)
        p = F.normalize(y, dim=1) #####
        out_ct= p.unsqueeze(1)    #projection head

        out = self.fc(nn.ReLU(inplace=True)(y))

        return feature, out_ct, out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_s(
    Temperature: int,
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet_s:
    model = ResNet_s(Temperature, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def student_model(Temperature, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet_s:    #resnet18
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_s(Temperature, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)



class Projection_head(nn.Module):
    def __init__(self):
        super(Projection_head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x= self.avgpool(x)
        x= torch.flatten(x,1)
        out_ct= self.fc(x)
        y = nn.ReLU(inplace=True)(out_ct)
        out = self.fc2(y)

        return out_ct, out

def projection_head():
    model = Projection_head()
    return model





class Decoder(nn.Module):
    def __init__(self, channel_size, feature_size):
        super(Decoder, self).__init__()
        self.channel_size=channel_size
        self.feature_size= feature_size

        if self.feature_size == 1:
            stride = 2
        elif self.feature_size == 2:
            stride = 2
        else:
            stride = 1

        if self.feature_size == 1:
            p1, p2 = 10, 12
        elif self.feature_size == 2:
            p1, p2 = 10, 12
        elif self.feature_size == 4:
            p1, p2 = 8, 8
        elif self.feature_size == 8:
            p1, p2 = 6, 8  # 6,8
        elif self.feature_size == 16:
            p1, p2 = 5, 5

        self.decoder = nn.Sequential(
            nn.Conv2d(self.channel_size, int(12), kernel_size=3, stride=stride, padding=p1),
            nn.BatchNorm2d(int(12)),
            nn.ReLU(),
            nn.Conv2d(int(12), 3, kernel_size=3, stride=1, padding=p2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)





class tiny_server_model_3(nn.Module):
    def __init__(self):
        super(tiny_server_model_3, self).__init__()
        self.conv1= nn.Conv2d(64, 128, kernel_size=3, stride= 2, padding=1, bias=False)
        self.bn1= nn.BatchNorm2d(128)
        self.relu1= nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2= nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv3= nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn3= nn.BatchNorm2d(512)
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        feature=x    #(128,4,4)
        y = self.avgpool2(feature)
        y = torch.flatten(y, 1)
        y = self.fc3(y)
        x = self.conv2(x)
        x = self.relu2(x)
        x= self.bn2(x)
        # x = self.conv3(x)
        # x = nn.ReLU(inplace=True)(self.bn3(x))
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        out_ct = nn.ReLU(inplace=True)(self.fc(x))
        out = self.fc2(out_ct)

        return feature, y, out   # 원래는 out_ct





class server_model_2(nn.Module):
    def __init__(self):
        super(server_model_2, self).__init__()
        self.conv1= nn.Conv2d(64, 128, kernel_size=3, stride= 2, padding=1, bias=False)
        self.bn1= nn.BatchNorm2d(128)
        self.conv1_2= nn.Conv2d(128, 128, kernel_size=2, stride= 1, padding=1, bias=False)
        self.bn1_2= nn.BatchNorm2d(128)
        self.conv1_3= nn.Conv2d(128, 128, kernel_size=3, stride= 1, padding=1, bias=False)
        self.bn1_3= nn.BatchNorm2d(128)

        self.conv_aux= nn.Conv2d(128, 256, kernel_size=3, stride= 3, padding=1, bias=False)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2= nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2= nn.BatchNorm2d(256)
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_3= nn.BatchNorm2d(256)

        self.conv3= nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3= nn.BatchNorm2d(512)
        self.conv3_2= nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_2= nn.BatchNorm2d(512)
        self.conv3_3= nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False)
        self.bn3_3= nn.BatchNorm2d(512)
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(self.bn1(x))
        x = self.conv1_2(x)
        x = nn.ReLU(inplace=True)(self.bn1_2(x))
        x = self.conv1_3(x)
        x = nn.ReLU(inplace=True)(self.bn1_3(x))  #128,5,5
        # z= self.conv_aux(x)
        z=x
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(self.bn2(x))
        x = self.conv2_2(x)
        x = nn.ReLU(inplace=True)(self.bn2_2(x))
        x = self.conv2_3(x)
        x = nn.ReLU(inplace=True)(self.bn2_3(x))
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(self.bn3(x))
        x = self.conv3_2(x)
        x = nn.ReLU(inplace=True)(self.bn3_2(x))
        x = self.conv3_3(x)
        x = nn.ReLU(inplace=True)(self.bn3_3(x))
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        out_ct = nn.ReLU(inplace=True)(self.fc(x))
        out = self.fc2(out_ct)

        return z, out_ct, out



class tiny_client_model(nn.Module):
    def __init__(self):
        super(tiny_client_model, self).__init__()
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride= 2, padding=1, bias=False)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
    def forward(self, x):
        x= self.enc_conv1(x)
        x= nn.ReLU(inplace=True)(x)
        x= self.dec_conv1(x)
        x= nn.ReLU()(x)

        return x



s= student_model(7)
s.to(device)
input= torch.ones((128, 3, 32,32)).to(device)
f, o, o_k = s(input)
feature_size= list(f.shape[1:])
# torchsummary.summary(s, (3,32,32), device='cpu')  # 64,8,8




# Temperature=7
# s= student_model(Temperature)
# s.to(device)
# summary(s, (3,32,32))
# print(s)
# pred= s(torch.Tensor(np.ones((128,3,32,32))))
# print(len(pred))
# out, out_kd= pred
# print(len(out))
# print(len(out_kd))
#
# t= teacher_model()
# # s.to(device)
# # summary(s, (3,32,32))
# # print(s)
# predt= t(torch.Tensor(np.ones((128, 64,8,8))))
# print(len(predt))



# client_models = [student_model(7).to(device) for _ in range(3)]
#
# for name, param in client_models[1].named_parameters():
#     print(name, param.size())

#
# s= student_model(7)
# s.to(device)
# input= torch.ones((128, 3, 32,32)).to(device)
# feature, out, out_ct= s(input)
# print(feature.shape)   #128, 64, 8,8
# print(out.shape)       #128,10
# print(out_ct.shape)    #128,1,128