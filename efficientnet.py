
#自定义的激活函数
def ding_activation21(x):
    return x * torch.sigmoid(2*x)

#Efficientnet 系列模型
import copy
import math
import torch

from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Sequence

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops.misc import ConvNormActivation############修改了
# from  cbam import ChannelAttentionModule,SpatialAttentionModule,CBAM##########################################修改了
# from ..ops.misc import ConvNormActivation, SqueezeExcitation
from torchvision.models._utils import _make_divisible
from torchvision.ops import StochasticDepth



import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
###### My Definition
class CoordAtt(nn.Module):
  def __init__(self, channel, h, w, reduction=32):
    super(CoordAtt, self).__init__()
    self.h = h
    self.w = w

    self.pool_h = nn.AdaptiveAvgPool2d((h, 1))
    self.pool_w = nn.AdaptiveAvgPool2d((1, w))
    print((h, 1))
    print((1, w))

    # mip = max(8, channel//reduction)

    # self.conv1 = nn.Conv2d(in_channels=channel, mip, kernel_size=1, stride=1, padding=0)
    self.conv1 = nn.Conv2d(in_channels=channel, out_channels = max(8, channel//reduction), kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(max(8, channel//reduction))
    self.act = h_swish()

    self.conv_h = nn.Conv2d(in_channels = max(8, channel//reduction), out_channels=channel, kernel_size=1, stride=1, padding=0)
    self.conv_w = nn.Conv2d(in_channels = max(8, channel//reduction), out_channels=channel, kernel_size=1, stride=1, padding=0)
  
  def forward(self, x):
    identity = x
    n,c,h,w = x.size()
    x_h = self.pool_h(x)
    x_w = self.pool_w(x).permute(0, 1, 3, 2)

    y = torch.cat([x_h, x_w], dim=2)
    y = self.conv1(y)
    y = self.bn1(y)
    y = self.act(y)

    x_h, x_w = torch.split(y, [h, w], dim=2)
    x_w = x_w.permute(0, 1, 3, 2)

    a_h = self.conv_h(x_h).sigmoid()
    a_w = self.conv_w(x_w).sigmoid()

    out = identity * a_w * a_h

    return out















__all__ = ["EfficientNet", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
           "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]


model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    # 在EfficientNet文件的表1中列出的存储信息
    def __init__(self,
                 expand_ratio: float, kernel: int, stride: int,
                 input_channels: int, out_channels: int, num_layers: int,
                 width_mult: float, depth_mult: float) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'expand_ratio={expand_ratio}'
        s += ', kernel={kernel}'
        s += ', stride={stride}'
        s += ', input_channels={input_channels}'
        s += ', out_channels={out_channels}'
        s += ', num_layers={num_layers}'
        s += ')'
        return s.format(**self.__dict__)

    #定义 调整通道
    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    #定义 调整深度
    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module],
                 # se_layer: Callable[..., nn.Module] = SqueezeExcitation) -> None:
                 #cbam_layer: Callable[..., nn.Module] = CBAM) -> None:########### CBAM修改了
                 ca_layer: Callable[..., nn.Module] = CoordAtt, h=None, w=None) -> None:########### CA修改了
                 

        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
#修改部分
        activation_layer = nn.SiLU
        #activation_layer = ding_activation21


        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, expanded_channels, kernel_size=1,
                                             norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(ConvNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel,
                                         stride=cnf.stride, groups=expanded_channels,
                                         norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
#修改部分
        # layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))
        #layers.append(cbam_layer(expanded_channels))#########修改了
        layers.append(ca_layer(expanded_channels, h, w))#########修改了

        #layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(ding_activation21, inplace=True)))

        # project
        layers.append(ConvNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                         activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[MBConvConfig],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
#修改部分
        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                         activation_layer=nn.SiLU))
#        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
#                                         activation_layer=ding_activation21))

        # building inverted residual blocks
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        
        # edit
        current_size = 112  # because the `ConvNormActivation` at line 250 uses stride=2 so the size is divided by 2 here
        # /edit
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                if block_cnf.stride == 2:
                    current_size /= 2
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer, h=current_size, w=current_size))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
#修改部分
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                         norm_layer=norm_layer, activation_layer=nn.SiLU))
#        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
#                                         norm_layer=norm_layer, activation_layer=ding_activation21))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet_conf(width_mult: float, depth_mult: float, **kwargs: Any) -> List[MBConvConfig]:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    return inverted_residual_setting


def _efficientnet_model(
    arch: str,
    inverted_residual_setting: List[MBConvConfig],
    dropout: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    return _efficientnet_model("efficientnet_b0", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)


def efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.1, **kwargs)
    return _efficientnet_model("efficientnet_b1", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)


def efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.1, depth_mult=1.2, **kwargs)
    return _efficientnet_model("efficientnet_b2", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.2, depth_mult=1.4, **kwargs)
    return _efficientnet_model("efficientnet_b3", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)


def efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.4, depth_mult=1.8, **kwargs)
    return _efficientnet_model("efficientnet_b4", inverted_residual_setting, 0.4, pretrained, progress, **kwargs)


def efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.6, depth_mult=2.2, **kwargs)
    return _efficientnet_model("efficientnet_b5", inverted_residual_setting, 0.4, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)


def efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.8, depth_mult=2.6, **kwargs)
    return _efficientnet_model("efficientnet_b6", inverted_residual_setting, 0.5, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)


def efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=2.0, depth_mult=3.1, **kwargs)
    return _efficientnet_model("efficientnet_b7", inverted_residual_setting, 0.5, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)
