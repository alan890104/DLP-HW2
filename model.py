from torch import nn, Tensor
from torch.nn import (
    Conv2d,
    Linear,
    BatchNorm2d,
    Dropout,
    AvgPool2d,
    MaxPool2d,
    Sequential,
)
import torch.nn.functional as F
from collections import OrderedDict


# Implement an EEG net
class EEGNet(nn.Module):
    func_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }

    def __init__(self, act="elu", dropout: float = 0.5, **kwargs):
        super(EEGNet, self).__init__()

        act_func = self.func_map[act](**kwargs)
        self.first_conv = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False,
            ),
            BatchNorm2d(
                num_features=16,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
        )
        self.depthwise_conv = Sequential(
            Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False,
            ),
            BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            act_func,
            AvgPool2d(
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=0,
            ),
            Dropout(p=dropout, inplace=True),
        )
        self.separable_conv = Sequential(
            Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False,
            ),
            BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            act_func,
            AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0,
            ),
            Dropout(p=dropout, inplace=True),
        )
        self.classify = Sequential(
            Linear(
                in_features=736,
                out_features=2,
                bias=True,
            ),
        )

    def forward(self, x: Tensor) -> dict:
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x


# Implement a deep convolutional net
class DeepConvNet(nn.Module):
    func_map = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }

    def __init__(self, act: str = "elu", dropout: float = 0.5, **kwargs) -> None:
        super(DeepConvNet, self).__init__()
        act_func = self.func_map[act]()

        channels = [25, 25, 50, 100, 200]
        kernels = [(2, 1), (1, 5), (1, 5), (1, 5)]

        self.convs = nn.ModuleList()
        self.convs.append(Conv2d(1, 25, kernel_size=(1, 5)))

        for i in range(len(channels) - 1):
            conv = Sequential(
                Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernels[i],
                    stride=1,
                    padding=0,
                ),
                BatchNorm2d(num_features=channels[i + 1]),
                act_func,
                MaxPool2d(kernel_size=(1, 2)),
                Dropout(p=dropout),
            )
            self.convs.append(conv)

        self.classify = Linear(in_features=8600, out_features=2)

    def forward(self, x: Tensor):
        for conv in self.convs:
            x = conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x
