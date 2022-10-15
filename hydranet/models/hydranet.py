#!/usr/bin/python
# -*- coding: utf-8 -*-

from bifpn import BiFPN
from numpy import size
from regnet import regnet_y_400mf, RegNet_Y_400MF_Weights
from util import getChannels
from torch import nn, Tensor


class Hydranet(nn.Module):
    def __init__(
        self,
        multi_scale_features_channels: size = 88,
        num_multi_scale_features: size = 5,
    ) -> None:
        super().__init__()
        self.backbone = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        backbone_output_shape = self.backbone.get_output_shapes()
        self.neck = BiFPN(
            in_channels=getChannels(backbone_output_shape),
            out_channels=multi_scale_features_channels,
            num_outs=num_multi_scale_features,
        )

    def get_dummy_input(self) -> Tensor:
        return self.backbone.get_dummy_input()

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        for feature in features:
            print(feature.shape)
        return self.neck(features)


if __name__ == "__main__":
    net = Hydranet()
    feature = net.forward(net.get_dummy_input())
    # for out in feature:
    # print(out.shape)
