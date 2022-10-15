#!/usr/bin/python
# -*- coding: utf-8 -*-

from bifpn import BiFPN
from regnet import regnet_y_400mf, RegNet_Y_400MF_Weights
from util import getChannels
from torch import nn, Tensor


class Hydranet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        backbone_output_shape = self.backbone.get_output_shapes()
        print(backbone_output_shape)
        self.neck = BiFPN(
            in_channels=getChannels(backbone_output_shape), out_channels=88, num_outs=5
        )

    def get_dummy_input(self):
        return self.backbone.get_dummy_input()

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        # return features
        return self.neck(features)


if __name__ == "__main__":
    net = Hydranet()
    feature = net.forward(net.get_dummy_input())
    for out in feature:
        print(out.shape)
