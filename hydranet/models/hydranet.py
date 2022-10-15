#!/usr/bin/python
# -*- coding: utf-8 -*-

from turtle import back
from bifpn import BiFPN
from regnet import RegNet, regnet_y_400mf, RegNet_Y_400MF_Weights
from torch import nn


class Hydranet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        backbone_output_shape = self.backbone.get_output_shapes()
        for shape in backbone_output_shape:
            self.neck = nn.Sequential(
                BiFPN(shape[1]), BiFPN(shape[1]), BiFPN(shape[1]), BiFPN(shape[1])
            )


if __name__ == "__main__":
    net = Hydranet()
    # print(net)
