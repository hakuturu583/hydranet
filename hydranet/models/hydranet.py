#!/usr/bin/python
# -*- coding: utf-8 -*-

from bifpn import BiFPN
from regnet import regnet_y_400mf, RegNet_Y_400MF_Weights
from torch import nn, Tensor


class Hydranet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        backbone_output_shape = self.backbone.get_output_shapes()
        self.neck = nn.Sequential()
        for index, shape in enumerate(backbone_output_shape):
        #    print(shape)
            self.neck.add_module("bifpn" + str(index), BiFPN(shape[1]))

    def get_dummy_input(self):
        return self.backbone.get_dummy_input()

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        for bifpn in self.neck:
            print("num_channels : " + str(bifpn.num_channels))
            print(bifpn)
            features = bifpn(features)
        return features
        #return self.neck(features)

if __name__ == "__main__":
    net = Hydranet()
    net.forward(net.get_dummy_input())
    #print(net)
