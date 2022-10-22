#!/usr/bin/python
# -*- coding: utf-8 -*-

from hydranet.models.retinahead_model import RetinaHead
from hydranet.models.bifpn_model import BiFPN
from hydranet.models.regnet_model import regnet_y_400mf, RegNet_Y_400MF_Weights
from hydranet.models.util import getChannels
from hydranet.models.module import Anchors, ClipBoxes, BBoxTransform
from hydranet.models.losses import FocalLoss
from numpy import size
import torch
from torch import nn, Tensor
from torchvision.datasets.kitti import Kitti
import math


class Hydranet(nn.Module):
    def __init__(
        self,
        multi_scale_features_channels: size = 88,
        num_multi_scale_features: size = 5,
        num_object_classes: size = 5,
        object_detection_threshold=0.01,
        object_detection_iou_threshold=0.5,
    ) -> None:
        super().__init__()
        self.backbone = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        backbone_output_shape = self.backbone.get_output_shapes()
        self.neck = BiFPN(
            in_channels=getChannels(backbone_output_shape),
            out_channels=multi_scale_features_channels,
            num_outs=num_multi_scale_features,
        )
        self.object_detection_head = RetinaHead(
            num_classes=num_object_classes, in_channels=multi_scale_features_channels
        )
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.threshold = object_detection_threshold
        self.iou_threshold = object_detection_iou_threshold
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.freeze_bn()
        self.criterion = FocalLoss()

    def get_dummy_input(self) -> Tensor:
        return self.backbone.get_dummy_input()

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        outs = self.object_detection_head(features[-5:])
        # classification = torch.cat([out for out in outs[0]], dim=1)
        # regression = torch.cat([out for out in outs[1]], dim=1)
        # anchors = self.anchors(inputs)
        return x

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == "__main__":
    net = Hydranet()
    net(net.get_dummy_input())
    # net.train()
