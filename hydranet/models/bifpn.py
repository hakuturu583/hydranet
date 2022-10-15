#!/usr/bin/python
# -*- coding: utf-8 -*-

# This code is based on https://github.com/toandaominh1997/EfficientDet.Pytorch/blob/master/models/bifpn.py

# Copyright 2020 toandaominh1997. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os
import argparse
from typing import List
import torch.nn as nn
import torch.nn.functional as F


from module import ConvModule, xavier_init
import torch


class BiFPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        stack=1,
        add_extra_convs=False,
        extra_convs_on_inputs=True,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
    ):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(
                BiFPNModule(
                    channels=out_channels,
                    levels=self.backbone_end_level - self.start_level,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(
        self,
        channels,
        levels,
        init=0.5,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
        eps=0.0001,
    ):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.eps = eps
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()
        for jj in range(2):
            for i in range(self.levels - 1):  # 1,2,3
                fpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False,
                    )
                )
                self.bifpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps  # normalize
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.eps  # normalize
        # build top-down
        idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = (
                w1[0, i - 1] * pathtd[i - 1]
                + w1[1, i - 1]
                * F.interpolate(pathtd[i], scale_factor=2, mode="nearest")
            ) / (w1[0, i - 1] + w1[1, i - 1] + self.eps)
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1
        # build down-top
        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = (
                w2[0, i] * pathtd[i + 1]
                + w2[1, i] * F.max_pool2d(pathtd[i], kernel_size=2)
                + w2[2, i] * inputs_clone[i + 1]
            ) / (w2[0, i] + w2[1, i] + w2[2, i] + self.eps)
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1

        pathtd[levels - 1] = (
            w1[0, levels - 1] * pathtd[levels - 1]
            + w1[1, levels - 1] * F.max_pool2d(pathtd[levels - 2], kernel_size=2)
        ) / (w1[0, levels - 1] + w1[1, levels - 1] + self.eps)
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd

    def get_output_shapes(self, eval: bool = True) -> List[torch.Size]:
        if eval:
            self.eval()
        shapes = []
        # for output in self.forward(self.get_dummy_input()):
        #    shapes.append(output.shape)
        return shapes

    def to_onnx(
        self,
        filename=os.path.dirname(__file__) + "/../onnx/regnet.onnx",
        eval: bool = True,
    ) -> None:
        if eval:
            self.eval()
        torch.onnx.export(self, self.get_dummy_input(), filename, verbose=True)

    def to_torch_script(
        self,
        filename=os.path.dirname(__file__) + "/../onnx/regnet.pt",
        eval: bool = True,
    ) -> None:
        if eval:
            self.eval()
        torch.jit.trace(self, self.get_dummy_input()).save(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Python script for RegNet model")
    parser.add_argument(
        "cmd", choices=["print", "print_output_shapes", "onnx", "torchscript"]
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file path",
        default=os.path.dirname(__file__) + "/../onnx/regnet.onnx",
    )
    parser.parse_args()
    args = parser.parse_args()

    from regnet import regnet_y_400mf, RegNet_Y_400MF_Weights
    from util import getChannels

    backbone_output_shape = regnet_y_400mf(
        weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1
    ).get_output_shapes()
    net = BiFPN(
        in_channels=getChannels(backbone_output_shape), out_channels=88, num_outs=5
    )
    if args.cmd == "print":
        print(net)
    elif args.cmd == "print_output_shapes":
        net.get_output_shapes()
        # for shape in net.get_output_shapes():
        #    print(shape)
    elif args.cmd == "onnx":
        net.to_onnx(args.output)
    elif args.cmd == "torchscript":
        net.to_torch_script(args.output)
