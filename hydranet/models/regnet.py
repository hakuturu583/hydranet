#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import partial
from typing import Callable, Optional, Any

from torchvision.models.regnet import (
    BlockParams,
    WeightsEnum,
    regnet_y_400mf,
    RegNet_Y_400MF_Weights,
)
from torchvision.models.regnet import RegNet

from torchvision.models._utils import _ovewrite_named_param

from torch import nn, Tensor
import torch.onnx

import argparse
import os


class RegNet(RegNet):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block_params=block_params,
            num_classes=num_classes,
            stem_width=stem_width,
            stem_type=stem_type,
            block_type=block_type,
            norm_layer=norm_layer,
            activation=activation,
        )

    def get_output_shapes(self, eval: bool = True):
        if eval:
            self.eval()
        shapes = []
        for output in self.forward(self.get_dummy_input()):
            shapes.append(output.shape)
        return shapes

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        output = [x]
        for layer in self.trunk_output:
            x = layer(x)
            output.append(x)
        return output

    def get_dummy_input(self):
        return torch.randn((1, 3, 224, 224))

    def to_onnx(
        self,
        filename=os.path.dirname(__file__) + "/../onnx/regnet.onnx",
        eval: bool = True,
    ):
        if eval:
            self.eval()
        torch.onnx.export(self, self.get_dummy_input(), filename, verbose=True)

    def to_torch_script(
        self,
        filename=os.path.dirname(__file__) + "/../onnx/regnet.pt",
        eval: bool = True,
    ):
        if eval:
            self.eval()
        torch.jit.trace(self, self.get_dummy_input()).save(filename)


def regnet_y_400mf(
    *,
    weights: Optional[RegNet_Y_400MF_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.
    Args:
        weights (:class:`~torchvision.models.RegNet_Y_400MF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_400MF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.
    .. autoclass:: torchvision.models.RegNet_Y_400MF_Weights
        :members:
    """
    weights = RegNet_Y_400MF_Weights.verify(weights)

    params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs
    )
    return _regnet(params, weights, progress, **kwargs)


def _regnet(
    block_params: BlockParams,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> RegNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    norm_layer = kwargs.pop(
        "norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1)
    )
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


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
    net = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
    if args.cmd == "print":
        # print(dir(net))
        for block in net.trunk_output:
            print(block)
        # print(net)
    elif args.cmd == "print_output_shapes":
        for shape in net.get_output_shapes():
            print(shape)
    elif args.cmd == "onnx":
        net.to_onnx(args.output)
    elif args.cmd == "torchscript":
        net.to_torch_script(args.output)
