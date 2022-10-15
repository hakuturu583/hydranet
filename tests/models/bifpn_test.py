from hydranet.models.regnet_model import regnet_y_400mf, RegNet_Y_400MF_Weights
from hydranet.models.bifpn_model import BiFPN
from hydranet.models.util import getChannels
import torch


def test_print():
    backbone_output_shape = regnet_y_400mf(
        weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1
    ).get_output_shapes()
    net = BiFPN(
        in_channels=getChannels(backbone_output_shape), out_channels=88, num_outs=5
    )
    print(net)


def test_get_output_shapes():
    backbone_output_shape = regnet_y_400mf(
        weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1
    ).get_output_shapes()
    net = BiFPN(
        in_channels=getChannels(backbone_output_shape), out_channels=88, num_outs=5
    )
    shapes = net.get_output_shapes()
    assert len(shapes) == 5
    for index, shape in enumerate(shapes):
        if index == 0:
            assert shape == torch.Size([1, 88, 112, 112])
        elif index == 1:
            assert shape == torch.Size([1, 88, 56, 56])
        elif index == 2:
            assert shape == torch.Size([1, 88, 28, 28])
        elif index == 3:
            assert shape == torch.Size([1, 88, 14, 14])
        elif index == 4:
            assert shape == torch.Size([1, 88, 7, 7])


def test_onnx():
    backbone_output_shape = regnet_y_400mf(
        weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1
    ).get_output_shapes()
    net = BiFPN(
        in_channels=getChannels(backbone_output_shape), out_channels=88, num_outs=5
    )
    net.to_onnx("bifpn.onnx")
