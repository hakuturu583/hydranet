from hydranet.models.regnet_model import regnet_y_400mf
from hydranet.models.bifpn_model import BiFPN
from hydranet.models.util import getChannels
import torch


def test_print():
    net = regnet_y_400mf()
    print(net)


def test_get_output_shapes():
    net = regnet_y_400mf()
    shapes = net.get_output_shapes()
    assert len(shapes) == 5
    for index, shape in enumerate(shapes):
        if index == 0:
            assert shape == torch.Size([1, 32, 112, 112])
        elif index == 1:
            assert shape == torch.Size([1, 48, 56, 56])
        elif index == 2:
            assert shape == torch.Size([1, 104, 28, 28])
        elif index == 3:
            assert shape == torch.Size([1, 208, 14, 14])
        elif index == 4:
            assert shape == torch.Size([1, 440, 7, 7])


def test_onnx():
    net = regnet_y_400mf()
    net.to_onnx("regnet.onnx")
