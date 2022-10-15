import pytest
from hydranet.models.regnet import regnet_y_400mf


def test_print():
    net = regnet_y_400mf()
    print(net)


def test_get_output_shapes():
    net = regnet_y_400mf()
    net.get_output_shapes()


def test_onnx():
    net = regnet_y_400mf()
    net.to_onnx("regnet.onnx")


def test_torchscript():
    net = regnet_y_400mf()
    net.to_torch_script("regnet.pt")
