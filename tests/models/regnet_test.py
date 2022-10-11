import pytest
from hydranet.models.regnet_head import RegNetHead, regnet_y_400mf


def test_print():
    net = regnet_y_400mf()
    print(net)


def test_onnx():
    net = regnet_y_400mf()
    net.to_onnx("regnet.onnx")
