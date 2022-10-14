from hydranet.models.regnet import RegNet, regnet_y_400mf
from torchvision.models.regnet import RegNet_Y_400MF_Weights
from torch2trt import torch2trt
import torch
import tensorrt
import os

if __name__ == "__main__":
    net = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1).eval().cuda()
    model_trt = torch2trt(
        net,
        [net.get_dummy_input().cuda()],
        fp16_mode=True,
        log_level=tensorrt.Logger.INFO,
    )
    torch.save(
        model_trt.state_dict(),
        os.path.join(os.path.dirname(__file__), "regnet_trt.pth"),
    )
    engine_file = os.path.join(os.path.dirname(__file__), "regnet.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())
