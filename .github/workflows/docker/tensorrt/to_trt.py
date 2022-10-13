from hydranet.models.regnet_head import RegNetHead, regnet_y_400mf
from torch2trt import torch2trt

if __name__ == "__main__":
    net = regnet_y_400mf().eval().cuda()
    model_trt = torch2trt(net, [net.get_dummy_input().cuda()])
    #torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))