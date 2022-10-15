from hydranet.models.hydranet_model import Hydranet
from torchvision.datasets.kitti import Kitti
from torchvision.transforms import PILToTensor
import torch


def train(model, use_cuda: bool = True) -> None:
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using {} device".format(device))
    model.train()
    loader = Kitti(root="kitti", download=True)
    trans = PILToTensor()
    for batch_idx, (data, target) in enumerate(loader):
        data = trans(data).to(device)
        for annotation in target:
            print(annotation['type'])
        #data = data.to(device)
        #target = target.to(device)
    return


if __name__ == "__main__":
    net = Hydranet()
    train(net)
