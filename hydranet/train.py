from hydranet.models.hydranet_model import Hydranet
from torchvision.datasets.kitti import Kitti


def train(model, use_cuda: bool = True) -> None:
    model.train()
    loader = Kitti(root="kitti", download=True)
    #for batch_idx, (data, target) in enumerate(loader):
    #    print(target)
    return


if __name__ == "__main__":
    net = Hydranet()
    train(net)
