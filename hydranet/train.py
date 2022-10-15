from hydranet.models.hydranet_model import Hydranet


def train(model, use_cuda: bool = True) -> None:
    model.train()
    return


if __name__ == "__main__":
    net = Hydranet()
    train(net)
