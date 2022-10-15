from numpy import size
from torch import Tensor
from typing import List


def getChannels(x: Tensor) -> List[size]:
    channels = []
    for val in x:
        channels.append(val[1])
    return channels
