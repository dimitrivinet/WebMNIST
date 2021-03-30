
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from typing import NamedTuple
from dataclasses import dataclass

import torchvision.transforms as T


# MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
# MNIST.resources = [
#     ("/".join([MIRROR, url.split('/')[-1]]), md5)
#     for url, md5 in MNIST.resources
# ]
# print(MNIST.resources)

train_transform = T.Compose([
    T.RandomRotation(360),
    T.ToTensor(),
])
valid_transform = T.ToTensor()


@dataclass
class MNISTDataset(NamedTuple):
    trainset = MNIST(
        root="./dataset", 
        train=True, 
        download=True, 
        transform=train_transform,
    )
    validset = MNIST(
        root="./dataset", 
        train=False, 
        download=True, 
        transform=valid_transform,
    )


@dataclass
class MNISTLoader(NamedTuple):
    trainloader = DataLoader(MNISTDataset.trainset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True,
    )
    validloader = DataLoader(MNISTDataset.validset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=6, 
        pin_memory=True,
    )