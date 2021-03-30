from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from typing import NamedTuple
from tqdm import tqdm
from webmnist.model import Model

import torch
import torch.nn as nn
import torchvision.transforms as T


MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST.resources = [
    ("/".join([MIRROR, url.split("/")[-1]]), md5)
    for url, md5 in MNIST.resources
]


train_transform = T.Compose([T.RandomRotation(360), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()])
valid_transform = T.ToTensor()

trainset = MNIST("./dataset", train=True, download=True, transform=train_transform)
validset = MNIST("./dataset", train=False, download=True, transform=valid_transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
validloader = DataLoader(validset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

model = Model()
criterion = nn.CrossEntropyLoss()
optim = AdamW(model.parameters(), lr=1e-3)

best_acc = 0.0
acc= 0.0

for epoch in tqdm(range(15), desc="Epoch"):
    model.train()
    with tqdm(trainloader, desc="Train") as pbar:
        total_loss = 0.0
        acc = 0.0
        for img, label in pbar:
            optim.zero_grad()

            total_loss += loss.item() / len(trainloader)
            acc += (torch.argmax(output, dim=1) == label).sum().item() / len(trainset)
            pbar.set_postfix(loss=total_loss, acc=f"{acc * 100:.2f}%")

    model.eval()
    with tqdm(validloader, desc="Valid") as pbar:
        total_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            for img, label in pbar:

                output = model(img)
                loss = criterion(output, label)
                
                total_loss += loss.item() / len(validloader)
                acc += (torch.argmax(output, dim=1) == label).sum().item() / len(validset)
                pbar.set_postfix(loss=total_loss, acc=f"{acc * 100:.2f}%")

    if acc > best_acc:
        torch.save(model.state_dict(), f"./trained_models/best.pth")
    torch.save(model.state_dict(), f"./trained_models/mnist_{epoch+1:02d}.pth")