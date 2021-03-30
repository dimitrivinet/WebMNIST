
from torch.optim import AdamW
from tqdm import tqdm
from webmnist.model import Model, LeNet5
from webmnist.data import MNISTDataset, MNISTLoader

import torch
import torch.nn as nn


def train(path: str, save_all: bool, epochs: int = 3,) -> None:
    dataset = MNISTDataset()
    loader = MNISTLoader()

    # model = Model()
    model = LeNet5(10)
    criterion = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=1e-3)

    best_acc = 0.0
    acc= 0.0

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        with tqdm(loader.trainloader, desc="Train") as pbar:
            total_loss = 0.0
            acc = 0.0
            for img, label in pbar:
                optim.zero_grad()

                output = model(img)
                loss = criterion(output, label)
                loss.backward()
                optim.step()

                total_loss += loss.item() / len(loader.trainloader)
                acc += (
                    torch.argmax(output, dim=1) == label
                ).sum().item() / len(dataset.trainset)
                pbar.set_postfix(loss=total_loss, acc=f"{acc * 100:.2f}%")

        model.eval()
        with tqdm(loader.validloader, desc="Valid") as pbar:
            total_loss = 0.0
            acc = 0.0
            with torch.no_grad():
                for img, label in pbar:

                    output = model(img)
                    loss = criterion(output, label)
                    
                    total_loss += loss.item() / len(loader.validloader)
                    acc += (
                        torch.argmax(output, dim=1) == label
                    ).sum().item() / len(dataset.validset)
                    pbar.set_postfix(loss=total_loss, acc=f"{acc * 100:.2f}%")

        if acc > best_acc:
            torch.save(model.state_dict(), f"{path}/best.pt")
            tqdm.write("saved best")
            best_acc = acc

        if save_all:
            torch.save(
                model.state_dict(), f"{path}/mnist_{epoch+1:02d}.pt"
            )