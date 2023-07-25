import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy


def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data = data.type(torch.float32).to("cuda")
        target = target.type(torch.long).to("cuda")

        output = model(data)
        loss = cross_entropy(output, target)
        train_loss += loss.item()
        target = target.type(torch.long)
        pred = output.argmax(dim=1, keepdim=True).type(torch.long)
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return train_loss, accuracy


@torch.no_grad()
def eval(
    model: nn.Module,
    test_loader: DataLoader,
):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.type(torch.float32).to("cuda")
            target = target.type(torch.long).to("cuda")

            output = model(data)
            loss = cross_entropy(output, target).item()
            target = target.type(torch.long)
            pred = output.argmax(dim=1, keepdim=True).type(torch.long)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += loss
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy
