from model import EEGNet, DeepConvNet
from utils import train, eval
from dataloader import BCIDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from pprint import pprint
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import random


def set_seed(seed: int = 890104):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def getParser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", choices=["EEGNet", "DeepConvNet"], default="EEGNet")
    parser.add_argument("--act", choices=["relu", "elu", "leaky_relu"], default="elu")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    return parser


if __name__ == "__main__":
    set_seed(890104)

    g = torch.Generator()
    g.manual_seed(890104)

    parser = getParser()
    args = parser.parse_args()

    if args.mode == "eval" and args.checkpoint is not None:
        if args.model == "EEGNet":
            model = EEGNet(act=args.act)
        elif args.model == "DeepConvNet":
            model = DeepConvNet(act=args.act)
        model.load_state_dict(torch.load(args.checkpoint))
        model = model.cuda()
        pprint(model)

        test_set = BCIDataset(eval=True)
        test_loader = DataLoader(
            test_set,
            batch_size=256,
            worker_init_fn=seed_worker,
            generator=g,
        )

        test_loss, test_acc = eval(
            model=model,
            test_loader=test_loader,
        )

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    else:
        relu_train_acc = []
        relu_test_acc = []

        elu_train_acc = []
        elu_test_acc = []

        leaky_relu_train_acc = []
        leaky_relu_test_acc = []

        for act in ["relu", "elu", "leaky_relu"]:
            if args.model == "EEGNet":
                model = EEGNet(act=act)
            elif args.model == "DeepConvNet":
                model = DeepConvNet(act=act)
            model = model.cuda()

            optimizer = Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
            train_set = BCIDataset()
            test_set = BCIDataset(eval=True)
            train_loader = DataLoader(
                train_set,
                batch_size=256,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=256,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

            best_acc = 0

            for epoch in range(450):
                train_loss, train_acc = train(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                )
                test_loss, test_acc = eval(
                    model=model,
                    test_loader=test_loader,
                )

                if epoch % 50 == 0:
                    print(
                        f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}"
                    )

                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), f"best-{act}-{args.model}.pth")

                if act == "relu":
                    relu_train_acc.append(train_acc)
                    relu_test_acc.append(test_acc)
                elif act == "elu":
                    elu_train_acc.append(train_acc)
                    elu_test_acc.append(test_acc)
                elif act == "leaky_relu":
                    leaky_relu_train_acc.append(train_acc)
                    leaky_relu_test_acc.append(test_acc)

            print(f"Best Accuracy for {act}: {best_acc:.4f}")

        plt.title(f"Activation Function Comparison ({args.model})")
        plt.plot(relu_train_acc, label="relu_train")
        plt.plot(relu_test_acc, label="relu_test")
        plt.plot(elu_train_acc, label="elu_train")
        plt.plot(elu_test_acc, label="elu_test")
        plt.plot(leaky_relu_train_acc, label="leaky_relu_train")
        plt.plot(leaky_relu_test_acc, label="leaky_relu_test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"activation-{args.model}.png")
