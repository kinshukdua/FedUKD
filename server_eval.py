import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Callable, List, Optional, Tuple
import numpy as np 
from model import UNet
from dataloader import segDataset
import torchvision.transforms as transforms
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import OrderedDict
device = torch.device("cpu")
color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

t = transforms.Compose([color_shift, blurriness])
dataset2 = segDataset("data/CityScape-Dataset-Unified/Test", "data/CityScape-Dataset-Unified", training = False, transform= t)
n_classes = len(dataset2.bin_classes)+1

def acc(y, pred_mask):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(
        y.cpu()
    )
    return seg_acc

model = UNet(n_channels=3, n_classes=n_classes, bilinear=True, channel_depth=16).to(device)


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    it1 = model.state_dict().items()
    it2 = state_dict.items()

    l1 = len(it1)
    l2 = len(it2)

    if l1 != l2:
        print(f"{l1} : {l2} length do not match")
    else:
        for i in model.state_dict():
            if not model.state_dict()[i].shape == state_dict[i].shape:
                print(
                    i,
                    model.state_dict()[i].shape,
                    state_dict[i].shape,
                    "Different",
                )

def evaluate(weights):
    print("Evaluation started on Server")
    set_parameters(model, weights)
    model.eval()
    val_loss_list = []
    val_acc_list = []
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_i, (x, y) in enumerate(dataset2):
        with torch.no_grad():
            pred_mask = model(x.unsqueeze(0).to(device))
        val_loss = criterion(pred_mask, y.unsqueeze(0).to(device))
        val_loss_list.append(val_loss.cpu().detach().numpy())
        val_acc_list.append(acc(y, pred_mask).numpy())

    print(
        " val loss : {:.5f} - val acc : {:.2f}".format(
            np.mean(val_loss_list), np.mean(val_acc_list)
        )
    )
    return (
        np.mean(val_loss_list).item(),
        {"accuracy": np.mean(val_acc_list).item()},
    )

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,  # Minimum number of clients to be sampled for the next round
    min_available_clients=3,  # Minimum number of clients that need to be connected to the server before a training round can start
    eval_fn = evaluate
)


fl.server.start_server(
    server_address="localhost:5000",
    config={"num_rounds": 3},
    strategy=strategy,
)
