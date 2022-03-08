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

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,  # Minimum number of clients to be sampled for the next round
    min_available_clients=3,  # Minimum number of clients that need to be connected to the server before a training round can start
)


fl.server.start_server(
    server_address="localhost:5000",
    config={"num_rounds": 10},
    strategy=strategy,
)
