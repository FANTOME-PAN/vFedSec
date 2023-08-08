import torch
import torch.nn.functional as F
from torch import nn


def generate_passive_party_local_module(party_type: str) -> nn.Module:
    return MLPLocal()


def generate_active_party_local_module() -> nn.Module:
    return MLPLocal()


def generate_global_module() -> nn.Module:
    # 47 for EMNIST
    return MLPGlobal(num_classes=10)


def get_criterion():
    return nn.CrossEntropyLoss()


class CNN2Local(nn.Module):

    def __init__(self):
        super(CNN2Local, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=448, out_features=600)
        # Global Module
        # self.fc2 = nn.Linear(in_features=600, out_features=120)
        # self.fc3 = nn.Linear(in_features=120, out_features=num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.flatten(start_dim=1)
        out = F.relu(self.fc1(out))
        return out


class MLPLocal(nn.Module):
    # https://arxiv.org/pdf/2007.06081.pdf, following this paper
    def __init__(self):
        super(MLPLocal, self).__init__()
        self.fc1 = nn.Linear(in_features=7 * 28, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=128)

    def forward(self, x):
        out = F.relu(self.fc1(x.flatten(start_dim=1)))
        out = F.relu(self.fc2(out))
        return out


class MLPGlobal(nn.Module):
    # https://arxiv.org/pdf/2007.06081.pdf, following this paper
    # 4 clients
    # 7 layer of CNN and then fully connected layers
    # since fully connect layer, where to cut won't make any difference

    def __init__(self, num_classes):
        super(MLPGlobal, self).__init__()
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=64)
        self.fc6 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        out = F.relu(self.fc3(x))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        return out


class CNN2Global(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN2Global, self).__init__()
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_classes)

    def forward(self, x):
        out = F.relu(self.fc2(x))
        out = self.fc3(out)
        return out
