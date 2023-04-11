import torch.nn.functional as F
from torch import nn


def generate_passive_party_local_module() -> nn.Module:
    return ExamplePassviePartyLocalModule()


def generate_active_party_local_module() -> nn.Module:
    return ExampleActivePartyLocalModule()


def generate_global_module() -> nn.Module:
    return ExampleGlobalModule()


def get_criterion():
    return nn.BCEWithLogitsLoss()


class ExamplePassviePartyLocalModule(nn.Module):
    def __init__(self):
        super(ExamplePassviePartyLocalModule, self).__init__()
        self.fc = nn.Linear(20, 16, bias=False)

    def forward(self, x):
        return self.fc(x)


class ExampleActivePartyLocalModule(nn.Module):
    def __init__(self):
        super(ExampleActivePartyLocalModule, self).__init__()
        self.fc = nn.Linear(61, 16, bias=True)

    def forward(self, x):
        return self.fc(x)


class ExampleGlobalModule(nn.Module):
    def __init__(self):
        super(ExampleGlobalModule, self).__init__()
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out = F.relu(x)
        out = self.fc(out)
        return out
