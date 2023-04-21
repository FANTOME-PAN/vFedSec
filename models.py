import torch.nn.functional as F
from torch import nn


def generate_passive_party_local_module(party_type: str) -> nn.Module:
    return {
        'A': ExamplePassviePartyLocalModuleA(),
        'B': ExamplePassviePartyLocalModuleB(),
    }[party_type]


def generate_active_party_local_module() -> nn.Module:
    return ExampleActivePartyLocalModule()


def generate_global_module() -> nn.Module:
    return ExampleGlobalModule()


def get_criterion():
    return nn.BCEWithLogitsLoss()


class ExamplePassviePartyLocalModuleA(nn.Module):
    def __init__(self):
        super(ExamplePassviePartyLocalModuleA, self).__init__()
        self.fc = nn.Linear(3, 64, bias=False)

    def forward(self, x):
        return self.fc(x)


class ExamplePassviePartyLocalModuleB(nn.Module):
    def __init__(self):
        super(ExamplePassviePartyLocalModuleB, self).__init__()
        self.fc = nn.Linear(20, 64, bias=False)

    def forward(self, x):
        return self.fc(x)


class ExampleActivePartyLocalModule(nn.Module):
    def __init__(self):
        super(ExampleActivePartyLocalModule, self).__init__()
        self.fc = nn.Linear(57, 64, bias=True)

    def forward(self, x):
        return self.fc(x)


class ExampleGlobalModule(nn.Module):
    def __init__(self):
        super(ExampleGlobalModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out = F.relu(x)
        out = self.fc(out)
        return out
