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
        self.fc = nn.Linear(63, 64, bias=False)

    def forward(self, x):
        return self.fc(x)


class ExamplePassviePartyLocalModuleB(nn.Module):
    def __init__(self):
        super(ExamplePassviePartyLocalModuleB, self).__init__()
        self.fc = nn.Linear(16, 64, bias=False)

    def forward(self, x):
        return self.fc(x)


class ExampleActivePartyLocalModule(nn.Module):
    def __init__(self):
        super(ExampleActivePartyLocalModule, self).__init__()
        self.fc = nn.Linear(27, 64, bias=True)

    def forward(self, x):
        return self.fc(x)


class ExampleGlobalModule(nn.Module):
    def __init__(self):
        super(ExampleGlobalModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = F.relu(x)
        out = self.fc(out)
        return out
    
class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.GRU(input_size = 1, hidden_size = 16, num_layers = 1, batch_first = True, dropout = 0)
        self.fc = nn.Linear(in_features = 16, out_features = 5)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:,-1,:])
        return x