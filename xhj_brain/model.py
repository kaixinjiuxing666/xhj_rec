from torch import nn

class MLP256(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32),nn.ReLU(),
                                 nn.Linear(32, 128),nn.ReLU(),
                                 nn.Linear(128, 256),nn.ReLU(),
                                 nn.Linear(256, 128),nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),

                                 )
        self.linear = nn.Linear(64, 10)

    def forward(self, X):
        X = self.net(X)
        X = self.linear(X)
        return X


class MLP512(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16),nn.ReLU(),
                                 nn.Linear(16, 64),nn.ReLU(),
                                 nn.Linear(64, 256),nn.ReLU(),
                                 nn.Linear(256, 512),nn.ReLU(),
                                 nn.Linear(512, 256),nn.ReLU(),
                                 nn.Linear(256, 64),nn.ReLU(),
                                 )
        self.linear = nn.Linear(64, 10)
    def forward(self, X):
        X = self.net(X)
        X = self.linear(X)
        return X
