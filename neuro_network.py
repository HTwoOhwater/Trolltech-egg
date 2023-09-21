import torch


class Simple(torch.nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        return self.fc1(x)


