import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the Wide and Deep Network
class WideAndDeep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 128, 64]):
        super(WideAndDeep, self).__init__()
        # Wide part
        self.wide = nn.Linear(input_dim, output_dim)

        # Deep part
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        self.deep = nn.Sequential(*layers)

        self.deep_out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        deep_out = self.deep_out(deep_out)

        return wide_out + deep_out