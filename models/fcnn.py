import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 16))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(16, output_dim))
        layers.append(nn.Softmax(dim=1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)