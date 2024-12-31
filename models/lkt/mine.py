import torch
import torch.nn as nn
import torch.optim as optim

class TNetwork(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(TNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=1)
        return self.fc(z)

def sample_marginal(z, batch_size):
    indices = torch.randperm(batch_size)
    z_shuffled = z[indices]
    return z_shuffled