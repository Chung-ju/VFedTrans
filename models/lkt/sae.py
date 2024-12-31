import torch
import torch.nn as nn

class SparseAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def l1_penalty(var):
    return torch.abs(var).sum()

def loss_function(recon_x, x, encoded, l1_lambda=1e-5):
    MSE_loss = nn.functional.mse_loss(recon_x, x)
    
    l1_loss = l1_penalty(encoded)
    
    return MSE_loss + l1_lambda * l1_loss