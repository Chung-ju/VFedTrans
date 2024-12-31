import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mean = self.fc2_mean(h1)
        logvar = self.fc2_logvar(h1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return z, recon_x, mean, logvar

def vae_loss(recon_x, x, mean, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    # recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    return recon_loss + KL_divergence