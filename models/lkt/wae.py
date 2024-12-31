import torch
import torch.nn as nn

class WAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super(WAE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

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
        return z, self.decode(z), mean, logvar


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return self.fc2(h)

def wae_loss(recon_x, x, discriminator, z, lambda_reg=10):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    # recon_loss = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    w_loss = torch.mean(discriminator(z)) - torch.mean(discriminator(torch.randn_like(z)))
    
    total_loss = recon_loss + lambda_reg * w_loss
    return total_loss