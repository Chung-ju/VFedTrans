import torch.nn as nn
from torch.nn import functional as F

class AE(nn.Module):
    def __init__(self, original_dim: int, latent_dim: int, activation: str='sigmoid') -> None:
        super(AE, self).__init__()
        
        self.activation_dict = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU
        }
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        self.encoder.add_module('FC1', nn.Linear(original_dim, 256))
        self.encoder.add_module('AF1', self.activation_dict[activation]())
        self.encoder.add_module('FC2', nn.Linear(256, 64))
        self.encoder.add_module('AF2', self.activation_dict[activation]())
        self.encoder.add_module('FC3', nn.Linear(64, latent_dim))
        self.encoder.add_module('AF3', self.activation_dict[activation]())
        
        self.decoder.add_module('FC1', nn.Linear(latent_dim, 64))
        self.decoder.add_module('AF1', self.activation_dict[activation]())
        self.decoder.add_module('FC2', nn.Linear(64, 256))
        self.decoder.add_module('AF2', self.activation_dict[activation]())
        self.decoder.add_module('FC3', nn.Linear(256, original_dim))
        self.decoder.add_module('AF3', self.activation_dict[activation]())
        
    def encode(self, X):
        return self.encoder(X)
    
    def decode(self, X):
        return self.decoder(X)
    
    def forward(self, X):
        z = self.encode(X)
        
        return z, self.decode(z)