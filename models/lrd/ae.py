import torch.nn as nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 latent_dim,
                 depth: int=6,
                 activation_name: str='sigmoid',
                 **kwargs) -> None:
        super(AutoEncoder, self).__init__()
        
        self.activation_name = activation_name
        self.activation_functions = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU
        }
        
        self.activation_forward = {
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu
        }

        n_units = []
        step = int((latent_dim - in_dim) / depth) * 2
        for i in range(int(depth / 2) + 1):
            layer_unit = in_dim + step * i if i != int(depth / 2) else latent_dim
            n_units.append(layer_unit)
        n_units += n_units[-2::-1]
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        for i in range(1, depth + 1):
            if i < depth / 2:
                self.encoder.add_module('Layer ' + str(i) + ': net', nn.Linear(n_units[i - 1], n_units[i]))
                self.encoder.add_module('Layer ' + str(i) + ': activation', self.activation_functions[activation_name]())
            elif i == int(depth / 2):
                self.encoder.add_module('Layer ' + str(i) + ': net', nn.Linear(n_units[i - 1], n_units[i]))
            elif i < depth:
                self.decoder.add_module('Layer ' + str(i) + ': net', nn.Linear(n_units[i - 1], n_units[i]))
                self.decoder.add_module('Layer ' + str(i) + ': activation', self.activation_functions[activation_name]())
            else:
                self.decoder.add_module('Layer ' + str(i) + ': net', nn.Linear(n_units[i - 1], n_units[i]))

    def forward(self, X):
        encoded = self.activation_forward[self.activation_name](self.encoder(X))
        decoded = self.activation_forward[self.activation_name](self.decoder(encoded))
        
        return encoded, decoded
    
    def to_np(self, X):
        return X.data.cpu().numpy()

    def generate(self, X):
        return self.to_np(self.forward(X)[1])
    
    @property
    def name(self):
        return 'AE'