import torch
import torch.nn as nn

class DomainLearner(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DomainLearner, self).__init__()
        self.domain_diff = nn.Parameter(torch.rand(size=(input_dim, latent_dim)))
    
    def forward(self, enc, X_fed):
        z = torch.matmul(torch.softmax(torch.matmul(enc, torch.transpose(torch.matmul(X_fed, self.domain_diff), 0, 1)) / enc.shape[1], dim=1), torch.matmul(X_fed, self.domain_diff))
        return z