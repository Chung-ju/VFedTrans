import torch
import torch.nn as nn
import torch.nn.functional as F

class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=32, n_a=32, n_steps=3):
        super(TabNet, self).__init__()
        
        # Input dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d                # Number of decision features
        self.n_a = n_a                # Number of attention features
        self.n_steps = n_steps        # Number of decision steps
        
        self.input_fc = nn.Linear(input_dim, n_d)
        
        # Construct feature selectors
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_d, n_a),
                nn.ReLU(),
                nn.Linear(n_a, 1)
            ) for _ in range(n_steps)
        ])
        
        # Construct decision layers
        self.decision_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_d, n_d),
                nn.ReLU(),
                nn.Linear(n_d, n_d)
            ) for _ in range(n_steps)
        ])
        
        self.output_fc = nn.Linear(n_d, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        attentions = []
        decisions = []
        
        for step in range(self.n_steps):
            attention = self.attention_layers[step](x)
            attention = torch.softmax(attention, dim=1)
            attentions.append(attention)
            
            decision = self.decision_layers[step](x)
            decisions.append(decision)
            
            x = x * attention
        
        x = torch.mean(torch.stack(decisions, dim=0), dim=0)
        
        out = self.output_fc(x)
        
        return out
