import torch.nn as nn
import torch.nn.functional as F
import torch

class NeRF(nn.Module):
    def __init__(self, hidden_dim=128, L_embed=6):
        super().__init__()
        in_dim = 3 + 3 * 2 * L_embed
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 4)
    
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(torch.cat([out, x], dim=-1)))
        out = self.layer4(out)
        return out
