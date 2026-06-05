import torch.nn as nn
import torch


class CoarseProjector(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.output_proj = nn.Linear(in_channels, hidden_dim)

    def forward(self, x): #x = [B,C,T]
        x = x.permute(0,2,1)# x = [B, T, C]
        
        x = self.output_proj(x)
        return x
    

if __name__ == "__main__":
    proj = CoarseProjector(in_channels=1024, hidden_dim=384)
    x = torch.randn(2, 1024, 58)
    print(proj(x).shape)  # expect torch.Size([2, 58, 384])
