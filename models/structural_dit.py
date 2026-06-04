#structure
#INPUT: Noisy coarse latent [B, C, T_Coarse]
#project channels to hidden dim  (Linear on C)
#Stack of 6 DiTBlocks
#Final LayerNorm
#Project hidden dim back to C (Linear)
#Output: predicted noise [B, C, T_Coarse] (Same shape as input)


import torch
import torch.nn as nn
from .components.timestep_embedding import TimestepEmbedding
from .components.transformer_block import DiTBlock

class StructuralDiT(nn.Module):
    def __init__(self, in_channels, hidden_dim, depth, num_heads):
        super().__init__()
        self.t_emb = TimestepEmbedding(hidden_dim)
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_channels)

    def forward(self, x, t): # X = [B,C,T_coarse]  t = [B]
        x = x.permute(0, 2, 1) #[B, T_coarse]

        #Project input and timestep embrddings
        x = self.input_proj(x)
        t_emb = self.t_emb(t)

        #Pass through all DiTBlocks
        for block in self.blocks:
            x = block(x, t_emb)

        #final norm and output projections
        x = self.norm(x)
        x = self.output_proj(x) # [V, T_coarse, C]

        x = x.permute(0, 2, 1) #[B, T_coarse]
        return x


