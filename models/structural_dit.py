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

    def forward(self, x, t):
        x = x.permute(0, 2, 1)