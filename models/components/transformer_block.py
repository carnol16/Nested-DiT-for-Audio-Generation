import torch
import torch.nn as nn
from .adaln import AdaLNZero
from .timestep_embedding import TimestepEmbedding

class DiTBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.adaln = AdaLNZero(dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim),
        )
    
    def forward(self, x, t_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, t_emb) #get all 6 adaLN vectors

        #self-attention sublayer
        x_normed = (1 +scale_msa) * self.norm1(x) + shift_msa
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x = x + gate_msa * attn_out

        #MLP sublayer
        x_normed = (1+scale_mlp) * self.norm2(x) + shift_mlp
        x = x + gate_mlp * self.mlp(x_normed)

        return x