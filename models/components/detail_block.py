import torch
import torch.nn as nn
from .adaln import AdaLNZero
from .cross_attention import CrossAttention




class DetailBlock(nn.Module):
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.adaln = AdaLNZero(dim)
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
    def forward(self, x, context, t_emb):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(x, t_emb) #get all 6 adaLN vectors

        #self-attention sublayer
        x_normed = (1 +scale_msa) * self.norm1(x) + shift_msa
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x = x + gate_msa * attn_out

        attn_out, _ = self.cross_attn(x, context)
        x = x + attn_out


        #MLP sublayer
        x_normed = (1+scale_mlp) * self.norm2(x) + shift_mlp
        x = x + gate_mlp * self.mlp(x_normed)

        return x


if __name__  == "__main__":

    block = DetailBlock(dim=384, num_heads=4)
    x = torch.randn(2, 348, 384)
    ctx = torch.randn(2, 58, 384)
    t_emb = torch.randn(2, 384)
    out = block(x, ctx, t_emb)
    print(out.shape)  # torch.Size([2, 348, 384])
