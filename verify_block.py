from models.components.transformer_block import DiTBlock
from models.components.timestep_embedding import TimestepEmbedding
import torch

dim, heads = 256, 4
block = DiTBlock(dim, heads)
t_emb = TimestepEmbedding(dim)(torch.tensor([0, 1, 2, 3]))
x = torch.randn(4, 16, dim)

out = block(x, t_emb)
print(out.shape)          # [4, 16, 256]
print((out - x).abs().mean())  # ~0.0 at init (gates are zero)
