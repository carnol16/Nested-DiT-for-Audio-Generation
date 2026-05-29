from models.components.adaln import AdaLNZero
from models.components.timestep_embedding import TimestepEmbedding
import torch

dim = 256
ada = AdaLNZero(dim)
t_emb = TimestepEmbedding(dim)(torch.tensor([0, 1, 2, 3]))
x = torch.randn(4, 16, dim)

outputs = ada(x, t_emb)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = outputs
print(shift_msa.shape)
print(gate_msa.abs().mean())
