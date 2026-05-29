import torch
import torch.nn as nn
import math

class AdaLNZero(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False) #disables the built in scale/shift
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        ) # Two linear layers with a SiLU activation between them. Output is 6 * dim since we need 6 seperate vectors

        #zero our the last linear layer's weights and bias
        # self.mlp[-1] accesses the last item in sequential - the Linear(dim, dim*6) layer
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x, t_emb):
        #x shape = [B,T,dim]: Topken Sequence
        #t_emb shape = [B, dim]: timestep embedding from TimestepEmbedding class

        params = self.mlp(t_emb) #[B, 6*dim]
        params = params.unsqueeze(1) #[B, 1, 6*dim] which is broadcasted over T : unsqueeze(1) adds a sequence dim. so the params broadcast over every token in x

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1) #splits the 6*dim into 6 equal pieces

        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp