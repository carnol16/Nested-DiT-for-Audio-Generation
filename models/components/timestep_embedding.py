# Converts an integer diffusion timestep (0–999) into a 256-dimensional float vector.
# Uses sinusoidal encoding so nearby timesteps produce similar vectors and distant
# timesteps produce different ones — giving the model a sense of "how noisy" the input is.

import torch
import torch.nn as nn
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device = t.device) / half) #build frequency table
        args = t[:, None].float() * freqs[None, :]

        return torch.cat([torch.sin(args), torch.cos(args)], dim= -1)