from models.structural_dit import StructuralDiT
from training.noise_schedule import NoiseSchedule
from training.sampler import ddim_sample
import torch

model = StructuralDiT(in_channels=1024, hidden_dim=256, depth=6, num_heads=4)
schedule = NoiseSchedule()

out = ddim_sample(model, schedule, shape=(1, 1024, 58), num_steps=50)
print(out.shape)    # [1, 1024, 58]
print(torch.isnan(out).any().item())   # False
