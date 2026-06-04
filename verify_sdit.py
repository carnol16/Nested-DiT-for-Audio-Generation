from models.structural_dit import StructuralDiT
import torch

model = StructuralDiT(in_channels=12, hidden_dim=256, depth=6, num_heads=4)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total/1e6:.1f}M")   # target: under 50M

x = torch.randn(2, 12, 50)   # batch=2, 12 channels, 50 time frames
t = torch.tensor([100, 500])

out = model(x, t)
print(out.shape)              # [2, 12, 50]
