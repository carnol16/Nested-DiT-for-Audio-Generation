import torch
import json
from pathlib import Path
import math

ROOT = Path(__file__).parent.parent

latent_files = list(Path(ROOT / "data/latents/test").glob("*.pt"))

total_sum = 0
total_sum_sq = 0
total_count = 0
global_min = float('inf')
global_max = float('-inf')

for pt_path in latent_files:
    tensor = torch.load(pt_path).flatten()

    total_sum    += tensor.sum()
    total_sum_sq += (tensor ** 2).sum()
    total_count  += tensor.numel()
    global_min    = min(global_min, tensor.min())
    global_max    = max(global_max, tensor.max())


mean = total_sum / total_count
std  = math.sqrt(total_sum_sq / total_count - mean ** 2)

stats = {
    "mean": mean.item(),
    "std": std,
    "min": global_min.item(),
    "max": global_max.item(),
}

out_path = ROOT / "data" / "stats.json"
with open(out_path, "w") as f:
    json.dump(stats, f, indent=2)

print(stats)

