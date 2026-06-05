import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from data.dataset import get_dataloader
from models.structural_dit import StructuralDiT
from training.noise_schedule import NoiseSchedule

LATENT_DIR = "data/latents/test"
STATS_PATH = "data/stats.json"
IN_CHANNELS = 1024
HIDDEN_DIM = 256
DEPTH = 6
NUM_HEADS = 4

def test_batch(batch_size):
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    try:
        loader = get_dataloader(LATENT_DIR, STATS_PATH, batch_size)
        model = StructuralDiT(IN_CHANNELS, HIDDEN_DIM, DEPTH, NUM_HEADS).to(device)
        schedule = NoiseSchedule()
        scaler = GradScaler('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for i, batch in enumerate(loader):
            if i >= 3:
                break
            x0 = batch.to(device)
            t = torch.randint(0, schedule.T, (x0.shape[0],), device=device)
            xt, noise = schedule.q_sample(x0, t)
            with autocast('cuda'):
                pred = model(xt, t)
                loss = nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"batch_size={batch_size:3d}  peak VRAM={peak:.2f} GB  OK")
    except torch.cuda.OutOfMemoryError:
        print(f"batch_size={batch_size:3d}  OOM")

if __name__ == "__main__":
    for bs in [4, 8, 16, 32]:
        test_batch(bs)
