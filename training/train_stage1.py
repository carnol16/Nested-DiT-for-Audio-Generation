import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from data.dataset import get_dataloader
from models.structural_dit import StructuralDiT
from training.noise_schedule import NoiseSchedule
from training.sampler import ddim_sample

#CONFIG

#directories
LATENT_DIR = "data/latents/test"
STATS_PATH = "data/stats.json"
CKPT_DIR = Path("checkpoints/stage1")
LOG_PATH = Path("logs/stage1.log")

#parameters
IN_CHANNELS = 1024
HIDDEN_DIM = 256
DEPTH = 6
NUM_HEADS = 4
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-2
TOTAL_STEPS = 50_000

#Training
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def train():
    #SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "w")

    loader = get_dataloader(LATENT_DIR, STATS_PATH, BATCH_SIZE)
    schedule = NoiseSchedule()

    #MODEL, OPTIMIZER, SCALER
    model = StructuralDiT(IN_CHANNELS, HIDDEN_DIM, DEPTH, NUM_HEADS).to(device)
    model.gradient_checkpointing_enable() if hasattr(model, 'gradient_checkpointing_enable') else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda')

    if device.type == "cuda":
        print(torch.cuda.memory_summary(device))

    #training loop
    step = 0
    while step < TOTAL_STEPS:
        for batch in loader:
            if step >= TOTAL_STEPS:
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

            #logging every 100 steps
            if step % 100 == 0:
                msg = f"step {step:06d} | loss {loss.item():.4f}"
                print(msg)
                log.write(msg + "\n")
                log.flush()

            # Sample every 5K steps
            if step > 0 and step % 5000 == 0:
                model.eval()
                with torch.inference_mode():
                    sample = ddim_sample(model, schedule,
                                         shape = (1, IN_CHANNELS,58),
                                         num_steps=50, device=device)
                torch.save(sample, f"checkpoints/stage1/sample_step{step}.pt")
                model.train()

            #Checkpoint every 10K steps
            if step > 0 and step % 10000 == 0:
                torch.save(model.state_dict(),
                           CKPT_DIR / f"ckpt_step{step}.pt")
            
            step += 1

    torch.save(model.state_dict(), CKPT_DIR / "ckpt_final.pt")
    log.close()
    print("Training complete.")

if __name__ == "__main__":
    train()   
    
