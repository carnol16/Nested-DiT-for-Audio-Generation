import torch

def ddim_sample(model, schedule, shape, num_steps=50, device="cpu"):
    # model: trained StructuralDiT
    # schedule: NoiseSchedule instance
    # shape: tuple, ex: (1, 1024, 58)
    #return: Denoised latent [*shape]

    #Pick evenly spaced timestamps (going backwards)
    step_indices = torch.linspace(0, schedule.T - 1, num_steps + 1).long()
    timesteps = step_indices.flip(0) # [num_steps+1], from T-1 down to 0

    #start with pure noise
    x = torch.randn(shape, device=device)

    #denoise loop: for each consecutive pair (t_now, t_prev) in timesteps
    for i in range(num_steps):
        t_now = timesteps[i].item()
        t_prev = timesteps[i + 1].item()

        ab_now = schedule.alpha_bar[t_now].to(device)
        ab_prev = schedule.alpha_bar[t_prev].to(device)

        t_batch = torch.full((shape[0],), t_now, dtype=torch.long, device=device)
        pred_noise = model(x, t_batch)

        #DDIM update rule
        x0_pred = (x - (1 - ab_now).sqrt() * pred_noise) / ab_now.sqrt()
        x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * pred_noise

        return x