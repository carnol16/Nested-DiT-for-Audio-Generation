from training.noise_schedule import NoiseSchedule
import torch

sched = NoiseSchedule()
print(sched.alpha_bar[0].item())    # near 1.0  (almost no noise)
print(sched.alpha_bar[999].item())  # near 0.0  (almost pure noise)

x0 = torch.randn(2, 1024, 58)
t  = torch.tensor([0, 999])
xt, noise = sched.q_sample(x0, t)
print(xt.shape)   # [2, 1024, 58]
