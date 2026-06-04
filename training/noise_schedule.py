import torch

class NoiseSchedule:

    def __init__(self, T=1000, beta_start = 1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, T) #[T]
        alphas = 1.0 - betas

        #cumulatice product of all alphas up to timestep t
        #Tells exactly how noist a sample is at step t
        self.alpha_bar = torch.cumprod(alphas,dim=0) 
        self.T = T
    
    def q_sample(self, x0, t):
        #x0: [B, C, T_coarse] - clean latent
        #t: [B]
        #This returns noisy latent at timestep t
        ab = self.alpha_bar[t.cpu()].to(x0.device)
        ab = ab.view(-1,1,1) #[B,1,1] for broadcasting
        noise = torch.randn_like(x0)
        #Returns both the noisy sample and the noise
        #since the training loop needs both (input and target)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise
    

        
