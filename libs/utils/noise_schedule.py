import torch
import numpy as np

def cosine_schedule(
    timesteps: int,
    s: float = 0.008,
    clip_min: float = 1e-9
):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = alphas_cumprod[1:]

    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas

    betas = torch.clip(betas, clip_min, 1.0)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {'betas': betas, 'alphas_cumprod': alphas_cumprod, 'alphas': alphas}

def make_logsnr_rand_fn(alpha_bar: torch.Tensor, mu: float = 0.0, sigma: float = 1.0):
    lam = torch.log(alpha_bar / (1 - alpha_bar + 1e-12))
    weights = torch.exp(-0.5 * ((lam - mu) / (sigma + 1e-12))**2)
    probs = (weights / weights.sum()).float()
    cdf = probs.cumsum(0)

    def rand_fn(size, device=None):
        u = torch.rand(size, device=device)
        idx = torch.searchsorted(cdf.to(u.device), u, right=True)
        return torch.clamp(idx, 0, len(probs)-1)

    return rand_fn