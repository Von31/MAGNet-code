from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu, 
                 is_rand_init_code, is_set_r, r_min, r_max, alive_thresh):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.is_rand_init_code = is_rand_init_code
        self.is_set_r = is_set_r
        self.r_min = r_min
        self.r_max = r_max
        self.alive_thresh = alive_thresh
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))
        self.register_buffer('code_sum', torch.zeros(self.nb_code, self.code_dim))
        self.register_buffer('code_count', torch.zeros(self.nb_code))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    @torch.no_grad()
    def init_codebook(self, x):
        if self.is_rand_init_code:
            new_w = torch.randn(self.nb_code, self.code_dim, device=x.device, dtype=x.dtype) * 0.01
        else:
            out = self._tile(x)
            new_w = out[:self.nb_code]

        if self.is_set_r:
            new_w = project_l2_band(new_w, r_min=self.r_min, r_max=self.r_max)
        self.codebook.copy_(new_w)
        self.code_sum.copy_(self.codebook)
        self.code_count.fill_(1.0)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        device = code_idx.device
        K = self.nb_code
        metrics = {
            "ppl": 0.0, "usage": 0.0, "top10": 0.0,
            "count_min": 0.0, "count_median": 0.0, "count_95_quantile": 0.0
        }
        if code_idx.numel() == 0:
            return metrics
        code_onehot = torch.zeros(K, code_idx.shape[0], device=device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, -1), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        total = code_count.sum().clamp_min(1e-6)
        prob = code_count / total
        ppl = torch.exp(-(prob * (prob + 1e-7).log()).sum())

        usage_mask = (code_count >= 1.0).float()
        usage = usage_mask.mean()
        topk = min(10, K)
        topk_share = prob.topk(topk).values.sum()
        metrics.update({
            "ppl":ppl.item(), "usage":usage.item(), "top10":topk_share.item(),
            "count_min":code_count.min().item(), "count_median":code_count.median().item(),
            "count_95_quantile":code_count.quantile(0.95).item()})
        return metrics
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        K, D = self.nb_code, self.code_dim
        device = x.device
        metrics = {
            "ppl": 0.0, "usage": 0.0, "top10": 0.0,
            "count_min": 0.0, "count_median": 0.0, "count_95_quantile": 0.0
        }
        if code_idx.numel() == 0:
            return metrics
        
        code_onehot = torch.zeros(K, x.shape[0], device=device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, -1), 1)

        batch_sum = code_onehot @ x  # nb_code, w
        batch_count = code_onehot.sum(dim=-1)  # nb_code

        self.code_sum.mul_(self.mu).add_(batch_sum, alpha=(1-self.mu))
        self.code_count.mul_(self.mu).add_(batch_count, alpha=(1-self.mu))

        alive = (self.code_count >= self.alive_thresh).float().unsqueeze(-1)
        code_update = self.code_sum / self.code_count.clamp_min(1e-6).unsqueeze(-1)

        out = self._tile(x)
        code_rand = out[:K]

        new_w = alive * code_update + (1 - alive) * code_rand
        if self.is_set_r:
            new_w = project_l2_band(new_w, r_min=self.r_min, r_max=self.r_max)
        self.codebook.copy_(new_w)

        total = batch_count.sum().clamp_min(1e-6)
        prob = batch_count / total
        ppl = torch.exp(-(prob * (prob + 1e-7).log()).sum())
        topk = min(10, K)
        topk_share = prob.topk(topk).values.sum()
        metrics.update({
            "ppl":ppl.item(), "usage":(batch_count > 0).float().mean().item(), 
            "top10":topk_share.item(),"count_min":batch_count.min().item(), 
            "count_median":batch_count.median().item(),
            "count_95_quantile":batch_count.quantile(0.95).item()})
   
        return metrics

    def quantize(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x32 = x.float()
            k32 = self.codebook.float().t()
            d = (x32**2).sum(dim=-1, keepdim=True) - 2 * (x32 @ k32) + (k32**2).sum(dim=0, keepdim=True)
            code_idx = d.argmin(dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def preprocess(self, z, mask=None):
        # NTC -> [NT, C]
        (*B, T, C) = z.shape
        z = z.view(-1, C)
        if mask is not None:
            m_weight = to_Tz(mask, T, mode="avg")
            m_valid  = to_Tz(mask, T, mode="nearest")
            weight = m_weight.view(-1)
            valid = (m_valid >= 0.5).reshape(-1)
        else:
            weight = None
            valid = None

        return z, weight, valid


    def forward(self, z, mask=None):
        (*B, T, C) = z.shape
        z, weight, valid = self.preprocess(z, mask)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(z if valid is None else z[valid])

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(z)
        z_q = self.dequantize(code_idx)

        # Loss
        se = (z - z_q.detach()) ** 2
        per_tok = se.mean(-1)
        if  weight is None:
            commit_loss = per_tok.mean()
        else:
            denom = weight.sum().clamp_min(1e-7)
            commit_loss = (per_tok * weight).sum() / denom
            # if commit_loss.item() > 1.0:
            #     import ipdb; ipdb.set_trace()

        # Update embeddings
        if self.training:
            vq_metrics = self.update_codebook(
                z if valid is None else z[valid], 
                code_idx if valid is None else code_idx[valid])
        else : 
            vq_metrics = self.compute_perplexity(
                code_idx if valid is None else code_idx[valid])

        if weight is not None:
            vq_metrics['denom'] = denom

        # Passthrough
        z_q_bar = z + (z_q - z).detach()

        # Maskout
        if valid is not None:
            z_q_bar = torch.where(
                valid[..., None], z_q_bar, torch.zeros_like(z_q_bar))        

        # Postprocess
        z_q_bar = z_q_bar.view(*B, T, C)
        
        return z_q_bar, commit_loss, vq_metrics


def to_Tz(mask_T, Tz, mode="avg"):  # mask_T: [N,T] or [N,T,1]
    m = mask_T.float()
    if m.dim() == 2:
        m = m.unsqueeze(1)              # [N,1,T]
    elif m.dim() == 3:
        m = m.reshape(m.shape[0], 1, m.shape[1])
    if mode == "avg":                   # loss weighting
        m = F.adaptive_avg_pool1d(m, output_size=Tz)   # [N,1,Tz]
    elif mode == "nearest":             # for valid/invalid
        m = F.interpolate(m, size=Tz, mode="nearest")  # [N,1,Tz]
    else:                              # for linear interpolation
        m = F.interpolate(m, size=Tz, mode="linear", align_corners=False)
    return m.squeeze(1)


def project_l2_band(w, r_min=None, r_max=None, eps=1e-6):
    n = w.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.ones_like(n)
    if r_max is not None: scale = torch.minimum(scale, r_max / n)
    if r_min is not None: scale = torch.maximum(scale, r_min / n)
    return w * scale