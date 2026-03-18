import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from typing import Literal
from einops import rearrange
import numpy as np
from libs.utils.tensor_dataclass import TensorDataclass
from libs.dataloaders import StdMeanIdx
from libs.model.dfot.config import DFOTConfig



class MotionToken(TensorDataclass):
    pose_latent: Float[Tensor, "*B T P D"]
    canonical_tm1_t_transforms: Float[Tensor, "*B T P 4*9"]
    canonical_self_partner_transforms: Float[Tensor, "*B T P P-1 4*9"]
    is_cat: bool = True

    @staticmethod
    def get_data_to_token_mapping_dict() -> dict:
        return {
                "canonical_tm1_t_transforms": "T_canonical_tm1_canonical_t",
                "canonical_self_partner_transforms": "T_self_canonical_partner_canonical",
                }

    @staticmethod
    def get_dim(
        latent_dim: int, 
        person_num:int, 
        is_wo_self_partner:bool=False,
    )->int:
        if is_wo_self_partner:
            return latent_dim + 4 * 9
        else:
            return latent_dim + person_num * 4 * 9
    @staticmethod
    def get_cond_dim(
        person_num:int,
    )->int:
        return person_num * 9

    def pack(self)->Float[Tensor, "*B T P D"]:
        
            (*S, _) = self.pose_latent.shape
            return torch.cat([
                self.pose_latent, 
                self.canonical_tm1_t_transforms,
                self.canonical_self_partner_transforms.reshape((*S, -1))], 
                dim=-1)
  

    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*B T P X"],
    ) -> 'MotionToken':
        (*B, T, P, D) = x.shape
        sp_dim = P * 4 * 9
        (
            pose_latent, 
            canonical_tm1_t_transforms,
            canonical_self_partner_transforms
        ) = torch.split(
            x, [D-sp_dim, 4*9, sp_dim-4*9], dim=-1
        )
        canonical_tm1_t_transforms = canonical_tm1_t_transforms.reshape((*B, T, P, 4*9))
        canonical_self_partner_transforms = canonical_self_partner_transforms.reshape((*B, T, P, P-1, 4*9))
        return MotionToken(
            pose_latent=pose_latent,
            canonical_tm1_t_transforms=canonical_tm1_t_transforms,
            canonical_self_partner_transforms=canonical_self_partner_transforms,
        )

    @classmethod
    def denormalize(
        cls,
        token: 'MotionToken',
        mean: Float[Tensor, "S"],
        std: Float[Tensor, "S"],
    ) -> 'MotionToken':
            idx = StdMeanIdx()
            (*BTP, P, _) = token.canonical_self_partner_transforms.shape
            canonical_tm1_t_transforms = token.canonical_tm1_t_transforms.reshape((*BTP, 4, 9)) * std[idx.T_canonical_tm1_canonical_t] + mean[idx.T_canonical_tm1_canonical_t]
            canonical_tm1_t_transforms = canonical_tm1_t_transforms.reshape((*BTP, 4*9))
            canonical_self_partner_transforms = token.canonical_self_partner_transforms.reshape((*BTP, P, 4, 9)) * std[idx.T_self_canonical_partner_canonical] + mean[idx.T_self_canonical_partner_canonical]
            canonical_self_partner_transforms = canonical_self_partner_transforms.reshape((*BTP, P, 4*9))
            return cls(
                pose_latent=token.pose_latent,
                canonical_tm1_t_transforms=canonical_tm1_t_transforms,
                canonical_self_partner_transforms=canonical_self_partner_transforms,
            )
        
    @classmethod
    def normalize(
        cls,
        token: 'MotionToken',
        mean: Float[Tensor, "S"],
        std: Float[Tensor, "S"]
    ) -> 'MotionToken':
        if token.is_cat:
            idx = StdMeanIdx()
            (*BTP, P, _) = token.canonical_self_partner_transforms.shape
            canonical_tm1_t_transforms = (token.canonical_tm1_t_transforms.reshape((*BTP, 4, 9)) - mean[idx.T_canonical_tm1_canonical_t]) / std[idx.T_canonical_tm1_canonical_t]
            canonical_tm1_t_transforms = canonical_tm1_t_transforms.reshape((*BTP, 4*9))
            canonical_self_partner_transforms = (token.canonical_self_partner_transforms.reshape((*BTP, P, 4, 9)) - mean[idx.T_self_canonical_partner_canonical]) / std[idx.T_self_canonical_partner_canonical]
            canonical_self_partner_transforms = canonical_self_partner_transforms.reshape((*BTP, P, 4*9))
            return cls(
                pose_latent=token.pose_latent,
                canonical_tm1_t_transforms=canonical_tm1_t_transforms,
                canonical_self_partner_transforms=canonical_self_partner_transforms,
            )
        else:
            return token
###
class MotionToken__(TensorDataclass):
    pose_latent: Float[Tensor, "*B Tz P D"]
    canonical_tm1_t_transforms: Float[Tensor, "*B Tz*4 P 9"]
    canonical_self_partner_transforms: Float[Tensor, "*B Tz*4 P P-1 9"]
    is_wo_self_partner: bool = False

    @staticmethod
    def get_data_to_token_mapping_dict() -> dict:
        return {
                "canonical_tm1_t_transforms": "T_canonical_tm1_canonical_t",
                "canonical_self_partner_transforms": "T_self_canonical_partner_canonical",
                }

    @staticmethod
    def get_dim(
        latent_dim: int, 
        person_num:int, 
        is_wo_self_partner:bool=False,
    )->int:
        if is_wo_self_partner:
            return latent_dim + 4 * 9
        else:
            return latent_dim + person_num * 4 * 9

    @staticmethod
    def get_cond_dim(
        person_num:int,
    )->int:
        return person_num * 9

    def pack(self)->Float[Tensor, "*B Tz P D"]:
        if self.is_wo_self_partner:
            (*S, _) = self.pose_latent.shape
            return torch.cat([
                self.pose_latent, 
                rearrange(self.canonical_tm1_t_transforms, "... (tz f) s d -> ... tz s (f d)", f=4)],
            dim=-1)
        else:

            (*S, _) = self.pose_latent.shape
            return torch.cat([
                self.pose_latent, 
                rearrange(self.canonical_tm1_t_transforms, "... (tz f) s d -> ... tz s (f d)", f=4),
                rearrange(self.canonical_self_partner_transforms, "... (tz f) s p d -> ... tz s (f p d)", f=4)], 
            dim=-1)

    @classmethod
    def unpack(
        cls,
        x: Float[Tensor, "*B Tz P X"],
        is_wo_self_partner: bool=False,
    ) -> 'MotionToken':
        (*B, Tz, P, D) = x.shape
        if is_wo_self_partner:
            sp_dim = 4 * 9
            (
                pose_latent, 
                canonical_tm1_t_transforms,
            ) = torch.split(
                x, [D-sp_dim, sp_dim], dim=-1
            )
            canonical_self_partner_transforms = torch.zeros((*B, Tz, P-1, 4*9), device=x.device, dtype=x.dtype)
        else:
            sp_dim = P * 4 * 9
            (
                pose_latent, 
                canonical_tm1_t_transforms,
                canonical_self_partner_transforms
            ) = torch.split(
                x, [D-sp_dim, 4*9, sp_dim-4*9], dim=-1
            )
        canonical_tm1_t_transforms = rearrange(canonical_tm1_t_transforms, "... tz s (f d) -> ... (tz f) s d", f=4)
        canonical_self_partner_transforms = rearrange(canonical_self_partner_transforms, "... tz s (f p d) -> ... (tz f) s p d", f=4, p=P-1)
        return MotionToken(
            pose_latent=pose_latent,
            canonical_tm1_t_transforms=canonical_tm1_t_transforms,
            canonical_self_partner_transforms=canonical_self_partner_transforms,
        )

    @classmethod
    def denormalize(
        cls,
        token: 'MotionToken',
        mean: Float[Tensor, "S"],
        std: Float[Tensor, "S"],
    ) -> 'MotionToken':
        idx = StdMeanIdx()
        canonical_tm1_t_transforms = token.canonical_tm1_t_transforms * std[idx.T_canonical_tm1_canonical_t] + mean[idx.T_canonical_tm1_canonical_t]
        canonical_self_partner_transforms = token.canonical_self_partner_transforms * std[idx.T_self_canonical_partner_canonical] + mean[idx.T_self_canonical_partner_canonical]
        return cls(
            pose_latent=token.pose_latent,
            canonical_tm1_t_transforms=canonical_tm1_t_transforms,
            canonical_self_partner_transforms=canonical_self_partner_transforms,
        )

    @classmethod
    def normalize(
        cls,
        token: 'MotionToken',
        mean: Float[Tensor, "S"],
        std: Float[Tensor, "S"]
    ) -> 'MotionToken':
        idx = StdMeanIdx()
        canonical_tm1_t_transforms = (token.canonical_tm1_t_transforms - mean[idx.T_canonical_tm1_canonical_t]) / std[idx.T_canonical_tm1_canonical_t]
        canonical_self_partner_transforms = (token.canonical_self_partner_transforms - mean[idx.T_self_canonical_partner_canonical]) / std[idx.T_self_canonical_partner_canonical]
        return cls(
            pose_latent=token.pose_latent,
            canonical_tm1_t_transforms=canonical_tm1_t_transforms,
            canonical_self_partner_transforms=canonical_self_partner_transforms,
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RotaryPosEmb(nn.Module):
    def __init__(
        self, 
        dim: int, 
        theta: float = 10000, 
    ) -> None:
        
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    def forward(
        self, 
        x: Float[Tensor, "*#B T P D"],
    ) -> Float[Tensor, "*#B T P D"]:
        batch_size, seq_len, person_num, _ = x.shape
        device = x.device

        pos = torch.arange(seq_len).float().repeat_interleave(person_num).to(device)

        freqs = torch.einsum("i,j->ij", pos, self.inv_freq.to(device))
        sin, cos = freqs.sin().unsqueeze(0), freqs.cos().unsqueeze(0)

        x = x.reshape(batch_size, seq_len*person_num, -1)

        x1, x2 = x[..., ::2], x[..., 1::2]

        # return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.reshape(batch_size, seq_len, person_num, -1)

class PersonEmb(nn.Module):
    def __init__(self, dim: int, person_num: int):
        super().__init__()
        self.person_emb = nn.Embedding(person_num, dim)

    def forward(
        self, 
        x: Float[Tensor, "*#B T P Dx"],
    ) -> Float[Tensor, "*#B T P De"]:
        device = x.device
        batch, timesteps, person_num, _ = x.shape
        idx = torch.arange(person_num).repeat(batch, timesteps, 1).to(device)
        return self.person_emb(idx).to(device)

class TransformerModel(nn.Module):
    def __init__(self, config: DFOTConfig):
        super(TransformerModel, self).__init__()

        self.config = config
        
        # transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_latent,
            nhead=config.num_heads, 
            dim_feedforward=config.d_feedforward, 
            dropout=config.dropout_p,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.layers,
        )

        # activation function
        Activation = {"gelu": nn.GELU, "relu": nn.ReLU}[config.activation]

        # input dimension
        x_dim = MotionToken.get_dim(
            latent_dim=config.vqvae_cfg.d_latent, 
            person_num=config.person_num,
            is_wo_self_partner=config.without_self_partner,
        )
            
        # positional encoding for time, person and noise level
        k_embed_dim = config.d_latent // 2
        if config.person_embedding_mode == "add":
            p_embed_dim = config.d_latent
            input_dim = x_dim + k_embed_dim 
        elif config.person_embedding_mode == "concat":
            p_embed_dim = config.d_latent // 2
            input_dim = x_dim + k_embed_dim + p_embed_dim
        else:
            raise ValueError(f"Invalid person embedding mode: {config.person_embedding_mode}")
        
        # self.t_embed = SinusoidalPosEmb(dim=config.d_latent)
        self.t_embed = RotaryPosEmb(dim=config.d_latent)
        self.k_embed = SinusoidalPosEmb(dim=k_embed_dim)
        self.p_embed = PersonEmb(dim=p_embed_dim, person_num=config.person_num)


        # initial mlp
        self.init_mlp = nn.Sequential(
            nn.Linear(input_dim, config.d_latent),
            Activation(),
            nn.Linear(config.d_latent, config.d_latent),
            Activation(),
            nn.Linear(config.d_latent, config.d_latent),
        )
        self.out = nn.Linear(config.d_latent, x_dim)

        
    def forward(
        self, 
        x: Float[Tensor, "B T P D"],
        k: Float[Tensor, "B T P"],
        is_causal: bool = False,
    ) -> Float[Tensor, "B T P D"]:
        B, T, P, _ = x.shape
        
        # get k embedding
        k_embed = rearrange(self.k_embed(k.flatten()), "(b t p) d -> b t p d", b=B, t=T, p=P)

        if self.config.person_embedding_mode == "add":
            # concat x and k_embed, and calculate x & k embedding
            x = torch.cat((x, k_embed), dim=-1)
            x = self.init_mlp(x)

            # add positional encoding & person embedding
            x = x + self.t_embed(x) + self.p_embed(x)
        elif self.config.person_embedding_mode == "concat":
            # concat x, k_embed, and p_embed, and calculate x & k & p embedding
            p_embed = self.p_embed(x)
            x = torch.cat((x, k_embed, p_embed), dim=-1)
            x = self.init_mlp(x)

            # add positional encoding
            x = x + self.t_embed(x)
        else:
            raise ValueError(f"Invalid person embedding mode: {self.config.person_embedding_mode}")

        x = x.reshape(B, T*P, -1)

        # get causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(len(x), x.device) if is_causal else None

        # add transformer
        x = self.transformer(x, mask=mask, is_causal=is_causal)

        # add output layer
        x = self.out(x)

        return x.reshape(B, T, P, -1)