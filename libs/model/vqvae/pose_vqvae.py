from __future__ import annotations

import dataclasses
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from libs.utils.tensor_dataclass import TensorDataclass
from libs.model.vqvae.config import VQVAEBaseConfig
from libs.model.vqvae.encdec import Encoder, Decoder
from libs.model.vqvae.quantizer import QuantizeEMAReset
from libs.dataloaders import StdMeanIdx, TrainingData
from einops import rearrange


class PoseToken(TensorDataclass):
    body_joint_rotations: Float[Tensor, "#B T 21 6"]
    canonical_root_transforms: Float[Tensor, "#B T 9"]
    canonical_tm1_t_transforms: Float[Tensor, "#B T 9"]
    betas: Float[Tensor, "#B T 10"]
    mask: Float[Tensor, "#B T 1"]

    @staticmethod
    def get_data_to_token_mapping_dict() -> dict:
        return {
                "body_joint_rotations": "body_joint_rotations",
                "canonical_root_transforms": "T_canonical_root",
                "canonical_tm1_t_transforms": "T_canonical_tm1_canonical_t",
                "betas": "betas",
                "mask": "tpose_mask",
                }

    @classmethod
    def convert_from_training_data(
        cls,
        data: TrainingData,
    ) -> 'PoseToken':
        token_data = {}
        for k, v in cls.get_data_to_token_mapping_dict().items():
            raw_data = getattr(data, v)
            (B, T, P, *D) = raw_data.shape
            raw_data = rearrange(raw_data, "B T P ... -> (B P) T ...")
            token_data[k] = raw_data

        return cls(**token_data)

    def convert_to_mutli_pose_token(self, person_num: int = 1) -> 'MultiPoseToken':
        token_data = {}
        for field in dataclasses.fields(self):
            raw_data = getattr(self, field.name)
            token_data[field.name] = rearrange(raw_data, "(B P) T ... -> B T P ...", P=person_num)
        return MultiPoseToken(**token_data)

    @staticmethod
    def get_val_dim() -> int:
        return 21 * 6 + 9

    @staticmethod
    def get_enc_cond_dim() -> int:
        return 10 + 9

    @staticmethod
    def get_dec_cond_dim() -> int:
        return 10 + 9 * 4

    def pack_val(self) -> Float[Tensor, "*B T D"]:
        (*BT, _, _) = self.body_joint_rotations.shape
        return torch.cat(
                [
                    self.body_joint_rotations.reshape((*BT, 21*6)),
                    self.canonical_root_transforms,
                ],
                dim=-1,
            )
 
    def pack_enc_cond(self) -> Float[Tensor, "*B T C"]:
        return torch.cat(
                [
                    self.betas,
                    self.canonical_tm1_t_transforms,
                ],
                dim=-1,
            )

    def pack_dec_cond(self) -> Float[Tensor, "*B Tz Cz"]:
        (*B, T, _) = self.canonical_tm1_t_transforms.shape
        return torch.cat(
                [
                    self.betas[..., ::4, :],
                    self.canonical_tm1_t_transforms.reshape(*B, T//4, 4*9),
                ],
                dim=-1,
            )

    @classmethod
    def unpack(
        cls,
        val: Float[Tensor, "*B T D"],
        cond_enc: Float[Tensor, "*B T C"],
        mask: Float[Tensor, "*B T 1"] | None = None,
    ) -> 'PoseToken':
        (*BT, _) = val.shape
        assert val.shape[0] == cond_enc.shape[0]
        body_joint_rotations = val[...,  :21*6].reshape((*BT, 21, 6))
        canonical_root_transforms = val[..., 21*6:21*6+9]
        
        betas = cond_enc[..., :10]
        canonical_tm1_t_transforms = cond_enc[..., 10:]

        if mask is None:
            mask = torch.ones((*BT, 1), dtype=torch.bool, device=val.device)
        return cls(
            body_joint_rotations=body_joint_rotations,
            canonical_root_transforms=canonical_root_transforms,
            canonical_tm1_t_transforms=canonical_tm1_t_transforms,
            betas=betas,
            mask=mask,
        )

    @classmethod
    def denormalize(
        cls,
        token: 'PoseToken',
        mean: Float[Tensor, "ms_state"],
        std: Float[Tensor, "ms_state"],
    ) -> 'PoseToken':
        idx = StdMeanIdx()  

        (*batch, timestep, _, _) = token.body_joint_rotations.shape
        body_joint_rotations=(token.body_joint_rotations.reshape(*batch, timestep, 21*6) * std[idx.body_joint_rotations] + mean[idx.body_joint_rotations]).reshape(*batch, timestep, 21, 6)
        canonical_root_transforms=token.canonical_root_transforms * std[idx.T_canonical_root] + mean[idx.T_canonical_root]
        canonical_tm1_t_transforms=token.canonical_tm1_t_transforms * std[idx.T_canonical_tm1_canonical_t] + mean[idx.T_canonical_tm1_canonical_t]
        betas = token.betas * std[idx.betas] + mean[idx.betas]
        mask = token.mask
        return cls(
            body_joint_rotations=body_joint_rotations,
            canonical_root_transforms=canonical_root_transforms,
            canonical_tm1_t_transforms=canonical_tm1_t_transforms,
            betas=betas,
            mask=mask,
        )

    @classmethod
    def normalize(
        cls,
        token: 'PoseToken',
        mean: Float[Tensor, "ms_state"],
        std: Float[Tensor, "ms_state"]
    ) -> 'PoseToken':
        idx = StdMeanIdx()
        (*batch, timestep, _, _) = token.body_joint_rotations.shape
        return cls( 
            body_joint_rotations=((token.body_joint_rotations.reshape(*batch, timestep, 21*6) - mean[idx.body_joint_rotations]) / std[idx.body_joint_rotations]).reshape(*batch, timestep, 21, 6),
            canonical_root_transforms=(token.canonical_root_transforms - mean[idx.T_canonical_root]) / std[idx.T_canonical_root],
            canonical_tm1_t_transforms=(token.canonical_tm1_t_transforms - mean[idx.T_canonical_tm1_canonical_t]) / std[idx.T_canonical_tm1_canonical_t],
            betas=(token.betas - mean[idx.betas]) / std[idx.betas],
            mask=token.mask,
        )

class MultiPoseToken(PoseToken):
    body_joint_rotations: Float[Tensor, "#B T P 21 6"]
    canonical_root_transforms: Float[Tensor, "#B T P 9"]
    canonical_tm1_t_transforms: Float[Tensor, "#B T P 9"]
    betas: Float[Tensor, "#B P 10"]
    mask: Float[Tensor, "#B T P 1"] 


class PoseVQVAE(nn.Module):
    def __init__(self, config: VQVAEBaseConfig):    
        super().__init__()

        self.wo_cond = config.without_cond

        d_val = PoseToken.get_val_dim()
        if self.wo_cond:
            d_enc_cond = 0
            d_dec_cond = 0
        else:
            d_enc_cond = PoseToken.get_enc_cond_dim()
            d_dec_cond = PoseToken.get_dec_cond_dim()
        
        self.quantizer = QuantizeEMAReset(
            config.code_nb, config.d_latent, config.ema_mu,
            config.is_rand_init_code, 
            config.is_set_r, config.r_min, config.r_max,
            config.alive_thresh,
            )
        self.encoder = Encoder(d_val, d_enc_cond, config)
        self.decoder = Decoder(d_val, d_dec_cond, config)

    def quantize(self, z):
        (*B, T, C) = z.shape
        # latents = self.quantizer.quantize(z.view(-1, C))
        latents = self.quantizer.quantize(z.reshape(-1, C))
        latents = latents.view(*B, T)
        return latents

    def dequantize(self, latents):
        (*B, T, C) = latents.shape
        z_q_x = self.quantizer.dequantize(latents)
        return z_q_x

    def encode(self, x, cond):
        if self.wo_cond:
            z_e_x = self.encoder(x)
        else:
            z_e_x = self.encoder(x, cond)
        latents = self.quantize(z_e_x)
        z_q_x = self.dequantize(latents)
        if z_e_x.abs().max() > 10:
            print('x', x.min(), x.max(), x.mean())
            print('z_e_x', z_e_x.min(), z_e_x.max(), z_e_x.mean())
            print('z_q_x', z_q_x.min(), z_q_x.max(), z_q_x.mean())
        return z_q_x

    def decode(self, z, cond):
        latents = self.quantize(z)
        z_q_x = self.dequantize(latents)
        if self.wo_cond:
            x_tilde = self.decoder(z_q_x)
        else:
            x_tilde = self.decoder(z_q_x, cond)
        return x_tilde

    def decode_wo_quantize(self, z, cond):
        if self.wo_cond:
            x_tilde = self.decoder(z)
        else:
            x_tilde = self.decoder(z, cond)
        return x_tilde

    def reconstruct(self, x, cond):
        z_e_x = self.encode(x, cond)
        latents = self.quantize(z_e_x)
        z_q_x = self.dequantize(latents)
        x_tilde = self.decode(z_q_x, cond)
        return x_tilde   
    
    def forward(self, val, cond_enc, cond_dec, mask=None):
        if self.wo_cond:
            z_e = self.encoder(val)
        else:
            z_e = self.encoder(val, cond_enc)
        
        z_q_st, commit_loss, vq_metrics = self.quantizer(
            z_e, mask=mask)
        
        if self.wo_cond:
            x_tilde = self.decoder(z_q_st)
        else:
            x_tilde = self.decoder(z_q_st, cond_dec)
        
        return {
            "x_tilde": x_tilde, 
            "commit_loss": commit_loss, 
            "vq_metrics": vq_metrics,
        }
