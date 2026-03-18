from jaxtyping import Float
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from safetensors.torch import safe_open
import numpy as np
from einops import rearrange

from libs.model.vqvae.config import VQVAEBaseConfig
from libs.model.vqvae.pose_vqvae import PoseVQVAE, PoseToken
from libs.model.vqvae.quantizer import to_Tz
from libs.dataloaders import TrainingData
from libs.utils.transforms import SO3, SE3
from libs.utils.fncsmpl import SmplModel

class PoseNetworkBase(nn.Module):
    def __init__(self, config: VQVAEBaseConfig, device: torch.device):
        super(PoseNetworkBase, self).__init__()
        print("Mode: Relative Canonical Cond")
        self.config = config
        self.device = device
        self.vqvae = PoseVQVAE(config).to(device)

        self.z_mask = nn.Parameter(torch.zeros(1, 1, config.d_latent))
 
        
    @classmethod
    def load(
        cls, 
        model_path: Path, 
        config: VQVAEBaseConfig,
        device: torch.device,
    ) -> "PoseNetworkBase":
        print(f"Loading model from {model_path}")
        model = cls(config, device)
        with safe_open(model_path, framework="pt") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        for k in model.state_dict().keys():
            if k not in state_dict or model.state_dict()[k].shape != state_dict[k].shape:
                print(f"Key {k} not found in state dict or shape mismatch")
                state_dict[k] = model.state_dict()[k]
            else:
                pass
        model.load_state_dict(state_dict)
        model.to(device)
        return model

    def _training_step(
        self,
        val: Float[Tensor, "B T D"],
        cond_enc: Float[Tensor, "B T C"],
        cond_dec: Float[Tensor, "B Tz Cz"],
        mask: Float[Tensor, "B T 1"],
        train_step: int,
    ) -> Tuple[Float[Tensor, "..."], Dict[str, Any]]:
        if self.config.is_noise_augment:
            val_in = val + torch.randn_like(val) * self.config.noise_augment_std
            cond_enc_in = cond_enc + torch.randn_like(cond_enc) * self.config.noise_augment_std
            cond_dec_in = cond_dec + torch.randn_like(cond_dec) * self.config.noise_augment_std
        else:
            val_in = val
            cond_enc_in = cond_enc
            cond_dec_in = cond_dec
        

        output = self.vqvae.forward(val_in, cond_enc_in, cond_dec_in, mask)
        loss, loss_log = self._calc_loss(output, val, cond_enc, cond_dec, mask, train_step)
        return loss, loss_log
    
    def _calc_loss(
        self,
        output: Dict[str, Any],
        target: Float[Tensor, "B T D"],
        cond_enc: Float[Tensor, "B T C"],
        cond_dec: Float[Tensor, "B Tz Cz"],
        mask: Float[Tensor, "B T 1"],
        train_step: int,
    ) -> Float[Tensor, "batch person_num"]:
        recon_loss = F.mse_loss(
            output["x_tilde"], target, reduction="none", weight=mask.expand(-1, -1, target.shape[-1]))
        loss = recon_loss + output["commit_loss"]
        return loss, None


    def _encode(
        self,
        val: Float[Tensor, "B T D"],
        cond: Float[Tensor, "B T C"],
    ) -> Float[Tensor, "B Tz"]:
        z = self.vqvae.encode(val, cond)
        return z

    def _decode(
        self,
        z: Float[Tensor, "B Tz Dz"],
        cond: Float[Tensor, "B T C"],
    ) -> Float[Tensor, "B T D"]:
        return self.vqvae.decode(z, cond)

    def _decode_wo_quantize(
        self,
        z: Float[Tensor, "B Tz Dz"],
        cond: Float[Tensor, "B T C"],
    ) -> Float[Tensor, "B T D"]:
        return self.vqvae.decode_wo_quantize(z, cond)

    def _reconstruct(
        self,
        val: Float[Tensor, "B T D"],
        cond: Float[Tensor, "B T C"],
    ) -> Float[Tensor, "B T D"]:
        return self.vqvae.reconstruct(val, cond)


class PoseNetwork(PoseNetworkBase):
    def __init__(self, config: VQVAEBaseConfig, device: torch.device):
        super().__init__(config, device)

        self._loss_weight = config.loss_weight

        if config.loss_func == "mse":
            self._loss_func = F.mse_loss
        elif config.loss_func == "l1":
            self._loss_func = F.l1_loss
        elif config.loss_func == "smooth_l1":
            self._loss_func = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss function: {config.loss_func}")

        ms = np.load(self.config.mean_std_path)
        self.mean = torch.from_numpy(ms["mean"]).to(self.device).to(torch.float32)
        self.std = torch.from_numpy(ms["std"]).to(self.device).to(torch.float32)

        if self._loss_weight.get("joint_position_loss", 0.) > 0.:
            self.model = SmplModel.load(self.config.smpl_model_path).to(self.device)

    def training_step(
        self,
        data: TrainingData,
        train_step: int,
    ) -> Tuple[Float[Tensor, "batch timestep x_dim"], Dict[str, Any]]:
        pose_token = PoseToken.convert_from_training_data(data)
        val = pose_token.pack_val()
        cond_enc = pose_token.pack_enc_cond()
        cond_dec = pose_token.pack_dec_cond()
        mask = pose_token.mask.to(cond_enc.dtype)
        
        return self._training_step(val, cond_enc, cond_dec, mask, train_step)


    def geodesic_loss_from_6d(
        self, 
        pred_6d: Float[Tensor, "... 6"], 
        target_6d: Float[Tensor, "... 6"], 
        eps: float =1e-7
    ) -> Float[Tensor, "..."]:
        R1 = SO3.from_6d_to_matrix(pred_6d)
        R2 = SO3.from_6d_to_matrix(target_6d)
        Rt = R1.transpose(-1, -2) @ R2
        cos_t = ((Rt[...,0,0] + Rt[...,1,1] + Rt[...,2,2]) - 1.0) * 0.5
        cos_t = torch.clamp(cos_t, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_t)
        loss = self._loss_func(theta, torch.zeros_like(theta), reduction="none")
        return loss

    def geodesic_chordal_loss_from_6d(
        self,
        pred_6d: Float[Tensor, "... T 21 6"],
        target_6d: Float[Tensor, "... T 21 6"],
    ) -> Float[Tensor, "..."]:
        R1 = SO3.from_6d_to_matrix(pred_6d)
        R2 = SO3.from_6d_to_matrix(target_6d)
        Rt = R1.transpose(-1,-2) @ R2
        tr = Rt[...,0,0] + Rt[...,1,1] + Rt[...,2,2]
        c = (tr - 1.0) * 0.5
        return 1.0 - c

    def position_loss_from_6d(
        self,
        pred_6d: Float[Tensor, "... T 21 6"],
        target_6d: Float[Tensor, "... T 21 6"],
        pred_T_canonical_root: Float[Tensor, "... T 9"],
        target_T_canonical_root: Float[Tensor, "... T 9"],
        betas: Float[Tensor, "... T 10"],
    ) -> Float[Tensor, "... T 21 3"]:
        (*B, T, _, _) = pred_6d.shape
        pred_quat = SO3.from_6d(pred_6d).wxyz
        target_quat = SO3.from_6d(target_6d).wxyz

        shaped_model = self.model.with_shape(betas)
        pred_pose_model = shaped_model.with_pose_decomposed(
            body_quats=pred_quat,
            T_world_root=SE3.from_9d(pred_T_canonical_root).wxyz_xyz,
            is_only_body=True,
        )
        target_pose_model = shaped_model.with_pose_decomposed(
            body_quats=target_quat,
            T_world_root=SE3.from_9d(target_T_canonical_root).wxyz_xyz,
            is_only_body=True,
        )
        pred_joint_position = pred_pose_model.Ts_world_joint[..., 4:]
        target_joint_position = target_pose_model.Ts_world_joint[..., 4:]
        
        return self._loss_func(pred_joint_position, target_joint_position, reduction="none")
        

    def _calc_loss(
        self,
        output: Dict[str, Any],
        target: Float[Tensor, "batch timestep val_dim"],
        cond_enc: Float[Tensor, "batch cond_dim"],
        cond_dec: Float[Tensor, "batch cond_dim"],
        mask: Float[Tensor, "batch timestep 1"],
        train_step: int,
    ) -> Tuple[Float[Tensor, "batch person_num"], Dict[str, Any]]:
        
        output_token = PoseToken.unpack(output["x_tilde"], cond_enc, mask)
        target_token = PoseToken.unpack(target, cond_enc, mask)
        output_dn_token = PoseToken.denormalize(output_token, self.mean, self.std)
        target_dn_token = PoseToken.denormalize(target_token, self.mean, self.std)
        
        loss = {}
        loss["body_joint_rotations"] = (self._loss_func(
            output_token.body_joint_rotations, 
            target_token.body_joint_rotations,
            reduction="none"
        ) * mask[..., None].expand(-1, -1, 21, 6)).mean()
        loss["canonical_root_rotations"] = (self._loss_func(
            output_token.canonical_root_transforms[..., :6], 
            target_token.canonical_root_transforms[..., :6],
            reduction="none"
        ) * mask.expand(-1, -1, 6)).mean()
        loss["canonical_root_translations"] = (self._loss_func(
            output_token.canonical_root_transforms[..., 6:], 
            target_token.canonical_root_transforms[..., 6:],
            reduction="none"
        ) * mask.expand(-1, -1, 3)).mean()

        loss["geo_body_joint_rotations"] = (self.geodesic_loss_from_6d(
            output_dn_token.body_joint_rotations, 
            target_dn_token.body_joint_rotations
            ) * mask.expand(-1, -1, 21)).mean()
        loss["geo_canonical_root_rotations"] = (self.geodesic_loss_from_6d(
            output_dn_token.canonical_root_transforms[..., :6], 
            target_dn_token.canonical_root_transforms[..., :6]
            ) * mask[:, :, 0]).mean()

        if self._loss_weight.get("joint_position_loss", 0.) > 0.:
            loss["joint_position_loss"] = (self.position_loss_from_6d(
                output_dn_token.body_joint_rotations, 
                target_dn_token.body_joint_rotations,
                output_dn_token.canonical_root_transforms, 
                target_dn_token.canonical_root_transforms,
                output_dn_token.betas,
            ) * mask[..., None].expand(-1, -1, 21, 3)).mean()

        loss["commit_loss"] = output["commit_loss"]
    
        loss_log = loss.copy()

        loss_weight_sum = 0
        loss_val = 0
        for key, value in self._loss_weight.items():
            if value > 0.:
                loss_val += value * loss[key]
                loss_weight_sum += value

        loss_log.update(output["vq_metrics"])
    
        return loss_val, loss_log


    def encode(
        self,
        val: Float[Tensor, "B T D"],
        cond: Float[Tensor, "B T C"],
        mask: Float[Tensor, "B T 1"]
    ) -> Tuple[Float[Tensor, "B Tz D"], Float[Tensor, "B Tz 1"]]:
        z = self._encode(val, cond)

        mask_z = to_Tz(mask, z.shape[1], "nearest")
        mask_z = mask_z > 0.5

        z = torch.where(mask_z[..., None], z, torch.zeros_like(z))
        return z, mask_z.unsqueeze(-1)
        
    def decode(
        self,
        z: Float[Tensor, "B Tz D"],
        cond: Float[Tensor, "B T C"],
        mask_z: Float[Tensor, "B Tz 1"]|None = None,
    ) -> Tuple[Float[Tensor, "B T D"], Float[Tensor, "B T 1"]]:
        x_tilde = self._decode(z, cond)
        
        if mask_z is not None:
            mask = to_Tz(mask_z, x_tilde.shape[1], "nearest")
            mask = mask > 0.5
            
            x_tilde = torch.where(mask[..., None], x_tilde, torch.zeros_like(x_tilde))

        return x_tilde

    def decode_wo_quantize(
        self,
        z: Float[Tensor, "B Tz D"],
        cond: Float[Tensor, "B T C"],
        mask_z: Float[Tensor, "B Tz 1"]|None = None,
    ) -> Tuple[Float[Tensor, "B T D"], Float[Tensor, "B T 1"]]:
        x_tilde = self._decode_wo_quantize(z, cond)
        
        if mask_z is not None:
            mask = to_Tz(mask_z, x_tilde.shape[1], "nearest")
            mask = mask > 0.5
            
            x_tilde = torch.where(mask[..., None], x_tilde, torch.zeros_like(x_tilde))

        return x_tilde

    def inference(
        self,
        data: TrainingData,
    ) -> PoseToken:
        B, T, P, _ = data.betas.shape
        pose_token = PoseToken.convert_from_training_data(data)
        val = pose_token.pack_val()
        enc_cond = pose_token.pack_enc_cond()
        dec_cond = pose_token.pack_dec_cond()
        mask = pose_token.mask
        z, mask_z = self.encode(val, enc_cond, mask)
        x_tilde = self.decode(z, dec_cond, mask_z)

        pred_token = PoseToken.unpack(x_tilde, enc_cond)
        return pred_token