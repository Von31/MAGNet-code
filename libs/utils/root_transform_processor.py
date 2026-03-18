import dataclasses
from libs.dataloaders import TrainingData
import torch
from torch import Tensor
from typing import Literal
from jaxtyping import Float
import numpy as np

# from libs.model.multi_diffusion_transformer import MotionToken
from libs.model.vqvae.pose_vqvae import MultiPoseToken
from libs.utils.transforms import SE3, SO3

class RootTransformProcessor:
    @staticmethod
    def calc_canonical_trans_using_temporal_trans_grad(
        T_canonical_tm1_t: Float[Tensor, "S T *P 7"],
        T_world_first_canonical: Float[Tensor, "S *P 7"] | None = None,
    ) -> Float[Tensor, "S T+1 *P 7"]:            
        # S, T, P, _ = T_canonical_tm1_t.shape
        T = T_canonical_tm1_t.shape[1]

        # calculate transforms from canonical_t0 to canonical_t
        if T_world_first_canonical is None:
            # T_world_canonical_0 = SE3.identity(T_canonical_tm1_t.device, T_canonical_tm1_t.dtype).wxyz_xyz[None, None, :].expand(S, P, 7)
            T_world_canonical_0 = torch.zeros_like(T_canonical_tm1_t[:, 0])
            T_world_canonical_0[..., 0] = 1.
        else:
            T_world_canonical_0 = T_world_first_canonical
        
        T_world_canonical_t_list = [T_world_canonical_0]
        for t in range(T):
            T_world_canonical_t = (SE3(T_world_canonical_t_list[-1]) @ SE3(T_canonical_tm1_t[:, t])).wxyz_xyz
            T_world_canonical_t_list.append(T_world_canonical_t)

        T_world_canonical = torch.stack(T_world_canonical_t_list, dim=1)    

        return T_world_canonical

    @staticmethod
    def calc_acc_canonical_trans_using_temporal_trans_grad(
        T_canonical_tm1_t: Float[Tensor, "S T P 7"],
        window_size: int=4,
    ) -> Float[Tensor, "S T-W+1 P 7"]:
        T_canonical_tmw_t = T_canonical_tm1_t[:, :-window_size+1]
        tmw_t_list = [T_canonical_tmw_t]
        for t in range(1, window_size):
            offset = window_size - t - 1
            if offset > 0:
                T_tmp = T_canonical_tm1_t[:, t:-offset]
            else:
                T_tmp = T_canonical_tm1_t[:, t:]
            T_canonical_tmw = (SE3(tmw_t_list[-1]) @ SE3(T_tmp)).wxyz_xyz
            tmw_t_list.append(T_canonical_tmw)

        return tmw_t_list[-1]        

    @classmethod
    def convert_root_transform(
        cls,
        # pred_motion: MotionToken | MultiPoseToken,
        pred_motion: MultiPoseToken,
        gt_motion: TrainingData | None = None,
        context_seq_len: int = 0,
        mode: Literal["temporal", "temporal_partner"] = "temporal",
        self_partner_transform_pred: Float[Tensor, "S T P P-1 9"] | None = None,
    ) -> dict[str, np.ndarray]:
        print("mode", mode)
        if context_seq_len > 0:
            assert gt_motion is not None, "gt_motion must be provided if context_seq_len > 0"

        # S, T, P, _ = pred_motion.betas.shape
        S, T, P, _ = pred_motion.canonical_root_transforms.shape

        if P == 1:
            mode = "temporal"

        if mode == "temporal":
            T_world_first_canonical = None
            if gt_motion is not None:
                if gt_motion.T_world_canonical.ndim == 3:
                    T_world_first_canonical = SE3.from_9d(gt_motion.T_world_canonical[None, context_seq_len, :, :].expand(S, -1, -1)).wxyz_xyz
                else:
                    T_world_first_canonical = SE3.from_9d(gt_motion.T_world_canonical[:, context_seq_len, :, :].expand(S, -1, -1)).wxyz_xyz

            T_canonical_tm1_t = SE3.from_9d(pred_motion.canonical_tm1_t_transforms[:, context_seq_len:]).wxyz_xyz
            T_canonical_root = SE3.from_9d(pred_motion.canonical_root_transforms[:, context_seq_len:]).wxyz_xyz

            T_world_canonical = cls.calc_canonical_trans_using_temporal_trans(
                T_canonical_tm1_t=T_canonical_tm1_t,
                T_world_first_canonical=T_world_first_canonical,
            )
            

        elif mode == "temporal_partner":
            T_world_first_self_canonical = None
            if gt_motion is not None:
                T_world_first_self_canonical = SE3.from_9d(gt_motion.T_world_canonical[None, context_seq_len].expand(S, -1, -1)).wxyz_xyz

            T_canonical_tm1_t = SE3.from_9d(pred_motion.canonical_tm1_t_transforms[:, context_seq_len:]).wxyz_xyz
            T_canonical_root = SE3.from_9d(pred_motion.canonical_root_transforms[:, context_seq_len:]).wxyz_xyz
            if self_partner_transform_pred is not None:
                T_canonical_self_partner = SE3.from_9d(self_partner_transform_pred[:, context_seq_len:, :]).wxyz_xyz
            else:
                T_canonical_self_partner = SE3.from_9d(pred_motion.canonical_self_partner_transforms[:, context_seq_len:, :]).wxyz_xyz

            T_world_canonical = cls.calc_canonical_trans_using_partner_temporal_trans(
                T_canonical_tm1_t=T_canonical_tm1_t,
                T_canonical_self_partner=T_canonical_self_partner,
                T_world_first_canonical=T_world_first_self_canonical,
                anchor_person_id=0,
            )
            
        else:
            raise ValueError(f"Invalid mode: {mode}")



        T_world_root = (SE3(T_world_canonical) @ SE3(T_canonical_root)).wxyz_xyz

        if context_seq_len > 0:
            context_motion = SE3.from_9d(gt_motion.T_world_root[:context_seq_len]).wxyz_xyz
            T_world_root = torch.cat([context_motion.unsqueeze(0).expand(S, -1, -1, -1),
                                      T_world_root], dim=1)
            
        return T_world_root

    @classmethod
    def calc_T_world_root(
        cls,
        self_partner_canonical: Float[Tensor, "B T P P-1 9"],
        canonical_tm1_t: Float[Tensor, "B T P 9"],
        canonical_root: Float[Tensor, "B T P 9"],
    ) -> Float[Tensor, "B T P 7"]:
        B = canonical_tm1_t.shape[0]

        # calculate transforms from world to canonical
        T_world_first_partner_canonical = SE3.from_9d(self_partner_canonical[:, 0, 0]).wxyz_xyz
        T_world_first_self_canonical = torch.zeros((B, 1, 7), device=self_partner_canonical.device, dtype=self_partner_canonical.dtype)
        T_world_first_self_canonical[..., 0] = 1.
        T_world_first_canonical = torch.cat([T_world_first_self_canonical, T_world_first_partner_canonical], dim=1)

        # T_canonical_tm1_t = SE3.from_9d(pred_motion.canonical_tm1_t_transforms[:, context_seq_len:]).wxyz_xyz
        T_canonical_tm1_t = SE3.from_9d(canonical_tm1_t).wxyz_xyz
        T_world_canonical = cls.calc_canonical_trans_using_temporal_trans(
            T_canonical_tm1_t=T_canonical_tm1_t,
            T_world_first_canonical=T_world_first_canonical,
        )

        T_world_root = (SE3(T_world_canonical) @ SE3.from_9d(canonical_root)).wxyz_xyz
        return T_world_root

    @staticmethod
    def calc_canonical_trans_using_temporal_trans(
        T_canonical_tm1_t: Float[Tensor, "S T P 7"],
        T_world_first_canonical: Float[Tensor, "S P 7"] | None = None,
    ) -> Float[Tensor, "S T P 7"]:
        S, T, P, _ = T_canonical_tm1_t.shape

        # calculate transforms from canonical_t0 to canonical_t
        # T_canonical_t0_t = T_canonical_tm1_t.clone()
        T_canonical_t0_t = torch.empty_like(T_canonical_tm1_t)
        T_canonical_t0_t[:, 0] = SE3.identity(T_canonical_tm1_t.device, T_canonical_tm1_t.dtype).wxyz_xyz[None, None, :].expand(S, P, 7)
        for t in range(T-1):
            # T_canonical_t0_t[:, t+1] = (SE3(T_canonical_t0_t[:, t]) @ SE3(T_canonical_t0_t[:, t+1])).wxyz_xyz
            T_canonical_t0_t[:, t+1] = (SE3(T_canonical_t0_t[:, t]) @ SE3(T_canonical_tm1_t[:, t+1])).wxyz_xyz

        # calculate transforms from world to canonical_t
        if T_world_first_canonical is not None:
            T_world_canonical = (SE3(T_world_first_canonical[:, None, :, :]) @ SE3(T_canonical_t0_t)).wxyz_xyz
        else:
            T_world_canonical = T_canonical_t0_t

        return T_world_canonical
    

    @classmethod
    def calc_canonical_trans_using_partner_temporal_trans(
        cls,
        T_canonical_tm1_t: Float[Tensor, "S T P 7"],
        T_canonical_self_partner: Float[Tensor, "S T P P-1 7"],
        T_world_first_canonical: Float[Tensor, "S P 7"] | None = None,
        anchor_person_id: int = 0,
    ) -> Float[Tensor, "S T P 7"]:

        T_world_first_anchor_canonical = None
        if T_world_first_canonical is not None:
            T_world_first_anchor_canonical = T_world_first_canonical[:, [anchor_person_id], :]

        T_world_anchor_canonical = cls.calc_canonical_trans_using_temporal_trans(
            T_canonical_tm1_t=T_canonical_tm1_t[:, :, [anchor_person_id], :],
            T_world_first_canonical=T_world_first_anchor_canonical,
        )
        
        T_world_partner_canonical = (SE3(T_world_anchor_canonical) @ SE3(T_canonical_self_partner[:, :, anchor_person_id])).wxyz_xyz

        T_world_canonical = torch.cat([T_world_anchor_canonical, T_world_partner_canonical], dim=2)

        return T_world_canonical