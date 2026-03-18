import dataclasses
import numpy as np
import torch
import torch.utils.data
from torch import Tensor
from jaxtyping import Bool, Float
from libs.utils.tensor_dataclass import TensorDataclass


from libs.dataloaders import StdMeanIdx

FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
CONTACT_VEL_THRESH = 0.005
CONTACT_HEIGHT_THRESH = 0.04

VIS_CONTACT = False
VIS_COORD = False



class TrainingData(TensorDataclass):
    betas: Float[Tensor, "*B T P 10"]

    body_joint_rotations: Float[Tensor, "*B T P 21 6"]
    hand_joint_rotations: Float[Tensor, "*B T P 30 6"]

    T_canonical_tm1_canonical_t: Float[Tensor, "*B T P 9"]
    T_canonical_root: Float[Tensor, "*B T P 9"]

    T_world_root: Float[Tensor, "*B T P 9"]
    T_world_canonical: Float[Tensor, "*B T P 9"]

    T_self_canonical_partner_canonical: Float[Tensor, "*B T P P-1 9"]

    body_contacts: Float[Tensor, "*B T P 21"]

    tpose_mask: Bool[Tensor, "*B T P 1"]

    def __len__(self):
        return self.betas.shape[0]

    @staticmethod
    def normalize(
        x: Float[Tensor, "*B T P D"],
        mean: Float[Tensor, "X"],
        std: Float[Tensor, "X"],
    ) -> Float[Tensor, "*B T P D"]:
        # normalize all data except for boolean value such as body_contacts and tpose_mask
        P = x.shape[-2]

        # last 9 values are for T_self_canonical_partner_canonical
        # so we need to repeat the last 9 values for partner num
        target = mean.shape[0] + 9 * (P-2)

        if P <= 2:
            # partner num is 0 or 1
            mean_ = mean[:target]
            std_ = std[:target]
        else:
            # partner num is 2 or more, repeat last 9 values for partner num
            mean_ = torch.cat([mean, mean[-9:].repeat(P-2)], dim=0)
            std_ = torch.cat([std, std[-9:].repeat(P-2)], dim=0)

        x[..., :target] = (x[..., :target] - mean_) / std_
        return x

    @staticmethod
    def denormalize(
        x: Float[Tensor, "*B T P D"],
        mean: Float[Tensor, "X"],
        std: Float[Tensor, "X"],
    ) -> Float[Tensor, "*B T P D"]:
        # denormalize all data except for boolean value such as body_contacts and tpose_mask
        person_num = x.shape[-2]

        # last 9 values are for T_self_canonical_partner_canonical
        # so we need to repeat the last 9 values for partner num
        target = mean.shape[0] + 9 * (person_num-2)

        if person_num <= 2:
            # partner num is 0 or 1
            mean_ = mean[:target]
            std_ = std[:target]
        else:
            # partner num is 2 or more
            mean_ = torch.cat([mean, mean[-9:].repeat(person_num-2)], dim=0)
            std_ = torch.cat([std, std[-9:].repeat(person_num-2)], dim=0)

        x[..., :target] = x[..., :target] * std_ + mean_
        return x

    @staticmethod
    def normalize_unpacked(
        x: 'TrainingData', 
        mean: Float[Tensor, "X"], 
        std: Float[Tensor, "X"],
    ) -> 'TrainingData':
        idx = StdMeanIdx()
        for f in dataclasses.fields(x):
            if f.name in ["body_contacts", "tpose_mask"]:
                continue
            data = getattr(x, f.name)
            data_idx = getattr(idx, f.name)
            shape = data.shape
            data = data.reshape(-1, len(data_idx))
            data = (data - mean[data_idx]) / std[data_idx]
            data = data.reshape(shape)
            setattr(x, f.name, data)
        return x

    @staticmethod
    def denormalize_unpacked(
        x: 'TrainingData', 
        mean: Float[Tensor, "X"], 
        std: Float[Tensor, "X"],
    ) -> 'TrainingData':
        idx = StdMeanIdx()
        for f in dataclasses.fields(x):
            if f.name in ["body_contacts", "tpose_mask"]:
                continue
            data = getattr(x, f.name)
            data_idx = getattr(idx, f.name)
            shape = data.shape
            data = data.reshape(-1, len(data_idx))
            data = data * std[data_idx] + mean[data_idx]
            data = data.reshape(shape)
            setattr(x, f.name, data)
        return x

    @staticmethod
    def get_packed_dim(person_num: int) -> int:
        return (10 + 51 * 6 + 4 * 9 + 9 * (person_num-1) + 21 + 1)

    def pack(self) -> Float[Tensor, "*B T P D"]:
        size = self.betas.shape[:-1]

        data_list = [
            self.betas,
            self.body_joint_rotations,
            self.hand_joint_rotations,
            self.T_canonical_tm1_canonical_t,
            self.T_canonical_root,
            self.T_world_root,
            self.T_world_canonical,
            self.T_self_canonical_partner_canonical,
            self.body_contacts,
            self.tpose_mask.to(torch.float32),
        ]
        return torch.cat(
            [
                x.reshape((*size, -1))
                for x in data_list
            ],
            dim=-1,
        )

    @classmethod
    def unpack(
        cls, 
        x: Float[Tensor, "*B T P D"],
    )-> 'TrainingData':
        size = x.shape[:-1]
        P = x.shape[-2]
        assert x.shape[-1] == cls.get_packed_dim(P)

        (
            betas_flat,
            body_joint_rotations_flat,
            hand_joint_rotations_flat,
            T_canonical_tm1_canonical_t_flat,
            T_canonical_root_flat,
            T_world_root_flat,
            T_world_canonical_flat,
            T_self_canonical_partner_canonical_flat,
            body_contacts_flat,
            tpose_mask_flat
        ) = torch.split(x, [10, 21*6, 30*6, 9, 9, 9, 9, 9*(P-1), 21, 1], dim=-1)

        betas = betas_flat.reshape((*size, 10))
        body_joint_rotations = body_joint_rotations_flat.reshape((*size, 21, 6))
        hand_joint_rotations = hand_joint_rotations_flat.reshape((*size, 30, 6))
        T_canonical_tm1_canonical_t = T_canonical_tm1_canonical_t_flat.reshape((*size, 9))
        T_canonical_root = T_canonical_root_flat.reshape((*size, 9))
        T_world_root = T_world_root_flat.reshape((*size, 9))
        T_world_canonical = T_world_canonical_flat.reshape((*size, 9))
        T_self_canonical_partner_canonical = T_self_canonical_partner_canonical_flat.reshape((*size, P-1, 9))
        body_contacts = body_contacts_flat.reshape((*size, 21))
        tpose_mask = tpose_mask_flat.reshape((*size, 1)).bool()
    
        unpacked = cls(
            betas=betas,
            body_joint_rotations=body_joint_rotations,
            hand_joint_rotations=hand_joint_rotations,
            T_canonical_tm1_canonical_t=T_canonical_tm1_canonical_t,
            T_canonical_root=T_canonical_root,
            T_world_root=T_world_root,
            T_world_canonical=T_world_canonical,
            T_self_canonical_partner_canonical=T_self_canonical_partner_canonical,
            body_contacts=body_contacts,
            tpose_mask=tpose_mask,
        )

        return unpacked

    






