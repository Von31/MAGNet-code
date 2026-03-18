import torch
from torch import Tensor
import numpy as np
from typing import List, Dict
from jaxtyping import Float
import dataclasses

from libs.dataloaders import TrainingData, StdMeanIdx
from libs.utils.transforms import SE3

def padding_training_data(
    data: TrainingData,
    target_len: int,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    start_end_idx: tuple[int, int] | None = None,
) -> TrainingData:
    identity_9d = SE3.identity(data.betas.device, data.betas.dtype).as_9d()
    if mean is not None and std is not None:
        idx = StdMeanIdx().T_canonical_tm1_canonical_t
        identity_9d = (identity_9d - mean[idx]) / std[idx]

    for field in dataclasses.fields(data):
        v = getattr(data, field.name)

        if start_end_idx is not None:
            v = v[start_end_idx[0]:start_end_idx[1]]

        if v.shape[0] < target_len:
            shortfall = target_len - v.shape[0]
            # pad zeros for tpose_mask
            if field.name == "tpose_mask":
                zero_mask = torch.zeros((shortfall, *v.shape[1:]), dtype=v.dtype, device=v.device)
                v = torch.cat([v, zero_mask], axis=0)

            # pad identity for T_canonical_tm1_canonical_t
            elif field.name == "T_canonical_tm1_canonical_t":
                identity_mask = identity_9d.clone().reshape((1,)*len(v.shape[1:])+(-1,))
                identity_mask = identity_mask.repeat((shortfall,) + v.shape[1:-1] + (1,))
                v = torch.cat([v, identity_mask], axis=0)

            # pad last motion for betas, joint_rotations, contacts, T_world_root, T_world_canonical, T_self_canonical_partner_canonical
            else:
                stop_motion = v[-1].unsqueeze(0).repeat((shortfall,) + (1,) *len(v.shape[1:]))
                v = torch.cat([v, stop_motion], axis=0)

        elif v.shape[0] > target_len:
            v = v[:target_len]
        else:
            pass
            
        setattr(data, field.name, v)

    return data

def padding_packed_training_data(
    data: Float[Tensor, "T P D"],
    padding_len: int,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    is_zero_padding: bool =True,
) -> Float[Tensor, "T P D"]:
    T, P, D = data.shape
    if is_zero_padding:
        zero_mask = torch.zeros((padding_len, P, D), dtype=data.dtype, device=data.device)
        padding_data = torch.cat([zero_mask, data, zero_mask.clone()], dim=0)
    else:
        padding_data = torch.empty(T+padding_len*2, P, D, device=data.device, dtype=data.dtype)
        padding_data[:padding_len] = data[0].unsqueeze(0).expand(padding_len, P, D)
        padding_data[padding_len:T+padding_len] = data
        padding_data[T+padding_len:] = data[-1].unsqueeze(0).expand(padding_len, P, D)

        idx = StdMeanIdx()

        # T_canonical_tm1_canonical_t
        identity_9d = torch.array([1, 0, 0, 0, 0, 0, 0], dtype=data.dtype, device=data.device)
        if mean is not None and std is not None:
            identity_9d = (identity_9d - mean[idx.T_canonical_tm1_canonical_t]) / std[idx.T_canonical_tm1_canonical_t]
        padding_data[:padding_len, :, idx.T_canonical_tm1_canonical_t] = identity_9d[None, None, :].expand(padding_len, P, 9)
        padding_data[T+padding_len:, :, idx.T_canonical_tm1_canonical_t] = identity_9d[None, None, :].expand(padding_len, P, 9)

        # tpose_mask
        zero_mask = torch.zeros((padding_len, P), dtype=data.dtype, device=data.device)

        padding_data[:padding_len, :, -1] = zero_mask
        padding_data[T+padding_len:, :, -1] = zero_mask
    return padding_data

def shuffle_person_dim_dict(
    data_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    betas = data_dict["betas"]
    *B, T, P, _ = betas.shape
    pdim = len(B) + 1
    device = betas.device

    # Random permutation
    perms = torch.rand(*B, P, device=device).argsort(dim=-1)  # [B, P]

    processed_dict = {}
    for k, v in data_dict.items():
        # this func does not support T_self_canonical_partner_canonical
        if k == 'T_self_canonical_partner_canonical':
            raise NotImplementedError
        else:
            org_shape = v.shape
            v = v.reshape(*B, T, P, -1)
            v = v.gather(dim=pdim, index=perms[..., None, :, None].expand(*B, T, P, v.shape[-1]))
            processed_dict[k] = v.reshape(*org_shape)

    return processed_dict

_shuffle_cache = {}
def _partner_tables(P, device):
    key = (P, device.type, device.index)
    if key in _shuffle_cache: return _shuffle_cache[key]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    partners = torch.arange(P, device=device)[None, :].expand(P, P)
    partners = partners[~eye].view(P, P-1)
    old2pos = torch.full((P, P), -1, device=device, dtype=torch.long)
    pos = torch.arange(P-1, device=device, dtype=torch.long).expand(P, -1)
    old2pos.scatter_(1, partners, pos)
    _shuffle_cache[key] = (eye, old2pos)
    return _shuffle_cache[key]
    
def _partner_order_from_perms(perms):
    B, P = perms.shape
    device = perms.device
    eye, old2pos = _partner_tables(P, device)
    perms_tile = perms[:, None, :].expand(B, P, P)
    newpartner_old = perms_tile[~eye.expand(B, -1, -1)].view(B, P, P-1)
    rows = old2pos[perms]
    return torch.gather(rows, 2, newpartner_old)
    

def shuffle_person_dim(
    data: Float[Tensor, "B T P D"],
) -> Float[Tensor, "B T P D"]:
    B, T, P, D = data.shape
    if P <= 1: 
        return data

    device = data.device
    perms = torch.rand(B, P, device=device).argsort(dim=-1)
    data = data.gather(dim=2, index=perms[:, None, :, None].expand(B, T, P, D))

    if P == 2: 
        return data

    if P > 2:
        order = _partner_order_from_perms(perms)
        sp = data[..., -(P-1)*9-22:-22].view(B, T, P, P-1, 9)
        sp = sp.gather(dim=3, index=order[:, None, :, :, None].expand(B, T, P, P-1, 9))
        data[..., -(P-1)*9-22:-22] = sp.reshape(B, T, P, (P-1)*9)

    return data

def synthesize_multi_person_motion(
    dataset_list: List[TrainingData],
) -> TrainingData:
    *B, T, P, D = dataset_list[0].betas.shape
    device = dataset_list[0].betas.device
    dtype = dataset_list[0].betas.dtype
    size = (*B, 1, len(dataset_list))
    pdim = len(B) + 1

    # create random translation
    ## random translation, R [0.0, dataset_num], theta [0, 2pi]
    trans_theta = torch.rand(size, dtype=dtype, device=device) * np.pi * 2.
    trans_rad_r = torch.rand(size, dtype=dtype, device=device) * float(len(dataset_list)) * 1.0
    trans = torch.stack(
        (trans_rad_r * torch.cos(trans_theta), torch.zeros_like(trans_rad_r), trans_rad_r * torch.sin(trans_theta)),
        dim=-1
    )

    ## random rotation, theta [0, 2pi]
    rot_theta = torch.rand(size, dtype=dtype, device=device) * np.pi * 2.
    rot = SO3.from_y_radians(rot_theta).wxyz
    T_new_world_to_old_world = torch.cat([rot, trans], dim=-1).to(device)

    person_num_list = torch.tensor([d.betas.shape[-2] for d in dataset_list], device=device)
    T_new_world_to_old_world = torch.repeat_interleave(T_new_world_to_old_world, person_num_list, dim=pdim)

    # randomize person's order
    whole_data_dict = {
        k.name: torch.cat([getattr(d, k.name) for d in dataset_list], dim=pdim)
        for k in dataclasses.fields(dataset_list[0])
        if k.name != 'T_self_canonical_partner_canonical'
    }
    # whole_data_dict['T_new_world_to_old_world'] = T_new_world_to_old_world.expand(-1, T, -1, -1)
    whole_data_dict['T_new_world_to_old_world'] = T_new_world_to_old_world.expand(*B, T, -1, -1)
    whole_data_dict = shuffle_person_dim_dict(whole_data_dict)
    T_new_world_to_old_world = whole_data_dict.pop('T_new_world_to_old_world')

    T_world_canonical_7d = (SE3(T_new_world_to_old_world) @ SE3.from_9d(whole_data_dict['T_world_canonical'])).wxyz_xyz
    T_self_canonical_partner_canonical_ = (SE3(T_world_canonical_7d[..., :, None, :]).inverse() @ SE3(T_world_canonical_7d[..., None, :, :])).wxyz_xyz
    T_self_canonical_partner_canonical_list = []
    total_P = int(person_num_list.sum().item())
    for i in range(total_P):
        T_self_canonical_partner_canonical_list.append(T_self_canonical_partner_canonical_[..., i, torch.arange(total_P) != i, :])
    T_self_canonical_partner_canonical = torch.stack(T_self_canonical_partner_canonical_list, dim=-3)
    
    whole_data_dict['T_self_canonical_partner_canonical'] = SE3(T_self_canonical_partner_canonical).as_9d()
    whole_data_dict['T_world_canonical'] = SE3(T_world_canonical_7d).as_9d()
    whole_data_dict['T_world_root'] = (SE3(T_new_world_to_old_world) @ SE3.from_9d(whole_data_dict['T_world_root'])).as_9d()

    whole_data = TrainingData(**whole_data_dict)

    return whole_data
