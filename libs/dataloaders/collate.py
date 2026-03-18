import torch
from torch import Tensor
from jaxtyping import Float
from typing import Callable
import dataclasses

from libs.dataloaders import TrainingData
from libs.dataloaders import shuffle_person_dim

def collate_dataclass_list(
    person_num: int,
    mean: Float[Tensor, "*dim"],
    std: Float[Tensor, "*dim"],
    shuffle: bool,
    is_mask_additional_person: bool=True, 
) -> Callable[[list[TrainingData]], TrainingData]:
    def _truncate_to_person(td: TrainingData, p: int):
        for f in dataclasses.fields(td):
            v = getattr(td, f.name)
            if f.name == "T_self_canonical_partner_canonical":
                setattr(td, f.name, v[..., :p, :p-1, :])
            elif f.name in ["body_joint_rotations", "hand_joint_rotations"]:
                setattr(td, f.name, v[..., :p, :, :])
            else:
                setattr(td, f.name, v[..., :p, :])
        return td

    def collate_fn(
        batch: list[Float[Tensor, "T P D"]]
    ) -> TrainingData:
        # get person_num
        data_P = torch.tensor([b.shape[-2] for b in batch], dtype=torch.int64)
        add_P = person_num - data_P

        train_data = None
        if torch.all(add_P == 0):
            data = torch.stack(batch, dim=0)
            if shuffle:
                data = shuffle_person_dim(data)
            train_data = TrainingData.unpack(data)

        elif torch.all(add_P <= 0):
            packed_list = []
            for b in batch:
                bb = b
                if shuffle:
                    bb = shuffle_person_dim(bb.unsqueeze(0))[0]
                td = TrainingData.unpack(bb)
                td = _truncate_to_person(td, person_num)
                packed_list.append(td.pack())
            packed = torch.stack(packed_list, dim=0)
            train_data = TrainingData.unpack(packed)

        elif torch.all(add_P >= 0):
            if is_mask_additional_person:
                data_list = []
                max_add = int(add_P.max().item())
                for ap in range(max_add+1):
                    idx = torch.where(add_P == ap)[0]
                    if idx.numel() == 0:
                        continue

                    raw_mb = [batch[int(i)] for i in idx]
                    raw_mb = torch.stack(raw_mb, dim=0)

                    if ap > 0:
                        B, T, P, D = raw_mb.shape
                        void_self_partner = torch.zeros(B, T, P, ap*9)
                        raw_mb = torch.cat([raw_mb[:, :, :, :-22], void_self_partner, raw_mb[:, :, :, -22:]], dim=-1)
                        void_data = torch.rand(B, T, ap, D + ap*9)
                        void_data[..., -1] = 0 # tpose_mask = False
                        raw_mb = torch.cat([raw_mb, void_data], dim=2)

                    if shuffle:
                        raw_mb = shuffle_person_dim(raw_mb)

                    data_list.append(raw_mb)

                data = torch.cat(data_list, dim=0)
                train_data = TrainingData.unpack(data)

            else:
                data_list = []
                max_add = int(add_P.max().item())
                for ap in range(0, max_add + 1):
                    idx = torch.where(add_P == ap)[0]
                    if idx.numel() == 0:
                        continue

                    raw_mb = [batch[int(i)] for i in idx]
                    raw_mb = torch.stack(raw_mb, dim=0)
                    
                    if ap == 0:
                        if shuffle:
                            raw_mb = shuffle_person_dim(raw_mb)
                        data_list.append(raw_mb)

                    else:
                        raw_mb_dn = TrainingData.denormalize(raw_mb, mean, std)
                        seeds = [TrainingData.unpack(raw_mb_dn)]

                        max_pair_num = ap // 2
                        pair_num = torch.randint(0, max_pair_num + 1, (1,)).item()
                        solo_num = ap - pair_num * 2


                        for _ in range(pair_num):
                            cand = [i for i, x in enumerate(batch) if x.shape[-2] == 2]
                            pick = torch.randint(0, len(cand), (raw_mb.shape[0],))
                            pair_mb = torch.stack([batch[cand[j.item()]] for j in pick], dim=0)
                            pair_mb_dn = TrainingData.denormalize(pair_mb, mean, std)
                            seeds.append(TrainingData.unpack(pair_mb_dn))

                        for _ in range(solo_num):
                            cand = [i for i, x in enumerate(batch) if x.shape[-2] == 1]
                            pick = torch.randint(0, len(cand), (raw_mb.shape[0],))
                            solo_mb = torch.stack([batch[cand[j.item()]] for j in pick], dim=0)
                            solo_mb_dn = TrainingData.denormalize(solo_mb, mean, std)
                            seeds.append(TrainingData.unpack(solo_mb_dn))

                        multi = synthesize_multi_person_motion(seeds)
                        multi = TrainingData.normalize(multi.pack(), mean, std)
                        data_list.append(multi)

                packed = torch.cat(data_list, dim=0)
                train_data = TrainingData.unpack(packed)

        else:
            raise ValueError("additional_person_num must be non-negative")

        return train_data

    return collate_fn