import torch
from torch import Tensor
import numpy as np
from jaxtyping import Float

from pathlib import Path
import h5py

from libs.dataloaders import DataType, TrainingData, padding_packed_training_data

def load_from_hdf5(
    data_type: DataType,
    hdf5_path: Path,
    file_list: list[str],
    person_num: int,
    mean_std_file_path: Path | None = None,
    device: torch.device | None = None,
    is_mask_additional_person: bool = True,
) -> dict[str, TrainingData]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mean_std_file_path is not None:
        ms = np.load(mean_std_file_path)
        mean = torch.from_numpy(ms["mean"]).to(torch.float32).to(device)
        std = torch.from_numpy(ms["std"]).to(torch.float32).to(device)

    is_add_person = False
    raw_packed_data_dict: dict[str, Float[Tensor, "T P D"]] = {}
    with h5py.File(hdf5_path, "r") as hdf5_file:
        data_type_str = data_type.name.lower()

        for file_name in file_list:
            ds = hdf5_file[data_type_str][file_name]
            raw_packed_data = torch.as_tensor(ds[()], dtype=torch.float32, device=device)
            # packed_data = raw_packed_data if mean_std_file_path is None else TrainingData.normalize(raw_packed_data, mean, std)
            raw_packed_data_dict[file_name] = raw_packed_data
            is_add_person = raw_packed_data.shape[1] < person_num

    data_dict: dict[str, TrainingData] = {}
    person_num_dict: dict[str, int] = {}
    if is_add_person:
        if is_mask_additional_person:
            for file_name, raw_packed_data in raw_packed_data_dict.items():
                T, P, D = raw_packed_data.shape
                add_P = person_num - P
                packed_data = raw_packed_data if mean_std_file_path is None else TrainingData.normalize(raw_packed_data, mean, std)
                void_self_partner = torch.zeros(T, P, add_P*9, device=device, dtype=packed_data.dtype)
                packed_data = torch.cat([packed_data[:, :, :-22], void_self_partner, packed_data[:, :, -22:]], dim=-1)
                void_data = torch.zeros(T, add_P, D+add_P*9, device=device, dtype=packed_data.dtype)
                void_data[..., -1] = 0
                packed_data = torch.cat([packed_data, void_data], dim=-2)
                data_dict[file_name] = TrainingData.unpack(packed_data)
                person_num_dict[file_name] = P
        else:
            P = raw_packed_data_dict[file_list[0]].shape[1]
            raw_data_dict = {k: TrainingData.unpack(v) for k, v in raw_packed_data_dict.items()}
            partner_num = (person_num - P) // P
            perm_list = torch.randint(0, len(file_list), (len(file_list), partner_num))
            for (file_name, raw_data), perm in zip(raw_data_dict.items(), perm_list):
                raw_data_list = [raw_data_dict[file_list[j]] for j in perm]
                raw_data_list = [raw_data] + raw_data_list
                max_len = max([v.betas.shape[0] for v in raw_data_list])
                padded_raw_data_list = [padding_training_data(data=v, target_len=max_len, mean=mean, std=std, start_end_idx=None)
                          for v in raw_data_list]
                raw_data = synthesize_multi_person_motion(padded_raw_data_list)
                data_dict[file_name] = raw_data if mean_std_file_path is None else TrainingData.normalize_unpacked(raw_data, mean, std)
                person_num_dict[file_name] = P
    else:
        # data_dict = {k: TrainingData.unpack(v) for k, v in packed_data_dict.items()}
        for file_name, raw_packed_data in raw_packed_data_dict.items():
            packed_data = raw_packed_data if mean_std_file_path is None else TrainingData.normalize(raw_packed_data, mean, std)
            data_dict[file_name] = TrainingData.unpack(packed_data)
            person_num_dict[file_name] = person_num

    return data_dict, person_num_dict