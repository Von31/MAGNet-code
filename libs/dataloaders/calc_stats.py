from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
from typing import List

from libs.dataloaders import TrainingData

def calculate_mean_std(
    hdf5_file_list: List[Path],
    save_path: Path,
) -> None:
    data = []
    data_person_num = []

    data_dim = 10 + 51 * 6 + 9 + 9 + 9 + 9
    ex_data_dim = 9
    data = np.empty((100000000, data_dim))
    ex_data = np.empty((400000000, ex_data_dim))
    last_idx = 0
    last_ex_idx = 0

    for hdf5_path in hdf5_file_list:
        print(f"Processing {hdf5_path}...")
        hdf5_file = h5py.File(hdf5_path, "r")

        # for data_type in list(hdf5_file.keys()):
        for data_type in ['train']:
            print(f"Processing {data_type} partition...")
            curr_data_dim = None

            for file in tqdm(list(hdf5_file[data_type].keys())):
                file_data = np.array(hdf5_file[data_type][file])

                if curr_data_dim is None:
                    person_num = file_data.shape[-2]
                    curr_data_dim = TrainingData.get_packed_dim(person_num=person_num)

                file_data = file_data[::10]                
                file_data = file_data.reshape(-1, curr_data_dim)
                next_idx = last_idx + file_data.shape[0]
                data[last_idx:next_idx] = file_data[:, :data_dim]
                next_ex_idx = last_ex_idx + file_data.shape[0]*(person_num-1)
                ex_data[last_ex_idx:next_ex_idx] = file_data[:, data_dim:data_dim + ex_data_dim*(person_num-1)].reshape(-1, ex_data_dim)
                last_idx = next_idx
                last_ex_idx = next_ex_idx
                
            print(f"total data_size {last_idx}, ex_data_size {last_ex_idx}")


        if hdf5_file is not None:
            hdf5_file.close()

    data = data[:last_idx]
    ex_data = ex_data[:last_ex_idx]

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    mean_ = np.mean(ex_data, axis=0)
    std_ = np.std(ex_data, axis=0)

    mean = np.concatenate([mean, mean_], axis=0)
    std = np.concatenate([std, std_], axis=0)

    # set std to 1e-2 if it is too small
    std[std<1e-2] = 1e-2

    np.savez(save_path, mean=mean, std=std)
