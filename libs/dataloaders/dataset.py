import torch
import numpy as np
import yaml
import h5py
from pathlib import Path
from typing import Literal, cast

from libs.dataloaders import TrainingData, DataType, padding_packed_training_data 

class Hdf5Dataset(torch.utils.data.Dataset[TrainingData]):
    def __init__(
        self,
        data_type: DataType,
        hdf5_path: Path,
        file_list_path: Path,
        mean_std_path: Path,
        cache_files: bool,
        subseq_len: int,
        slice_method: Literal["deterministic", "random_uniform_len", "random_variable_len", "first"] = "first",
        random_variable_len_min: int = 10,
        random_variable_len_proportion: float = 0.3,
        cache_size_limit_gb: float = 10.,
        is_first_clean: bool = False,
    ) -> None:
        self._data_type = data_type.name.lower()
        self._hdf5_path = hdf5_path
        self._subseq_len = subseq_len
        self._slice_method = slice_method
        self._random_variable_len_min = random_variable_len_min
        self._random_variable_len_proportion = random_variable_len_proportion
        self._is_first_clean = is_first_clean

        mean_std = np.load(mean_std_path)
        self._mean = torch.from_numpy(mean_std["mean"]).to(torch.float32)
        self._std = torch.from_numpy(mean_std["std"]).to(torch.float32)

        with open(file_list_path, 'r') as f:
            groups = yaml.safe_load(f)


        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            self._groups = [
                p
                for p in groups[self._data_type]
            ]
            assert len(self._groups) > 0

            if self._slice_method == "deterministic":
                self._index_list = [
                    (cast(h5py.Dataset, hdf5_file[self._data_type][g]).shape[0] // self._subseq_len + 1)
                    for g in self._groups
                ]
                self._index_list = np.cumsum(self._index_list)
                self._approximated_length = self._index_list[-1]
            elif self._slice_method == "random_uniform_len":
                # self._index_list = [
                #     max(cast(h5py.Dataset, hdf5_file[self._data_type][g]).shape[0] - self._subseq_len + 1, 1)
                #     for g in self._groups
                # ]
                # self._index_list = np.cumsum(self._index_list)
                # self._approximated_length = self._index_list[-1]
                self._index_list = [
                    max(cast(h5py.Dataset, hdf5_file[self._data_type][g]).shape[0] // self._subseq_len + 1, 1)
                    for g in self._groups
                ]
                self._index_list = np.cumsum(self._index_list)
                self._approximated_length = self._index_list[-1]
            else:
                self._index_list = None
                self._approximated_length = len(self._groups)

        self._cache: dict[str, dict[str, Any]] | None = {} if cache_files else None
        self._cache_size_bytes: int = 0
        self._cache_size_limit_bytes: int = self._to_bytes(cache_size_limit_gb)
        self._h5: h5py.File | None = None

    def _to_bytes(self, limit_gib: float) -> int:
        GiB = 1024 ** 3
        return int(limit_gib * GiB)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._hdf5_path, "r")


    def __getitem__(self, index: int):
        if self._slice_method in ["deterministic", "random_uniform_len"]:
            group_index = np.searchsorted(self._index_list, index, side="right")
            slice_index = index
            if group_index > 0:
                slice_index = index - self._index_list[group_index - 1]
        else:
            group_index = index % len(self._groups)
        del index

        group = self._groups[group_index]
        hdf5_file = None

        if self._cache is not None:
            if group not in self._cache:
                self._ensure_open()
                ds = self._h5[self._data_type][group]
                raw_np = ds[()]
                if raw_np.dtype != np.float32:
                    raw_np = raw_np.astype(np.float32, copy=False)
                raw_packed_data = torch.as_tensor(raw_np)
                packed_data = TrainingData.normalize(raw_packed_data, self._mean, self._std)
                packed_data = padding_packed_training_data(packed_data, self._subseq_len, self._mean, self._std)
                # packed_data = padding_packed_training_data(raw_packed_data, self._subseq_len)
                self._cache_size_bytes += packed_data.nbytes
                if self._cache_size_bytes < self._cache_size_limit_bytes:
                    self._cache[group] = packed_data
            else:
                packed_data = self._cache[group]
        else:
            self._ensure_open()
            ds = self._h5[self._data_type][group]
            raw_np = ds[()]
            raw_packed_data = torch.as_tensor(raw_np, dtype=torch.float32)
            packed_data = TrainingData.normalize(raw_packed_data, self._mean, self._std)
            packed_data = padding_packed_training_data(packed_data, self._subseq_len, self._mean, self._std)
            # packed_data = padding_packed_training_data(raw_packed_data, self._subseq_len)

        # total_t = packed_data.shape[0] - self._subseq_len
        total_t = packed_data.shape[0] - 2*self._subseq_len

        if self._slice_method == "deterministic":
            # A deterministic, non-overlapping slice.
            valid_start_indices = max(total_t - self._subseq_len + 1, 1)
            # start_t = (slice_index * self._subseq_len) % valid_start_indices
            start_t = (slice_index * self._subseq_len) % valid_start_indices + self._subseq_len
            end_t = start_t + self._subseq_len

        elif self._slice_method == "random_uniform_len":
            # start_t = torch.randint(0, max(total_t-int(self._subseq_len*0.75)+1, 1), (1,)).item()
            if self._is_first_clean:
                s_offset = self._subseq_len
            else:
                s_offset = int(self._subseq_len*0.5)
            e_offset = max(total_t + int(self._subseq_len*0.5), s_offset+1)
            start_t = torch.randint(s_offset, e_offset, (1,)).item()
            end_t = start_t + self._subseq_len

        elif self._slice_method == "first":
            # Use the first subsequence.
            start_t = 0
            end_t = self._subseq_len

        else:
            raise ValueError(f"Invalid slice strategy: {self._slice_method}")


        training_data = packed_data[start_t:end_t]

        if hdf5_file is not None:
            hdf5_file.close()

        return training_data
    

    def __len__(self) -> int:
        return self._approximated_length

class InterleavedDataset(torch.utils.data.Dataset[TrainingData]):
    def __init__(
        self, 
        datasets: list[torch.utils.data.Dataset[TrainingData]], 
        probs: list[float] | None = None,
        dataset_length: int | None = None,
    ) -> None:
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.cum  = np.array([sum(self.lens[:i+1]) for i in range(len(self.lens))])
        # self.probs = probs or [l/sum(self.lens) for l in self.lens]
        self.probs = probs
        self.dataset_length = dataset_length if probs is not None else None
        print("InterleavedDataset:", sum(self.lens), self.lens, self.probs, self.dataset_length)

    def __len__(self) -> int:
        if self.dataset_length is not None:
            return self.dataset_length
        return sum(self.lens)

    def __getitem__(self, idx: int) -> TrainingData:
        # optional: sample *which* dataset at random instead of by idx
        if self.probs is None:
            ds_id = self.cum.searchsorted(idx, side="right")
            sample_idx = idx - self.cum[ds_id-1] if ds_id > 0 else idx
        else:
            ds_id = random.choices(range(len(self.datasets)), weights=self.probs)[0]
            sample_idx = random.randrange(len(self.datasets[ds_id]))
        return self.datasets[ds_id][sample_idx]