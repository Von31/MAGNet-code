import dataclasses
from typing import List, Dict, Optional
from pathlib import Path
import time
from omegaconf import OmegaConf, MISSING
from argparse import ArgumentParser

import numpy as np
import torch
import os

from libs.train.dfot_train import load_cfg
from libs.dataloaders import DataType, TrainingData, load_from_hdf5, padding_training_data
from libs.model.dfot.config import DFOTSamplingConfig
from libs.model.vqvae import pose_vqvae
from libs.model.dfot import network
from libs.utils.random_seed import set_seed
from libs.utils.root_transform_processor import RootTransformProcessor
from libs.utils.transforms import SE3, SO3

import sys

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

@dataclasses.dataclass
class InferenceConfig:
    output_dir: Path = MISSING
    checkpoint_dir: Path = MISSING
    epoch: str = MISSING

    dataset_split: str = MISSING

    result_dir_prefix: str = MISSING

    smpl_path: Path = Path("./data/smplx/SMPLX_NEUTRAL.npz")
    save_motion_data: bool = True
    random_seed: int = 1234

    person_num: int = 1
    is_mask_additional_person: bool = True

    sampling_cfg: DFOTSamplingConfig = MISSING
    def get_default_sampling_cfg():
        return DFOTSamplingConfig(
            task_mode = "joint_prediction",

            # noise schedule
            context_seq_len = 32,
            sampling_schedule = "causal_uncertainty",
            sampling_subseq_len = 16,
            
            sampling_steps = 30, # denoising steps
            sampling_num = 10, # number of samples
            denoising_process = "ddim", # ddim, ddpm
            ddim_eta = 0.0,

            # cfg
            cfg_scale_dict = {
                "clean": 1.0,
            },

            ar_seq_stride = 16, 
            sampling_seq_len = 200,

            root_transform_mode = "temporal",

        )

    save_motion_data: bool = True
    random_seed: int = 1234

    is_person_flip: bool = False

    dataset_path: Path = MISSING
    inference_data_list: List[str] = MISSING



def load_inf_cfg(yaml_path: str = None):
    base = OmegaConf.structured(InferenceConfig)
    if yaml_path is None:
        dataset_path = InferenceConfig.get_default_dataset_path(base.dataset_name)
        inference_data_list = InferenceConfig.get_default_data_list(base.dataset_name, base.dataset_split)
        cfg = OmegaConf.merge(base, {"dataset_path": dataset_path, "inference_data_list": inference_data_list})
    else:
        yml  = OmegaConf.load(yaml_path)
        cfg  = OmegaConf.merge(base, yml)
    
    if cfg.epoch == "best":
        cfg.checkpoint_dir = cfg.checkpoint_dir / "best_checkpoints"
    else:
        epoch = cfg.epoch.replace("K", "000").replace("k", "000")
        cfg.checkpoint_dir = cfg.checkpoint_dir / f"checkpoints_{epoch}"

        if not cfg.checkpoint_dir.exists():
            cfg.checkpoint_dir = cfg.checkpoint_dir.parent / f"save_checkpoints_{epoch}"

    OmegaConf.set_readonly(cfg, True)

    return OmegaConf.to_object(cfg)


def get_model_path(checkpoint_dir: Path) -> tuple[Path, Path]:
    return checkpoint_dir / "model.safetensors", checkpoint_dir / "../config.yaml"


def main(inf_cfg: InferenceConfig) -> None:
    set_seed(inf_cfg.random_seed)
    device = torch.device("cuda")

    model_path, config_path = get_model_path(inf_cfg.checkpoint_dir)
    config = load_cfg(config_path)
    model_module = network.DFOTNetwork
    token_module = pose_vqvae.MultiPoseToken
    model = model_module.load(model_path, config.model_cfg, device)
    
    ms_data = np.load(config.mean_std_path)
    mean = torch.from_numpy(ms_data["mean"]).to(torch.float32).to(device)
    std = torch.from_numpy(ms_data["std"]).to(torch.float32).to(device)


    # load test data
    if inf_cfg.dataset_split == "test":
        data_type = DataType.TEST
    elif inf_cfg.dataset_split == "val":
        data_type = DataType.VAL
    elif inf_cfg.dataset_split == "train":
        data_type = DataType.TRAIN
    hdf5_path = inf_cfg.dataset_path
    file_list = inf_cfg.inference_data_list

    kwargs = {
        "data_type": data_type,
        "hdf5_path": hdf5_path,
        "mean_std_file_path": config.mean_std_path,
        "file_list": file_list,
        "person_num": config.person_num,
        "device": device,
        "is_mask_additional_person": inf_cfg.is_mask_additional_person,
    }
    test_data, test_person_num = load_from_hdf5(**kwargs)

    def create_offset_data(data, offset):
        new_data_dict = {}
        for f in dataclasses.fields(data):
            new_data_dict[f.name] = getattr(data, f.name)[offset:]
        return TrainingData(**new_data_dict)
    def create_flip_data(data):
        new_data_dict = {}
        for f in dataclasses.fields(data):
            new_data_dict[f.name] = getattr(data, f.name)[:, [1, 0]]
        return TrainingData(**new_data_dict)
    additional_data = {}
    additional_person_num = {}
    if inf_cfg.is_person_flip:
        for file_name, data in test_data.items():
            flip_data = create_flip_data(data)
            additional_data[file_name + "_flip"] = flip_data
            additional_person_num[file_name + "_flip"] = test_person_num[file_name]
    test_data.update(additional_data)
    test_person_num.update(additional_person_num)

    print(f'test_data file list: {test_data.keys()}')


    out_dir = None
    if inf_cfg.save_motion_data:
        date_str = time.strftime("%Y%m%d_%H%M%S")        
        experiment_name = inf_cfg.checkpoint_dir.parents[1].name
        out_dir = Path(f"{inf_cfg.output_dir}/{experiment_name}/{inf_cfg.result_dir_prefix}_{inf_cfg.sampling_cfg.sampling_task}_{date_str}")
        os.makedirs(out_dir, exist_ok=True)
        OmegaConf.save(config=inf_cfg, f=Path(f"{out_dir}/inf_cfg.yaml"))

        root_processor = RootTransformProcessor()


    for file_name, raw_gt_motion in test_data.items():
        print(file_name)
        try:
            raw_pred_motion, _ = model.sample_sequence(
                sampling_config=inf_cfg.sampling_cfg,
                data=raw_gt_motion,
            )
        except Exception as e:
            print(f"SKIPPING {file_name}: {e}")
            continue

        if inf_cfg.save_motion_data:
            multi_pred_motion = token_module.denormalize(raw_pred_motion, mean, std)
            gt_timesteps = min(raw_gt_motion.betas.shape[0], raw_pred_motion.betas.shape[1])
            gt_motion = TrainingData.denormalize_unpacked(raw_gt_motion, mean, std)
            gt_motion = padding_training_data(data=gt_motion, target_len=multi_pred_motion.body_joint_rotations.shape[1])
            
            T_world_root = root_processor.convert_root_transform(
                pred_motion=multi_pred_motion,
                mode="temporal",
                context_seq_len=inf_cfg.sampling_cfg.context_seq_len if inf_cfg.sampling_cfg.context_seq_len > 0 else 4,
                gt_motion=gt_motion,
            )
            T_world_root = torch.cat([SE3.from_9d(gt_motion.T_world_root).wxyz_xyz.unsqueeze(0), T_world_root], dim=0)
            body_joint_rotations = torch.cat([SO3.from_6d(gt_motion.body_joint_rotations).wxyz.unsqueeze(0), SO3.from_6d(multi_pred_motion.body_joint_rotations).wxyz], dim=0)
            hand_joint_rotations = torch.zeros(body_joint_rotations.shape[:-2] + (30, 4), dtype=body_joint_rotations.dtype, device=body_joint_rotations.device)
            betas = gt_motion.betas.unsqueeze(0).repeat(1+multi_pred_motion.betas.shape[0], 1, 1, 1)

            person_num = test_person_num[file_name]
            joint_phase_start_timesteps = -1
            if inf_cfg.sampling_cfg.sampling_task == "pp2joint":
                # First frame index of joint continuation (matches partner GT span in
                # _prepare_sampling_data: min(sampling_seq_len, clip_len), 4-frame aligned).
                T_gt_frames = min(
                    inf_cfg.sampling_cfg.sampling_seq_len,
                    int(raw_gt_motion.betas.shape[0]),
                )
                T_gt_frames -= T_gt_frames % 4
                joint_phase_start_timesteps = int(T_gt_frames)

            processed_data = {
                "betas": betas[:, :, :person_num].numpy(force=True),
                "T_world_root": T_world_root[:, :, :person_num].numpy(force=True),
                "body_joint_rotations": body_joint_rotations[:, :, :person_num].numpy(force=True),
                "hand_joint_rotations": hand_joint_rotations[:, :, :person_num].numpy(force=True),
                "context_timesteps": inf_cfg.sampling_cfg.context_seq_len,
                "gt_timesteps": gt_timesteps,
                "timesteps": betas.shape[1],
                "mode": inf_cfg.sampling_cfg.sampling_task,
                "joint_phase_start_timesteps": joint_phase_start_timesteps,
            }

            file_name_ = file_name

            np.savez(
                f"{out_dir}/{file_name_}.npz",
                **processed_data,
            )
            print(f"saved! {out_dir}/{file_name_}.npz")


   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", "--cfg", type=str, default=None)
    args = parser.parse_args()
    inf_cfg = load_inf_cfg(args.config)

    main(inf_cfg)