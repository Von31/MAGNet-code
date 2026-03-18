import dataclasses
from typing import List, Dict
from pathlib import Path
import time

from yaml import Token
from omegaconf import OmegaConf, MISSING
from argparse import ArgumentParser

import numpy as np
import torch
import os
from libs.utils import fncsmpl

from libs.train.vqvae_train import load_cfg
from libs.dataloaders import DataType, TrainingData, load_from_hdf5
from libs.model.vqvae import pose_vqvae, network
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

    root_transform_mode: str = "temporal"
    person_num: int = 1
    is_mask_additional_person: bool = True


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

    OmegaConf.set_readonly(cfg, True)

    return OmegaConf.to_object(cfg)


def get_model_path(checkpoint_dir: Path) -> tuple[Path, Path]:
    return checkpoint_dir / "model.safetensors", checkpoint_dir / "../config.yaml"


def main(inf_cfg: InferenceConfig) -> None:
    set_seed(inf_cfg.random_seed)
    device = torch.device("cuda")

    model_path, config_path = get_model_path(inf_cfg.checkpoint_dir)
    config = load_cfg(config_path)
    network_module = network.PoseNetwork
    token_module = pose_vqvae.PoseToken
    model = network_module.load(model_path, config.model_cfg, device)
    
    body_model = fncsmpl.SmplModel.load(inf_cfg.smpl_path).to(device)

    ms_data = np.load(config.mean_std_path)
    mean = torch.from_numpy(ms_data["mean"]).to(torch.float32).to(device)
    std = torch.from_numpy(ms_data["std"]).to(torch.float32).to(device)


    # load test data
    if inf_cfg.dataset_split == "test":
        data_type = DataType.TEST
    elif inf_cfg.dataset_split == "train":
        data_type = DataType.TRAIN
    hdf5_path = inf_cfg.dataset_path
    file_list = inf_cfg.inference_data_list

    kwargs = {
        "data_type": data_type,
        "hdf5_path": hdf5_path,
        "mean_std_file_path": config.mean_std_path,
        "file_list": file_list,
        "person_num": inf_cfg.person_num,
        "device": device,
        "is_mask_additional_person": inf_cfg.is_mask_additional_person,
    }
    test_data, _ = load_from_hdf5(**kwargs)

    print(f'test_data file list: {test_data.keys()}')

    out_dir = None
    if inf_cfg.save_motion_data:
        date_str = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = inf_cfg.checkpoint_dir.parents[1].name
        out_dir = Path(f"{inf_cfg.output_dir}/{experiment_name}/{inf_cfg.result_dir_prefix}_{date_str}")
        os.makedirs(out_dir, exist_ok=True)
        OmegaConf.save(config=inf_cfg, f=Path(f"{out_dir}/inf_cfg.yaml"))

        root_processor = RootTransformProcessor()


    pred_dict: dict[str, List[pose_module]] = {}
    gt_motion_dict = {}
    for file_name, raw_gt_motion in test_data.items():
        for field in dataclasses.fields(raw_gt_motion):
            raw_data = getattr(raw_gt_motion, field.name)
            # T = config.sequence_len
            T = (raw_data.shape[0] // 4) * 4
            setattr(raw_gt_motion, field.name, raw_data[:T].unsqueeze(0))

        raw_pred_motion = model.inference(raw_gt_motion)

        if inf_cfg.save_motion_data:           
            pred_motion = token_module.denormalize(raw_pred_motion, mean, std)
            multi_pred_motion = pred_motion.convert_to_mutli_pose_token(inf_cfg.person_num)
            gt_motion = TrainingData.denormalize_unpacked(raw_gt_motion, mean, std)
            
            T_world_root = root_processor.convert_root_transform(
                pred_motion=multi_pred_motion,
                mode=inf_cfg.root_transform_mode,
                context_seq_len=0,
                gt_motion=gt_motion,
            )
            T_world_root = torch.cat([SE3.from_9d(gt_motion.T_world_root).wxyz_xyz, T_world_root], dim=0)
            body_joint_rotations = torch.cat([SO3.from_6d(gt_motion.body_joint_rotations).wxyz, SO3.from_6d(multi_pred_motion.body_joint_rotations).wxyz], dim=0)
            hand_joint_rotations = torch.zeros(body_joint_rotations.shape[:-2] + (30, 4), dtype=body_joint_rotations.dtype, device=body_joint_rotations.device)
            betas = gt_motion.betas.expand(2, -1, -1, -1)

            processed_data = {
                "betas": betas.numpy(force=True),
                "T_world_root": T_world_root.numpy(force=True),
                "body_joint_rotations": body_joint_rotations.numpy(force=True),
                "hand_joint_rotations": hand_joint_rotations.numpy(force=True),
                "context_timesteps": 0,
                "timesteps": betas.shape[1],
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