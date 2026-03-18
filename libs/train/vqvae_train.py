from pathlib import Path
from typing import List
import time
import dataclasses
import shutil

import tensorboardX
import torch
import numpy as np
import math
from loguru import logger
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
import torch

from libs.utils import training_utils
from libs.dataloaders import DatasetName, DataType, Hdf5Dataset, InterleavedDataset, collate_dataclass_list
from libs.model.vqvae import config, network
from libs.utils.random_seed import set_seed
import wandb

from argparse import ArgumentParser
from omegaconf import OmegaConf, MISSING

WANDB_CFG_PATH = Path("configs/wandb.yaml")
if WANDB_CFG_PATH.exists():
    _wandb_cfg = OmegaConf.load(WANDB_CFG_PATH)
    USE_WANDB = _wandb_cfg.get("enabled", False)
else:
    _wandb_cfg = None
    USE_WANDB = False


def get_wandb_logger(configs):
    config_optimizer={
      "learning_rate": configs.learning_rate,
      "batch_size": configs.batch_size,
      "weight_decay": configs.weight_decay,
      "warmup_steps": configs.warmup_steps,
      "max_grad_norm": configs.max_grad_norm,
      "slice_method": configs.slice_method,
      "notes": "Bob is the reference frame"}
    config_dict_model = {k: v for k, v in configs.model_cfg.__dict__.items()}
    config_dict = {**config_dict_model, **config_optimizer}
    wandb.init(project=_wandb_cfg.get("project_name", configs.wandb_project_name),
                entity=_wandb_cfg.get("entity_name", configs.wandb_entity_name),
                id=configs.experiment_name,
                config= config_dict
      )

@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    experiment_dir: str = 'experiments'
    experiment_name: str = 'test'

    dataset_dir: Path = Path('./data/multi_smplx/')
    dataset_list: List[DatasetName] = dataclasses.field(default_factory=lambda: [
        DatasetName.DUOBOX,
    ])
    dataset_list_prob: List[float] | None = None
    dataset_length: int | None = None
    is_skip_val: bool = False

    mean_std_path: Path = Path('./data/multi_smplx/interaction_data_mean_std.npz')

    pretrained_model_path: Path | None = None

    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    total_steps: int = 300000
    min_lr_ratio: float = 0.05
    max_grad_norm: float = 0.5
    
    random_seed: int = 1111

    num_workers: int = 0
    sequence_len: int = 64
    slice_method: str = "deterministic"
    is_mask_additional_person: bool = True

    cache_size_limit_gb_train: float = 10.
    cache_size_limit_gb_val: float = 1.

    wandb_project_name: str = "multi_dfot"
    wandb_entity_name: str = "kieh_workspace"

    model_cfg: config.VQVAEConfig = MISSING

    def _default_model_cfg():
        return config.VQVAEConfig(
            warmup_steps=TrainingConfig.warmup_steps,
            mean_std_path=TrainingConfig.mean_std_path,
        )



def load_cfg(yaml_path: str = None):
    base = OmegaConf.structured(TrainingConfig)
    if yaml_path is None:
        model_cfg = TrainingConfig._default_model_cfg()
        cfg = OmegaConf.merge(base, {"model_cfg": model_cfg})
    else:
        yml  = OmegaConf.load(yaml_path)
        cfg  = OmegaConf.merge(base, yml)

    # OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)

    return OmegaConf.to_object(cfg)


def get_experiment_dir(experiment_parent_dir: str, experiment_name: str, version: int = 0) -> Path:
    """Creates a directory to put experiment files in, suffixed with a version
    number. Similar to PyTorchf lightning."""
    experiment_dir = (
        Path.cwd()
        / experiment_parent_dir
        / experiment_name
        / f"v{version}"
    )
    if experiment_dir.exists():
        return get_experiment_dir(experiment_parent_dir, experiment_name, version + 1)
    else:
        return experiment_dir
    
def get_model_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "model.safetensors", checkpoint_dir / "../model_config.yaml"

def run_training(
    config: TrainingConfig,
    restore_checkpoint_dir: Path | None = None,
) -> None:
    if restore_checkpoint_dir is not None:
        print(f"Restoring from {restore_checkpoint_dir}")
        config = load_cfg(restore_checkpoint_dir.parent / "config.yaml")

    set_seed(config.random_seed)

    experiment_dir = get_experiment_dir(config.experiment_dir, config.experiment_name)
    assert not experiment_dir.exists()
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir=str(experiment_dir)),
        dataloader_config=DataLoaderConfiguration(split_batches=True),
    )
    writer = (
        tensorboardX.SummaryWriter(logdir=str(experiment_dir), flush_secs=10)
        if accelerator.is_main_process
        else None
    )
    device = accelerator.device
    if USE_WANDB:
        get_wandb_logger(config)

    # Setup.
    network_class = network.PoseNetwork

    if config.pretrained_model_path is not None and restore_checkpoint_dir is None:
        prt_model_path = config.pretrained_model_path / "model.safetensors"
        model = network_class.load(prt_model_path, config.model_cfg, device)
    else:
        model = network_class(config.model_cfg, device)
    
    if accelerator.is_main_process:
        training_utils.pdb_safety_net()

        experiment_dir.mkdir(exist_ok=True, parents=True)
        (experiment_dir / "git_commit.txt").write_text(
            training_utils.get_git_commit_hash()
        )
        (experiment_dir / "git_diff.txt").write_text(training_utils.get_git_diff())
        OmegaConf.save(config=config, f=experiment_dir / "config.yaml", resolve=False)

        # Add hyperparameters to TensorBoard.
        assert writer is not None
        writer.add_hparams(
            hparam_dict=training_utils.flattened_hparam_dict_from_dataclass(config),
            metric_dict={},
            name=".",  # Hack to avoid timestamped subdirectory.
        )

        # Write logs to file.
        logger.add(experiment_dir / "trainlog.log", rotation="100 MB")
    

    dataset_module = Hdf5Dataset
    ms = np.load(config.mean_std_path)
    mean = torch.from_numpy(ms['mean']).to(torch.float32)
    std = torch.from_numpy(ms['std']).to(torch.float32)
    train_collate_fn = collate_dataclass_list(
        person_num=1, mean=mean, std=std, shuffle=True, is_mask_additional_person=config.is_mask_additional_person)
    val_collate_fn = collate_dataclass_list(
        person_num=1, mean=mean, std=std, shuffle=False, is_mask_additional_person=config.is_mask_additional_person)

    dataset_kwargs = {
        "mean_std_path": config.mean_std_path,
        "cache_files": True,
        "subseq_len": config.sequence_len,
        "slice_method": config.slice_method,
    }

    train_dataset_kwargs = dataset_kwargs.copy()
    train_dataset_kwargs['data_type'] = DataType.TRAIN
    train_dataset_kwargs['cache_size_limit_gb'] = config.cache_size_limit_gb_train / float(len(config.dataset_list))
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs['data_type'] = DataType.VAL
    val_dataset_kwargs['slice_method'] = "deterministic"
    val_dataset_kwargs['cache_size_limit_gb'] = config.cache_size_limit_gb_val / float(len(config.dataset_list))

    train_dataset_list = []
    val_dataset_list = []
    for dataset_name in config.dataset_list:
        hdf5_path = config.dataset_dir / f'{dataset_name.name.lower()}_dataset.hdf5'
        file_list_path = config.dataset_dir / f'{dataset_name.name.lower()}_dataset_files.txt'
        train_dataset_list.append(dataset_module(hdf5_path=hdf5_path, file_list_path=file_list_path, **train_dataset_kwargs))
        if not config.is_skip_val:
            val_dataset_list.append(dataset_module(hdf5_path=hdf5_path, file_list_path=file_list_path, **val_dataset_kwargs))
    train_dataset = InterleavedDataset(train_dataset_list, probs=config.dataset_list_prob, dataset_length=config.dataset_length)
    if not config.is_skip_val:
        val_dataset = InterleavedDataset(val_dataset_list)
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        # pin_memory=False,
        collate_fn=train_collate_fn,
    )
    if config.is_skip_val:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=True,
            # pin_memory=False,
            collate_fn=val_collate_fn,
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def cosine_with_warmup(total_steps, warmup_steps, min_lr_ratio=0.05):
        def fn(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step > total_steps:
                return min_lr_ratio
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return fn

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=cosine_with_warmup(total_steps=config.total_steps, warmup_steps=config.warmup_steps, min_lr_ratio=config.min_lr_ratio)
    )

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optim, lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
    # )
    model, train_loader, val_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, val_loader, optim, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)
    if restore_checkpoint_dir is not None:
        print(f"Loading checkpoint from {restore_checkpoint_dir}")
        accelerator.load_state(restore_checkpoint_dir)
    
    initial_step = int(scheduler.state_dict()["last_epoch"])
    accelerator.save_state(str(experiment_dir / f"checkpoints_{initial_step}"))

    loop_metrics_gen = training_utils.loop_metric_generator(counter_init=initial_step)
    prev_checkpoint_path: Path | None = None
  
    max_patience = 10000
    early_stop = False
    best_val = float('inf')
    while True:
        patience = 0
        for train_batch in train_loader:
            try:
                  loop_metrics = next(loop_metrics_gen)
                  step = loop_metrics.counter
            except StopIteration:
                  logger.error("loop_metrics_gen generator exhausted unexpectedly")
                  break
            except Exception as e:
                  logger.error(f"Error in loop_metrics generation: {str(e)}")
                  break
            step = loop_metrics.counter

            model.train()
            loss, loss_log = model.training_step(train_batch, step)

            log_outputs = {'loss': loss.item()}
            log_outputs["learning_rate"] = scheduler.get_last_lr()[0]
            accelerator.log(log_outputs, step=step)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            if not accelerator.is_main_process:
                continue

          # Logging.
            if step % 10 == 0:
                assert writer is not None
                for k, v in log_outputs.items():
                    writer.add_scalar(k, v, step)

            # Print status update to terminal.
            if step % 20 == 0:
                # validation
                val = torch.zeros_like(loss)
                val_log = {}
                if val_loader is not None:
                    model.eval()
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_, val_log_ = model.training_step(val_batch, step)

                            val += val_ * len(val_batch)
                            if val_log_ is not None:
                                if val_log == {}:
                                    for k in val_log_:
                                        val_log[k] = 0.
                                for k, v in val_log_.items():
                                    val_log[k] += v * len(val_batch)

                        val /= len(val_loader.dataset)
                        for k in list(val_log.keys()):
                            val_log[k+"_val"] = val_log.pop(k)/len(val_loader.dataset)

                mem_free, mem_total = torch.cuda.mem_get_info()
                log_msg = (
                    f"step: {step} ({loop_metrics.iterations_per_sec:.2f} it/sec)"
                    f" mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G"
                    f" lr: {scheduler.get_last_lr()[0]:.7f}"
                    f" loss: {loss.item():.6f}"
                )
                wandb_log = {"step": step, "loss": loss.item()}
                if val_loader is not None:
                    log_msg += f" val: {val.item():.6f}"
                    wandb_log["val"] = val.item()
                logger.info(log_msg)
                # wandb.log({"step": step, "loss": loss.item(), "val": val.item()})

                if loss_log is not None:
                    print("loss:", end=" ")
                    for k, v in loss_log.items():
                        print(f"{k}: {v:.6f}", end=", ")
                    if val_log:
                        print("\nval :", end=" ")
                        for k, v in val_log.items():
                            print(f"{k[:-4]}: {v:.6f}", end=", ")
                    print("")
                    wandb_log.update(loss_log)
                    wandb_log.update(val_log)
                if USE_WANDB:
                    wandb.log(wandb_log)
            
                #break training and save checkpoint if val is >best_val
                patience += 1
                if val_loader is not None and val.item() < best_val and step > 1000:
                    patience = 0
                    best_val = val.item()
                    checkpoint_path = experiment_dir / f"best_checkpoints"
                    accelerator.save_state(str(checkpoint_path))
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                if val_loader is not None and patience > max_patience:
                    logger.info(f"Early stopping triggered after {patience} validation checks")
                    early_stop = True  # Set flag to break while loop
                    break
            
            # Checkpointing.
            if step > initial_step and step % 5000 == 0:
                # Save checkpoint.
                checkpoint_path = experiment_dir / f"checkpoints_{step}"
                accelerator.save_state(str(checkpoint_path))
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
                # Keep checkpoints from only every 100k steps.
                if prev_checkpoint_path is not None:
                    shutil.rmtree(prev_checkpoint_path)
                prev_checkpoint_path = None if step % 50_000 == 0 else checkpoint_path
                del checkpoint_path

            # debugs = time.time()

            # if early_stop:  # Check flag after for loop
            #     break
        

if __name__ == "__main__":
    # tyro.cli(run_training)
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--config", "--cfg", type=str, default=None)
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else None

    config = load_cfg(args.config)
    run_training(config, restore_checkpoint_dir=checkpoint_dir)
    if USE_WANDB:
        wandb.finish()