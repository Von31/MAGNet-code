import dataclasses
from typing import Dict
from pathlib import Path
from omegaconf import MISSING

@dataclasses.dataclass(frozen=False)
class VAEBaseConfig():
    activation: str = "gelu"
    layers: int = 3
    d_latent: int = 126
    d_feedforward: int = 512
    dropout_p: float = 0.05

@dataclasses.dataclass(frozen=False)
class VAEConfig(VAEBaseConfig):
    loss_func: str = "smoth_l1"
    mean_std_path: Path = MISSING

    base_beta: float = 0.2
    warmup_steps: int = 10000
    free_bits_tau: float = 0.

    loss_weight: Dict[str, float] | None = dataclasses.field(default_factory=lambda: {
        "body_joint_rotations": 1.0,
        "canonical_root_rotations": 1.0,
        "canonical_root_translations": 1.0,
        "canonical_tm1_t_rotations": 1.0,
        "canonical_tm1_t_translations": 1.0,
    })

@dataclasses.dataclass(frozen=False)
class VQVAEBaseConfig():
    activation: str = "gelu"
    norm: str = "LN"
    depth: int = 3
    d_latent: int = 512
    d_hidden: int = 512
    code_nb: int = 1024

    without_cond: bool = False

    is_rand_init_code: bool = False
    is_set_r: bool = False
    r_min: float|None = None
    r_max: float|None = None
    alive_thresh: float = 1.0

    down_t: int = 2
    stride_t: int = 2
    dilation_growth_rate: int = 3
    
    ema_mu: float = 0.99
    is_add_last_ln: bool = False

    is_noise_augment: bool = False
    noise_augment_std: float = 0.02

@dataclasses.dataclass(frozen=False)
class VQVAEConfig(VQVAEBaseConfig):
    loss_func: str = "mse"
    smpl_model_path: Path = Path("./body_model/smplx/SMPLX_NEUTRAL.npz")
    mean_std_path: Path = MISSING

    base_beta: float = 0.2
    warmup_steps: int = 10000
    free_bits_tau: float = 0.

    loss_weight: Dict[str, float] | None = dataclasses.field(default_factory=lambda: {
        "body_joint_rotations": 1.0,
        "canonical_root_rotations": 1.0,
        "canonical_root_translations": 1.0,
        "commit_loss": 0.3,
    })
    

@dataclasses.dataclass(frozen=False)
class DFOTBaseConfig():
    person_num: int = 1
    diffusion_objective: str = "pred_x0"
    
    context: str = "none"
    context_sequence_length: int = 32
    variable_context_prob: float = 0.5
    context_dropout_prob: float = 0.1
    sequence_length: int = 64

    transformer_type: str = "original"
    noise_level: str = "random_independent"
    future_noise_uniform: bool = False
    max_t: int = 1000 # for discrete-time diffusion
    diffusion_noise_rand_type: str = "uniform"
    diffusion_noise_logsnr_mu: float = -0.1
    diffusion_noise_logsnr_sigma: float = 1.0
    
    person_embedding_mode: str = "add"
    is_min_snr_weight: bool = False

    activation: str = "gelu"

    layers: int = 6
    d_latent: int = 512
    d_feedforward: int = 2048
    num_heads: int = 8
    dropout_p: float = 0.05



@dataclasses.dataclass(frozen=False)
class VQVAEDFOTConfig(DFOTBaseConfig): #(DFOTConfig):
    loss_func: str = "smooth_l1"

    loss_weight: Dict[str, float] | None = dataclasses.field(default_factory=lambda: {
        "pose_latent": 1.0,
    })

    vqvae_model_path: Path = MISSING
    vqvae_cfg: VQVAEConfig = MISSING

    smpl_model_path: Path = Path("./body_model/smplx/SMPLX_NEUTRAL.npz")
    mean_std_path: Path = MISSING
    warmup_steps: int = MISSING