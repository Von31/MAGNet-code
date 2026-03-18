import dataclasses
from typing import Dict, List
from pathlib import Path
from omegaconf import MISSING
from libs.model.vqvae.config import VQVAEConfig

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
    max_t: int = 1000 # discrete diffusion timesteps
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

    without_self_partner: bool = False



@dataclasses.dataclass(frozen=False)
class DFOTConfig(DFOTBaseConfig):
    loss_func: str = "smooth_l1"
    loss_weight: Dict[str, float] | None = dataclasses.field(default_factory=lambda: {
        "pose_latent": 1.0,
    })

    vqvae_model_path: Path = MISSING
    vqvae_cfg: VQVAEConfig = MISSING

    smpl_model_path: Path = Path("./data/smplx/SMPLX_NEUTRAL.npz")
    mean_std_path: Path = MISSING
    warmup_steps: int = MISSING

    time_embedding_mode: str = "rope"
    
@dataclasses.dataclass(frozen=False)
class DFOTSamplingConfig:
    config_name: str = "dfot_sampling_config"

    sampling_schedule: str = "full_sequence" #["full_sequence", "autoregressive", "causal_uncertainty"]
    sampling_task: str = "joint" #["joint", "next_token", "agentic_turn_taking","agentic_sync", "partner_inpainting", "partner_prediction", "motion_control", "inbetweening"]
      
    context_seq_len: int = 32
    sampling_subseq_len: int = 16
    ar_seq_stride: int = 8
    sampling_seq_len: int = 200

    sampling_num: int = 10
    sampling_steps: int = 30
    denoising_process: str = "ddim" #["ddim", "ddpm"]
    ddim_eta: float = 0.0

    cfg_scale_dict: Dict[str, float] = dataclasses.field(
    default_factory = lambda: {
        "clean": 1.0,
    })
    partial_mask_noise_level: float = 0.5
    
    inbetweening_key_frame_indices: List[int] = dataclasses.field(default_factory=lambda: [0, 63])
    
    offset_proportion : float = 0.5

    init_noise: str = "random"

    



    # Temporal smoothing (post-hoc Laplacian smoothing after each chunk)
    use_temporal_smoothing: bool = False
    temporal_smoothing_strength: float = 0.3
    temporal_smoothing_start_step: int = 0

    # Re-orthogonalize transforms after each AR chunk
    is_update_history_transforms: bool = False

    # x₀-space smoothing guidance (gradient through VQ-VAE decoder)
    use_smoothing_guidance: bool = False
    smoothing_guidance_weight: float = 1.0
    smoothing_guidance_start_ratio: float = 0.0
    smoothing_guidance_end_ratio: float = 1.0
    smoothing_guidance_pose_latent_only: bool = False

    # x₀-space foot skating guidance (gradient through VQ-VAE decoder)
    use_foot_skating_guidance: bool = False
    foot_skating_guidance_weight: float = 1.0
    foot_skating_guidance_start_ratio: float = 0.0
    foot_skating_guidance_end_ratio: float = 1.0

    root_transform_mode: str = "temporal" #["temporal", "temporal_partner"]
    
    

