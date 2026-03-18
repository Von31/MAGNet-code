import torch
import torch.nn.functional as F
from typing import Callable, Literal
from jaxtyping import Float, Int

from torch import Tensor

from einops import rearrange

from libs.model.dfot.config import DFOTConfig, DFOTSamplingConfig

FOOT_JOINT_INDICES = [6, 7, 9, 10]  # left_ankle, right_ankle, left_foot, right_foot


def apply_temporal_smoothing(
    x_pred: Float[Tensor, "S T P D"],
    chunk_idx: int,
    strength: float = 0.3,
    start_step: int = 0,
) -> tuple[Float[Tensor, "S T P D"], float]:
    """
    Post-hoc Laplacian temporal smoothing in token space.

    Blends each interior frame toward the midpoint of its temporal
    neighbours.  Called once after each denoising chunk completes.

    Args:
        x_pred:     Predicted tokens for the chunk  [S, T, P, D].
        chunk_idx:  Autoregressive chunk index (0 = initial window).
        strength:   Blend factor in [0, 1].  0 = no-op, 0.5 = strong.
        start_step: Skip smoothing for chunks before this index.

    Returns:
        (smoothed_x_pred, residual_norm)
    """
    if chunk_idx < start_step:
        return x_pred, 0.0

    T_len = x_pred.shape[1]
    if T_len < 3:
        return x_pred, 0.0

    neighbour_avg = 0.5 * (x_pred[:, :-2] + x_pred[:, 2:])   # [S, T-2, P, D]
    residual = x_pred[:, 1:-1] - neighbour_avg                 # [S, T-2, P, D]
    residual_norm = residual.norm().item()

    x_out = x_pred.clone()
    x_out[:, 1:-1] = x_pred[:, 1:-1] - strength * residual

    return x_out, residual_norm


def compute_smoothing_x0_gradient(
    x0_pred: Float[Tensor, "B T P D"],
    decode_fn: Callable,
    betas: Float[Tensor, "B P 10"],
    pose_latent_dim: int,
) -> Float[Tensor, "B T P D"]:
    """
    Compute smoothing gradient on x0 with two components:
      1. Pose: Laplacian on decoded body_joint_rotations via VQ-VAE decoder.
         Gradient flows through decoder to pose_latent dims.
      2. Transforms: Direct Laplacian on raw transform dims of x0
         (canonical_tm1_t, canonical_self_partner -- no decoding needed).
    """
    x0_input = x0_pred.detach().clone()

    # --- Pose smoothing: through VQ-VAE decoder ---
    with torch.enable_grad():
        x0_grad = x0_input.requires_grad_(True)

        pose_latent = x0_grad[..., :pose_latent_dim]
        canonical_tm1_t = x0_grad[..., pose_latent_dim:pose_latent_dim + 4 * 9]

        B, Tz, P, _ = pose_latent.shape
        dec_cond = torch.cat([
            betas[:, None, :, :].expand(-1, Tz, -1, -1),
            canonical_tm1_t,
        ], dim=-1)

        decoded = decode_fn(pose_latent, dec_cond)

        body_joint_rot = decoded[..., :21 * 6]
        laplacian = body_joint_rot[:, :-2] - 2 * body_joint_rot[:, 1:-1] + body_joint_rot[:, 2:]
        pose_smoothness_loss = (laplacian ** 2).mean()

        pose_grad = torch.autograd.grad(pose_smoothness_loss, x0_grad)[0]

    pose_grad = pose_grad.detach()
    pose_grad[..., pose_latent_dim:] = 0.0

    # --- Transform smoothing: direct Laplacian on raw transform dims ---
    transforms = x0_input[..., pose_latent_dim:]  # [B, T, P, D_transforms]
    transform_lap = transforms[:, :-2] - 2 * transforms[:, 1:-1] + transforms[:, 2:]
    T = transforms.shape[1]
    transform_grad = torch.zeros_like(transforms)
    scale = 2.0 / transform_lap.numel()
    transform_grad[:, :T-2]  += scale * transform_lap
    transform_grad[:, 1:T-1] += scale * (-2.0) * transform_lap
    transform_grad[:, 2:T]   += scale * transform_lap

    # --- Combine ---
    grad = pose_grad
    grad[..., pose_latent_dim:] = transform_grad
    return grad


def compute_foot_skating_x0_gradient(
    x0_pred: Float[Tensor, "B T P D"],
    decode_fn: Callable,
    betas: Float[Tensor, "B P 10"],
    pose_latent_dim: int,
) -> Float[Tensor, "B T P D"]:
    """
    Compute dL_skate/dx0 -- gradient through VQ-VAE decoder only (no transformer).

    Steps:
        1. Extract pose_latent and canonical_tm1_t from x0.
        2. Decode through VQ-VAE decoder to get body_joint_rotations.
        3. Compute temporal velocity on foot joints [6,7,9,10].
        4. Loss = mean(||delta foot_rot||^2).
        5. Return gradient w.r.t. x0.
    """
    x0_input = x0_pred.detach().clone()

    with torch.enable_grad():
        x0_grad = x0_input.requires_grad_(True)

        pose_latent = x0_grad[..., :pose_latent_dim]
        canonical_tm1_t = x0_grad[..., pose_latent_dim:pose_latent_dim + 4 * 9]

        B, Tz, P, _ = pose_latent.shape
        dec_cond = torch.cat([
            betas[:, None, :, :].expand(-1, Tz, -1, -1),
            canonical_tm1_t,
        ], dim=-1)

        decoded = decode_fn(pose_latent, dec_cond)

        body_joint_rot = decoded[..., :21 * 6].reshape(
            *decoded.shape[:3], 21, 6
        )

        foot_rot = body_joint_rot[..., FOOT_JOINT_INDICES, :]  # [B, T', P, 4, 6]
        foot_vel = foot_rot[:, 1:] - foot_rot[:, :-1]          # [B, T'-1, P, 4, 6]
        skating_loss = (foot_vel ** 2).mean()

        grad = torch.autograd.grad(skating_loss, x0_grad)[0]

    grad = grad.detach()
    grad[..., pose_latent_dim:] = 0.0
    return grad


class DFOTClassifierFreeGuidance:
    def __init__(
        self,
        config: DFOTConfig,
        sampling_config: DFOTSamplingConfig,
        prediction_fn: Callable,
        sample_step_fn: Callable,
        q_sample_fn: Callable,
        model_output_process_fn: Callable | None = None,
    ) -> None:
        self.sconfig = sampling_config
        self.config = config

        self.max_noise = config.max_t - 1

        self.is_rearrange_motion_seq = getattr(config, "is_rearrange_motion_seq", None) # token order for dyadic prediction
        self.is_dyadic_pred = (self.is_rearrange_motion_seq is not None)

        self.cfg_scale_dict = getattr(sampling_config, "cfg_scale_dict", {})

        print(f'cfg_scale_dict: {self.cfg_scale_dict}')

        self.prediction_fn = prediction_fn
        self.sample_step_fn = sample_step_fn
        self.q_sample_fn = q_sample_fn
        self.model_output_process_fn = model_output_process_fn

        self.use_smoothing_guidance = getattr(sampling_config, "use_smoothing_guidance", False)
        self.smoothing_weight = getattr(sampling_config, "smoothing_guidance_weight", 1.0)
        self.smoothing_start_ratio = getattr(sampling_config, "smoothing_guidance_start_ratio", 0.0)
        self.smoothing_end_ratio = getattr(sampling_config, "smoothing_guidance_end_ratio", 1.0)
        self.pose_latent_dim: int | None = None

        self.use_foot_skating_guidance = getattr(sampling_config, "use_foot_skating_guidance", False)
        self.foot_skating_weight = getattr(sampling_config, "foot_skating_guidance_weight", 1.0)
        self.foot_skating_start_ratio = getattr(sampling_config, "foot_skating_guidance_start_ratio", 0.0)
        self.foot_skating_end_ratio = getattr(sampling_config, "foot_skating_guidance_end_ratio", 1.0)
        self.decode_fn: Callable | None = None
        self.betas: Float[Tensor, "B P 10"] | None = None

        self._step_counter = 0
        self._total_steps = None

    def set_total_steps(self, total_steps: int):
        self._total_steps = total_steps
        self._step_counter = 0

    def _should_apply_smoothing(self) -> bool:
        if not self.use_smoothing_guidance or self._total_steps is None:
            return self.use_smoothing_guidance
        progress = self._step_counter / max(self._total_steps, 1)
        return self.smoothing_start_ratio <= progress <= self.smoothing_end_ratio

    def _should_apply_foot_skating(self) -> bool:
        if not self.use_foot_skating_guidance or self._total_steps is None:
            return self.use_foot_skating_guidance
        progress = self._step_counter / max(self._total_steps, 1)
        return self.foot_skating_start_ratio <= progress <= self.foot_skating_end_ratio

    def __call__(
        self,
        x_t: Float[Tensor, "B T P D"],
        curr_noise_level: Int[Tensor, "B T P"] | Float[Tensor, "B T P"],
        next_noise_level: Int[Tensor, "B T P"] | Float[Tensor, "B T P"],
        context_token_len: int,
        cond: Float[Tensor, "B P C"] | None = None,
    ) -> Float[Tensor, "B T P D"]:

        # --- standard CFG model output ---
        with torch.inference_mode():
            if context_token_len == 0:
                model_output = self.sample_model_output(
                    mask_mode = "clean",
                    x_t = x_t,
                    curr_noise_level = curr_noise_level,
                    next_noise_level = next_noise_level,
                    context_token_len = context_token_len,
                    cond = cond,
                )
            else:
                model_output_dict = {}
                model_output = torch.zeros_like(x_t)
                for key, value in self.cfg_scale_dict.items():
                    sample_model_output = self.sample_model_output(
                        mask_mode = key,
                        x_t = x_t,
                        curr_noise_level = curr_noise_level,
                        next_noise_level = next_noise_level,
                        context_token_len = context_token_len,
                        cond = cond,
                    )
                    model_output_dict[key] = sample_model_output
                    model_output += value * sample_model_output

        # --- x0-space guidance (post-CFG, outside inference_mode) ---
        apply_smoothing = (
            self._should_apply_smoothing()
            and self.model_output_process_fn is not None
            and self.decode_fn is not None
            and self.betas is not None
        )
        apply_skating = (
            self._should_apply_foot_skating()
            and self.model_output_process_fn is not None
            and self.decode_fn is not None
            and self.betas is not None
        )

        if apply_smoothing or apply_skating:
            processed = self.model_output_process_fn(x_t, curr_noise_level, model_output)
            x0_pred = processed['x_start']
            x0_guided = x0_pred.clone()

            if apply_smoothing:
                smoothing_grad = compute_smoothing_x0_gradient(
                    x0_pred, self.decode_fn, self.betas, self.pose_latent_dim,
                )
                if self._step_counter == 0 or (self._step_counter % 100 == 0):
                    grad_norm = smoothing_grad.norm().item()
                    print(f"[SmoothingGuidance] step={self._step_counter}/{self._total_steps} "
                          f"grad_norm={grad_norm:.6f} weight={self.smoothing_weight}")
                x0_guided = x0_guided - self.smoothing_weight * smoothing_grad

            if apply_skating:
                skating_grad = compute_foot_skating_x0_gradient(
                    x0_pred, self.decode_fn, self.betas, self.pose_latent_dim,
                )
                if self._step_counter == 0 or (self._step_counter % 100 == 0):
                    grad_norm = skating_grad.norm().item()
                    print(f"[FootSkatingGuidance] step={self._step_counter}/{self._total_steps} "
                          f"grad_norm={grad_norm:.6f} weight={self.foot_skating_weight}")
                x0_guided = x0_guided - self.foot_skating_weight * skating_grad

            model_output = self._recompute_model_output(x_t, curr_noise_level, model_output, x0_guided)

        with torch.inference_mode():
            x_t_1_pred = self.sample_step_fn(
                x = x_t,
                curr_noise_level = curr_noise_level,
                next_noise_level = next_noise_level,
                model_output = model_output,
                ddim_eta = self.sconfig.ddim_eta,
            )

        torch.cuda.empty_cache()

        self._step_counter += 1
        return x_t_1_pred

    def _recompute_model_output(self, x_t, curr_noise_level, original_model_output, x0_guided):
        """Recompute model_output so that sample_step sees the guided x0.

        For pred_x0 objective, model_output IS x0, so we just return x0_guided.
        For pred_noise or pred_v, we back-derive the corresponding output.
        """
        objective = self.config.diffusion_objective
        if objective == "pred_x0":
            return x0_guided
        elif objective == "pred_noise":
            return original_model_output
        else:
            return original_model_output

    def sample_model_output(
        self,
        mask_mode: Literal["full", "subset", "partial", "first_person", "second_person", "clean"],
        x_t: Float[Tensor, "B T P D"],
        curr_noise_level: Int[Tensor, "B T P"] | Float[Tensor, "B T P"],
        next_noise_level: Int[Tensor, "B T P"] | Float[Tensor, "B T P"],
        context_token_len: int,
        cond: Float[Tensor, "B P C"] | None = None,
    ) -> Float[Tensor, "B T P D"]:
        (batch, time, person_num, _) = x_t.shape

        if self.sample_step_fn is None or self.q_sample_fn is None:
            raise ValueError("Sampling functions must be set using setup_sampling_functions")

        # get the context mask
        noise_x_t = self._get_noise_xt(
            mask_mode, x_t, curr_noise_level, context_token_len
        )

        # sample the next step
        out_dict = self.prediction_fn(noise_x_t, curr_noise_level, cond)
        model_output = out_dict['model_output']

        return model_output


    def _get_noise_xt(
        self,
        mask_mode: Literal["full", "subset", "partial", "first_person", "second_person", "clean"],
        x_t: Float[Tensor, "B T P D"],
        curr_noise_level: Int[Tensor, "B T P"] | Float[Tensor, "B T P"],
        context_token_len: int,
    ) -> Int[Tensor, "batch timestep person_num"]:
        if mask_mode == "clean":
            return x_t

        (B, T, P) = curr_noise_level.shape
        device = x_t.device
        flat_curr_noise_level = curr_noise_level.clone().reshape(B, T*P)
        flat_x_t = x_t.clone().reshape(B, T*P, -1)
        noise_flat_xt = self.q_sample_fn(
            x_start = torch.zeros_like(flat_x_t),
            k       = flat_curr_noise_level,
            noise   = torch.randn_like(flat_x_t)
        )

        if mask_mode == "full":
            flat_x_t[:, :context_token_len] = noise_flat_xt[:, :context_token_len]

        # mask first half of context tokens
        elif mask_mode == "subset":
            mask_subset_ratio = getattr(self.sconfig, "subset_mask_ratio", 0.5)
            mask_subset_len = int(context_token_len * mask_subset_ratio)
            flat_x_t[:, :mask_subset_len] = noise_flat_xt[:, :mask_subset_len]

        # mask all context with certain noise level
        elif mask_mode == "partial":
            mask_noise_level = getattr(self.sconfig, "partial_mask_noise_level", 0.5)
            flat_x_t[:, :context_token_len] = mask_noise_level * noise_flat_xt[:, :context_token_len] + (1 - mask_noise_level) * flat_x_t[:, :context_token_len]

        # mask all tokens of a specific person — use max noise level so the model
        # sees that person as fully noisy (unconditional), regardless of curr_noise_level
        elif mask_mode[:6] == "person":
            person_id = int(mask_mode[6])
            assert person_id < P, f"person_id must be less than {P}"
            person_idx = torch.arange(person_id, T*P, P, device=device)
            max_noise_level = torch.full_like(flat_curr_noise_level, self.max_noise)
            person_noise = self.q_sample_fn(
                x_start=torch.zeros_like(flat_x_t),
                k=max_noise_level,
                noise=torch.randn_like(flat_x_t),
            )
            flat_x_t[:, person_idx] = person_noise[:, person_idx]

        elif mask_mode[:7] == "~person":
            person_id = int(mask_mode[7])
            assert person_id < P, f"person_id must be less than {P}"
            mask = torch.ones(flat_curr_noise_level.shape[1], dtype=torch.bool, device=flat_curr_noise_level.device)
            mask[torch.arange(person_id, T*P, P, device=device)] = False
            max_noise_level = torch.full_like(flat_curr_noise_level, self.max_noise)
            person_noise = self.q_sample_fn(
                x_start=torch.zeros_like(flat_x_t),
                k=max_noise_level,
                noise=torch.randn_like(flat_x_t),
            )
            flat_x_t[:, mask] = person_noise[:, mask]

        else:
            raise ValueError(f"Invalid mask mode: {mask_mode}")

        new_x_t = flat_x_t.reshape(B, T, P, -1)
        return new_x_t
