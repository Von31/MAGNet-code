from jaxtyping import Float, Bool
from typing import Literal, Tuple, Dict, List
from functools import partial
from pathlib import Path
import dataclasses

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange, reduce
from safetensors.torch import safe_open
from tqdm import tqdm

from libs.model.dfot.config import DFOTBaseConfig, DFOTConfig
from libs.model.dfot.config import DFOTSamplingConfig
from libs.dataloaders import StdMeanIdx, TrainingData
from libs.model.vqvae.pose_vqvae import PoseToken, MultiPoseToken
from libs.model.vqvae.network import PoseNetwork
from libs.model.dfot.diffusion_transformer import TransformerModel, MotionToken
from libs.model.dfot.diffusion import DiscreteDiffusion
from libs.model.dfot.dfot_guidance import DFOTClassifierFreeGuidance, apply_temporal_smoothing
from libs.utils.noise_schedule import make_logsnr_rand_fn
from libs.utils.fncsmpl import SmplModel
from libs.utils.root_transform_processor import RootTransformProcessor
from libs.utils.transforms import SO3, SE3


VQVAE_TEMPORAL_FACTOR: int = 4   # VQ-VAE temporal down-sampling ratio
SE3_9D_DIM: int = 9              # Dimension of the SE(3) 9-D representation
FOOT_JOINT_INDICES: List[int] = [6, 7, 9, 10]  # Foot joint indices in SMPL-X




class DFOTBase(nn.Module):
    def __init__(self, config: DFOTBaseConfig, device: torch.device):
        super(DFOTBase, self).__init__()
        self.config = config
        self.device = device
        self.init_model()

    def init_model(self):
        self.diffuser = DiscreteDiffusion(self.config, TransformerModel(self.config))
        if self.config.diffusion_noise_rand_type == "uniform":
            self.rand_fn = partial(torch.randint, 0, self.config.max_t)
        elif self.config.diffusion_noise_rand_type == "logsnr":
            self.rand_fn = make_logsnr_rand_fn(
                self.diffuser.alphas_cumprod, self.config.diffusion_noise_logsnr_mu, self.config.diffusion_noise_logsnr_sigma)
        self.max_noise = self.config.max_t - 1

    @classmethod
    def load(
        cls,
        model_path: Path,
        config: DFOTBaseConfig,
        device: torch.device,
    ) -> "DFOTBase":
        model = cls(config, device)
        with safe_open(model_path, framework="pt") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        for k in model.state_dict().keys():
            if k not in state_dict or model.state_dict()[k].shape != state_dict[k].shape:
                print(f"Key {k} not found in state dict or shape mismatch")
                state_dict[k] = model.state_dict()[k]
        model.load_state_dict(state_dict)
        model.to(device)
        return model

    # ------------------------------------------------------------------ #
    #  Shared helpers to reduce duplication across sampling methods        #
    # ------------------------------------------------------------------ #

    def _create_guidance_sampler(
        self,
        sconfig: DFOTSamplingConfig,
        betas: Float[Tensor, "B P 10"] | None,
        total_steps: int,
    ) -> DFOTClassifierFreeGuidance:
        """Create and configure a guidance sampler (reused by every sampling method)."""
        guidance_sampler = DFOTClassifierFreeGuidance(
            config=self.config,
            sampling_config=sconfig,
            prediction_fn=self.diffuser.model_predictions,
            sample_step_fn=self.diffuser.sample_step,
            q_sample_fn=self.diffuser.q_sample,
            model_output_process_fn=self.diffuser.model_output_process,
        )
        guidance_sampler.set_total_steps(total_steps)
        guidance_sampler.pose_latent_dim = self.d_latent
        self._setup_guidance_decode(guidance_sampler, betas)
        return guidance_sampler

    def _pad_context(
        self,
        context: Float[Tensor, "B Tc P D"],
        context_mask: Bool[Tensor, "B Tc P"],
        T: int,
        motion_tokens_gt: Float[Tensor, "B Tgt P D"] | None = None,
    ):
        """Pad context, context_mask (and optionally motion_tokens_gt) from Tc to T."""
        S, Tc, P, D = context.shape
        padding = T - Tc
        assert padding > 0, f"context_seq_len {Tc} must be smaller than sampling_seq_len {T}"

        context = torch.cat([context, torch.zeros(S, padding, P, D, device=self.device)], dim=1)
        context_mask = torch.cat([
            context_mask,
            torch.zeros(S, padding, P, device=self.device, dtype=torch.bool),
        ], dim=1)

        if motion_tokens_gt is not None:
            T_gt = motion_tokens_gt.shape[1]
            if T > T_gt:
                motion_tokens_gt = torch.cat([
                    motion_tokens_gt,
                    torch.zeros(S, T - T_gt, P, D, device=self.device),
                ], dim=1)
            return context, context_mask, motion_tokens_gt

        return context, context_mask

    @staticmethod
    def _update_context_mask(
        context_mask: Bool[Tensor, "B T P"],
        curr_noise_level: Float[Tensor, "B T P"],
        Ts: int,
        Tcc: int,
        Tm: int,
    ):
        """Mark tokens as context when their noise level becomes -1."""
        context_mask[:, Ts + Tcc:Ts + Tm] = torch.where(
            torch.logical_and(
                context_mask[:, Ts + Tcc:Ts + Tm] == False,
                curr_noise_level[:, Tcc:] == -1,
            ),
            True,
            context_mask[:, Ts + Tcc:Ts + Tm],
        )

    def _setup_ar_params(self, sconfig: DFOTSamplingConfig, Tc: int, P: int, T_gt: int | None = None):
        """Compute common AR iteration parameters.

        Returns:
            (Tm, ar_stride, ar_itrs, T, ar_token_stride, Tca)
        """
        Tm = self.config.sequence_length // 4
        org_sampling_len = max((T_gt if T_gt is not None else sconfig.sampling_seq_len) // 4, Tm)
        ar_stride = sconfig.ar_seq_stride // 4
        ar_itrs = int(np.ceil((org_sampling_len - Tm) / ar_stride))
        T = ar_itrs * ar_stride + Tm
        ar_token_stride = ar_stride * P
        Tca = Tm - ar_token_stride
        return Tm, ar_stride, ar_itrs, T, ar_token_stride, Tca

    def _setup_rolling_params(self, sconfig: DFOTSamplingConfig, Tc: int, T_gt: int):
        """Compute rolling-window parameters.

        Returns:
            (Tm, stride, n_chunks, T)
        """
        Tm = self.config.sequence_length // 4
        stride = Tm - Tc
        n_chunks = max(1, int(np.ceil((T_gt - Tm) / stride)) + 1) if T_gt > Tm else 1
        T = (n_chunks - 1) * stride + Tm if n_chunks > 1 else max(T_gt, Tm)
        return Tm, stride, n_chunks, T

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def _training_step(
        self,
        x: Float[Tensor, "B T P D"],
        mask: Float[Tensor, "B T P"] | None = None,
        **kwargs
    ) -> Tuple[Float[Tensor, "B T P D"], Float[Tensor, "B T P"]]:
        B, T, P, D = x.shape

        noise_level, mask = self._get_training_noise_level(
            (B, T, P), mask, context=self.config.context)
        output = self.diffuser.forward(x, noise_level)

        model_output, target, snr_weight = output['model_output'], output['target'], output['snr_weight']
        kwargs["x"] = x
        kwargs["x_start_pred"] = output["x_start_pred"]
        loss, loss_mask, loss_weight = self._calc_loss(
            model_output=model_output, target=target, **kwargs)
        loss, loss_log = self._reweight_loss(loss, loss_mask, loss_weight, mask, snr_weight)

        return loss, loss_log

    def _get_training_noise_level(
        self,
        size: Tuple[int, int, int],
        mask: Float[Tensor, "B T P"] | None = None,
        context: Literal["fixed", "variable", "none"] = "fixed",
    ) -> Tuple[Float[Tensor, "B T P"], Float[Tensor, "B T P"]]:
        B, T, P = size
        if mask is None:
            mask = torch.ones(B, T, P, device=self.device, dtype=torch.bool)

        context_len = self.config.context_sequence_length
        context_mask = None
        if context == "fixed":
            context_mask = torch.zeros(B, T, P, device=self.device, dtype=torch.bool)
            context_mask[:, :context_len] = True
        elif context == "variable":
            context_len = torch.randint(int(context_len * self.config.variable_context_prob), context_len, (B,), device=self.device)
            context_mask = torch.arange(T, device=self.device).unsqueeze(0) < context_len.unsqueeze(1)  # (B, T)
            context_mask = context_mask.unsqueeze(-1).expand(-1, -1, P)
        elif context == "none":
            pass
        else:
            raise ValueError(f"Invalid context type: {context}")

        if self.config.noise_level == "random_independent":
            noise_level = self.rand_fn((B, T, P)).to(self.device)
        elif self.config.noise_level == "random_uniform":
            noise_level = self.rand_fn((B, 1, 1)).to(self.device).repeat(1, T, P)
        else:
            raise ValueError(f"Invalid noise level type: {self.config.noise_level}")

        noise_level = torch.where(
            reduce(mask.bool(), 'b t p ... -> b t p', 'any'),
            noise_level,
            torch.full_like(noise_level, self.max_noise)
        )

        if context_mask is not None:
            dropout = self.config.context_dropout_prob
            drop = torch.bernoulli(torch.ones_like(context_mask).to(torch.float32) * dropout) * self.max_noise
            noise_level = torch.where(context_mask, drop, noise_level)
            mask = torch.where(context_mask, False, mask)

        noise_level = noise_level.long()
        return noise_level, mask

    def _calc_loss(
        self,
        model_output: Float[Tensor, "B T P D"],
        target: Float[Tensor, "B T P D"],
        **kwargs
    ) -> Tuple[Float[Tensor, "B T P"], Dict[str, Float[Tensor, "B T P"]], Dict[str, float]]:
        raise NotImplementedError

    def _reweight_loss(
        self,
        loss: Dict[str, Float[Tensor, "B T P"]],
        loss_mask: Dict[str, Float[Tensor, "B T P"] | None],
        loss_weight: Dict[str, float],
        mask: Bool[Tensor, "B T P"] | None = None,
        snr_weight: Float[Tensor, "B T P 1"] | None = None,
    ) -> Tuple[Float[Tensor, ""], Dict[str, float]]:
        w = torch.ones_like(next(iter(loss.values())))
        if mask is not None:
            w = w * mask.to(w.dtype)
        if self.config.is_min_snr_weight and snr_weight is not None:
            w = w * snr_weight[..., 0].to(w.dtype)

        logs = {}
        final_loss = 0.
        for k, lk in loss.items():
            if k not in loss_weight:
                logs[k] = lk
                continue

            w_eff = w
            if loss_mask[k] is not None:
                w_eff = w_eff * loss_mask[k].to(w.dtype)

            denom = w_eff.sum().clamp_min(1e-8)
            lk_val = (lk * w_eff).sum() / denom
            logs[k] = lk_val.item()
            final_loss = final_loss + loss_weight[k] * lk_val

        return final_loss, logs

    
    # ------------------------------------------------------------------ #
    #  Sampling methods                                                   #
    # ------------------------------------------------------------------ #

    def _ar_sample_sequence(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
    ):
        sconfig = sampling_config
        S, Tc, P, D = context.shape

        Tm, ar_stride, ar_itrs, T, ar_token_stride, Tca = self._setup_ar_params(sconfig, Tc, P)

        context, context_mask = self._pad_context(context, context_mask, T)

        # Create initial xs_pred with noise
        xs_pred = torch.randn(S, T, P, D, device=self.device)
        xs_pred = torch.where(context_mask[..., None], context, xs_pred)

        # Generate sampling schedules
        noise_sch_matrix = self._create_schedule_matrix(
            sampling_schedule=sconfig.sampling_schedule,
            sampling_steps=sconfig.sampling_steps,
            token_len=Tm * P,
            sub_token_len=1 if sconfig.sampling_schedule == "autoregressive" else sconfig.sampling_subseq_len * P,
            offset_proportion=sconfig.offset_proportion,
        )
        noise_sch_matrix = noise_sch_matrix.to(self.device)
        noise_sch_matrix[:, :Tc * P] = -1
        noise_sch_matrix = self.prune_noise_sch_matrix(noise_sch_matrix)

        ar_noise_sch_matrix = self._create_ar_schedule_matrix(ar_token_stride, noise_sch_matrix)

        sampling_time = noise_sch_matrix.shape[0] - 1
        ar_sampling_time = ar_noise_sch_matrix.shape[0] - 1
        whole_sampling_time = sampling_time + ar_sampling_time * ar_itrs

        pbar = tqdm(total=whole_sampling_time, desc="Sampling")
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        # Temporal smoothing config
        use_temporal_smoothing = sconfig.use_temporal_smoothing
        smoothing_strength = sconfig.temporal_smoothing_strength
        smoothing_start_step = sconfig.temporal_smoothing_start_step

        def sampling_step(
            Ts: int,
            Tcc: int,
            curr_noise_level: Float[Tensor, "S T P"],
            next_noise_level: Float[Tensor, "S T P"],
        ):
            partial_curr_xs_pred = xs_pred[:, Ts:Ts + Tm]

            self._update_context_mask(context_mask, curr_noise_level, Ts, Tcc, Tm)

            partial_xs_pred_next = guidance_sampler(
                x_t=partial_curr_xs_pred,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                context_token_len=Tcc * P,
            )

            xs_pred[:, Ts:Ts + Tm] = torch.where(
                context_mask[:, Ts:Ts + Tm, :, None] == False,
                partial_xs_pred_next,
                xs_pred[:, Ts:Ts + Tm],
            )

        self.diffuser.eval()
        with torch.no_grad():
            Ts = 0

            for m in range(sampling_time):
                curr_noise_level = noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                next_noise_level = noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                sampling_step(Ts, Tc, curr_noise_level, next_noise_level)
                pbar.update(1)

            if use_temporal_smoothing:
                xs_pred[:, Ts:Ts + Tm], _ = apply_temporal_smoothing(
                    xs_pred[:, Ts:Ts + Tm], 0,
                    strength=smoothing_strength, start_step=smoothing_start_step)

            if sconfig.is_update_history_transforms:
                xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
                    xs_pred[:, Ts:Ts + Tm], Tc, sconfig.root_transform_mode)

            for ar in range(ar_itrs):
                Ts = Ts + ar_stride
                for m in range(ar_sampling_time):
                    curr_noise_level = ar_noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level = ar_noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step(Ts, Tca, curr_noise_level, next_noise_level)
                    pbar.update(1)

                if use_temporal_smoothing:
                    xs_pred[:, Ts:Ts + Tm], _ = apply_temporal_smoothing(
                        xs_pred[:, Ts:Ts + Tm], ar + 1,
                        strength=smoothing_strength, start_step=smoothing_start_step)

                if sconfig.is_update_history_transforms:
                    xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
                        xs_pred[:, Ts:Ts + Tm], Tca, sconfig.root_transform_mode)

        return xs_pred

    def prune_noise_sch_matrix(self, noise_sch_matrix: Float[Tensor, "steps tokens"]):
        diff = noise_sch_matrix[1:] - noise_sch_matrix[:-1]
        row_same = (diff == 0).all(dim=1)
        skip = (~row_same).float().argmax().item() if (~row_same).any() else 0
        noise_sch_matrix = noise_sch_matrix[skip:]
        return noise_sch_matrix

    def generate_noise_sch_matrix(self, sconfig: DFOTSamplingConfig, token_len: int, context_token_len: int, P: int = 2):
        noise_sch_matrix = self._create_schedule_matrix(
            sampling_schedule=sconfig.sampling_schedule,
            sampling_steps=sconfig.sampling_steps,
            token_len=token_len,
            sub_token_len=sconfig.sampling_subseq_len,
            offset_proportion=sconfig.offset_proportion,
        )
        noise_sch_matrix = noise_sch_matrix.to(self.device)
        noise_sch_matrix[:, :context_token_len] = -1
        noise_sch_matrix = self.prune_noise_sch_matrix(noise_sch_matrix)
        return noise_sch_matrix

    def _ar_agentic_turn_taking_sequence(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
        gt_seq_len: int = None,
    ):
        sconfig = sampling_config
        S, Tc, P, D = context.shape

        Tm, ar_stride, ar_itrs, T, ar_token_stride, Tca = self._setup_ar_params(sconfig, Tc, P, T_gt=gt_seq_len)

        context, context_mask = self._pad_context(context, context_mask, T)
        context_mask_A = context_mask
        context_mask_B = torch.clone(context_mask_A)

        xs_pred_A = torch.randn(S, T, P, D, device=self.device)
        xs_pred_B = torch.randn(S, T, P, D, device=self.device)
        xs_pred_A = torch.where(context_mask_A[..., None], context, xs_pred_A)
        xs_pred_B = torch.where(context_mask_B[..., None], context, xs_pred_B)

        noise_sch_matrix_A = self.generate_noise_sch_matrix(sconfig, token_len=Tm * P, context_token_len=Tc * P, P=P)
        noise_sch_matrix_B = self.generate_noise_sch_matrix(sconfig, token_len=Tm * P, context_token_len=Tc * P, P=P)

        ar_noise_sch_matrix_A = self._create_ar_schedule_matrix(ar_token_stride, noise_sch_matrix_A)
        ar_noise_sch_matrix_B = self._create_ar_schedule_matrix(ar_token_stride, noise_sch_matrix_B)

        sampling_time = noise_sch_matrix_A.shape[0] - 1
        ar_sampling_time = ar_noise_sch_matrix_A.shape[0] - 1
        whole_sampling_time = sampling_time + ar_sampling_time * ar_itrs

        pbar = tqdm(total=whole_sampling_time, desc="Sampling")
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        def sampling_step_tt(
            Ts: int,
            Tcc: int,
            curr_noise_level_A: Float[Tensor, "S T P"],
            next_noise_level_A: Float[Tensor, "S T P"],
            curr_noise_level_B: Float[Tensor, "S T P"],
            next_noise_level_B: Float[Tensor, "S T P"],
        ):
            """Denoise Agent A, then use its output as context for Agent B."""
            nonlocal context_mask_A, context_mask_B, xs_pred_A, xs_pred_B

            S_, T_, P_, D_ = xs_pred_A.shape
            xs_pred_A_ = xs_pred_A.reshape(S_, T_ * P_, D_)
            xs_pred_B_ = xs_pred_B.reshape(S_, T_ * P_, D_)
            end_token = (Ts + Tm) * P_
            start_token = Ts * P_
            start_context = (Ts + Tcc) * P_

            partial_curr_xs_pred_A = xs_pred_A_[:, start_token:end_token].reshape(S_, -1, P_, D_)

            curr_noise_level_A_ = curr_noise_level_A.clone().reshape(S_, -1)
            next_noise_level_A_ = next_noise_level_A.clone().reshape(S_, -1)
            curr_noise_level_B_ = curr_noise_level_B.clone().reshape(S_, -1)
            next_noise_level_B_ = next_noise_level_B.clone().reshape(S_, -1)

            context_mask_A_ = context_mask_A.clone().reshape(S_, -1)

            # Update context mask for Agent A with the denoised tokens
            new_context_mask = context_mask_A_.clone()
            new_context_mask[:, start_context:end_token] = torch.where(
                torch.logical_and(
                    context_mask_A_[:, start_context:end_token] == False,
                    curr_noise_level_A_[:, P_ * Tcc:] == -1,
                ),
                True,
                context_mask_A_[:, start_context:end_token],
            )

            context_mask_A_ = new_context_mask.clone()
            context_mask_B_ = new_context_mask.clone()

            # Step 1: Sample next token At for Agent A (leader)
            partial_xs_pred_next_A = guidance_sampler(
                x_t=partial_curr_xs_pred_A,
                curr_noise_level=curr_noise_level_A,
                next_noise_level=next_noise_level_A,
                context_token_len=Tcc * P,
            )

            xs_pred_A_[:, start_token:end_token] = torch.where(
                context_mask_A_[:, start_token:end_token, None] == False,
                partial_xs_pred_next_A.reshape(S_, -1, D_),
                xs_pred_A_[:, start_token:end_token],
            )

            # Step 2: Agentic B Prediction (use leader's A_t as context for B_t)
            curr_pointer_A = torch.where(context_mask_B_[:, start_context:end_token] == True)[-1]
            pointer_A = curr_pointer_A.max().item() if curr_pointer_A.numel() > 0 else 0

            partial_curr_xs_pred_B = xs_pred_B_[:, start_token:end_token]

            if pointer_A > 0:
                xs_pred_B_[:, start_token:end_token][:, pointer_A] = xs_pred_A_[:, start_token:end_token][:, pointer_A]
                partial_curr_xs_pred_B = xs_pred_B_[:, start_token:end_token]

                curr_noise_level_B_[:, pointer_A] = -1
                next_noise_level_B_[:, pointer_A] = -1

                context_mask_B_[:, pointer_A] = True
                context_mask_B_[:, pointer_A + 1] = False

            partial_curr_xs_pred_B = partial_curr_xs_pred_B.reshape(S_, -1, P_, D_)

            # Step 3: Sample next token Bt for Agent B (follower)
            partial_xs_pred_next_B = guidance_sampler(
                x_t=partial_curr_xs_pred_B,
                curr_noise_level=curr_noise_level_B_.reshape(S_, -1, P_),
                next_noise_level=next_noise_level_B_.reshape(S_, -1, P_),
                context_token_len=Tcc * P,
            )

            xs_pred_B[:, Ts:Ts + Tm] = torch.where(
                context_mask_B[:, Ts:Ts + Tm, :, None] == False,
                partial_xs_pred_next_B,
                xs_pred_B[:, Ts:Ts + Tm],
            )

            # Update context mask for Agent B
            context_mask_B[:, Ts + Tcc:Ts + Tm] = torch.where(
                torch.logical_and(
                    context_mask_B[:, Ts + Tcc:Ts + Tm] == False,
                    curr_noise_level_B[:, Tcc:] == -1,
                ),
                True,
                context_mask_B[:, Ts + Tcc:Ts + Tm],
            )

            # Update Bt in Agent A xs_pred with the denoised tokens Bt from Agent B
            xs_pred_B_ = xs_pred_B.reshape(S_, -1, D_)
            xs_pred_A_ = xs_pred_A.reshape(S_, -1, D_)
            xs_pred_A_[:, start_token:end_token][:, pointer_A + 1] = xs_pred_B_[:, start_token:end_token][:, pointer_A + 1]

            xs_pred_A = xs_pred_A_.reshape(S_, T_, P_, D_)
            xs_pred_B = xs_pred_B_.reshape(S_, T_, P_, D_)

        self.diffuser.eval()
        with torch.no_grad():
            Ts = 0

            for m in range(sampling_time):
                curr_noise_level_A = noise_sch_matrix_A[m].reshape(1, Tm, P).expand(S, Tm, P)
                next_noise_level_A = noise_sch_matrix_A[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                curr_noise_level_B = noise_sch_matrix_B[m].reshape(1, Tm, P).expand(S, Tm, P)
                next_noise_level_B = noise_sch_matrix_B[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                sampling_step_tt(Ts, Tc, curr_noise_level_A, next_noise_level_A, curr_noise_level_B, next_noise_level_B)
                pbar.update(1)

            for ar in range(ar_itrs):
                Ts = Ts + ar_stride
                for m in range(ar_sampling_time):
                    curr_noise_level_A = ar_noise_sch_matrix_A[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level_A = ar_noise_sch_matrix_A[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    curr_noise_level_B = ar_noise_sch_matrix_B[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level_B = ar_noise_sch_matrix_B[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step_tt(Ts, Tca, curr_noise_level_A, next_noise_level_A, curr_noise_level_B, next_noise_level_B)
                    pbar.update(1)

        return xs_pred_A

    def _ar_sample_sequence_motion_control(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
        motion_tokens_gt: Float[Tensor, "B T P D"] | None = None,
    ):
        sconfig = sampling_config
        S, Tc, P, D = context.shape

        Tm, ar_stride, ar_itrs, T, ar_token_stride, Tca = self._setup_ar_params(sconfig, Tc, P)

        # Extend T if GT is longer
        T_gt = motion_tokens_gt.shape[1]
        if T_gt > ar_itrs * ar_stride + Tm:
            ar_itrs = int(np.ceil((T_gt - Tm) / ar_stride))
        T = ar_itrs * ar_stride + Tm

        context, context_mask, motion_tokens_gt = self._pad_context(context, context_mask, T, motion_tokens_gt)

        # Motion Control: P(B_t | A_{0:t}, B_{0:t-1})
        context_mask[:, :Tc, 0] = True
        context_mask[:, :Tc, 1] = True
        context_mask[:, Tc, 0] = True  # Agent A gets one extra frame revealed

        xs_pred = torch.randn(S, T, P, D, device=self.device)
        xs_pred = torch.where(context_mask[..., None], motion_tokens_gt, xs_pred)

        # Generate sampling schedule
        noise_sch_matrix = self._create_schedule_matrix(
            sampling_schedule=sconfig.sampling_schedule,
            sampling_steps=sconfig.sampling_steps,
            token_len=Tm * P,
            sub_token_len=1,
            offset_proportion=sconfig.offset_proportion,
        )
        noise_sch_matrix = noise_sch_matrix.to(self.device)
        noise_sch_matrix[:, :Tc * P] = -1
        noise_sch_matrix = self.prune_noise_sch_matrix(noise_sch_matrix)

        # Motion Control: Agent A leads Agent B by 1 timestep
        noise_sch_matrix_ = noise_sch_matrix.clone().reshape(-1, Tm, P)
        agent_a_schedule = torch.full_like(noise_sch_matrix_[:, :, 1], 999)
        agent_a_schedule[:, :Tc + sconfig.sampling_subseq_len] = -1
        noise_sch_matrix_[:, :, 0] = agent_a_schedule
        noise_sch_matrix = noise_sch_matrix_.reshape(-1, Tm * P)

        ar_noise_sch_matrix = self._create_ar_schedule_matrix(ar_token_stride, noise_sch_matrix)

        # Override Agent A schedule in AR iterations
        ar_noise_sch_matrix_ = ar_noise_sch_matrix.reshape(-1, Tm, P)
        ar_agent_a_schedule = torch.full_like(ar_noise_sch_matrix_[:, :, 1], 999)
        ar_agent_a_schedule[:, :Tca + sconfig.sampling_subseq_len] = -1
        ar_noise_sch_matrix_[:, :, 0] = ar_agent_a_schedule
        ar_noise_sch_matrix = ar_noise_sch_matrix_.reshape(-1, Tm * P)

        sampling_time = noise_sch_matrix.shape[0] - 1
        ar_sampling_time = ar_noise_sch_matrix.shape[0] - 1
        whole_sampling_time = sampling_time + ar_sampling_time * ar_itrs

        pbar = tqdm(total=whole_sampling_time, desc="Sampling")
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        def sampling_step(
            Ts: int,
            Tcc: int,
            curr_noise_level: Float[Tensor, "S T P"],
            next_noise_level: Float[Tensor, "S T P"],
        ):
            nonlocal xs_pred, context_mask

            partial_curr_xs_pred = xs_pred[:, Ts:Ts + Tm]

            self._update_context_mask(context_mask, curr_noise_level, Ts, Tcc, Tm)

            # Inject ground truth for Agent A wherever it's marked as context
            xs_pred = torch.where(context_mask[..., None], motion_tokens_gt, xs_pred)
            partial_curr_xs_pred = xs_pred[:, Ts:Ts + Tm]

            partial_xs_pred_next = guidance_sampler(
                x_t=partial_curr_xs_pred,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                context_token_len=Tcc * P,
            )

            xs_pred[:, Ts:Ts + Tm] = torch.where(
                context_mask[:, Ts:Ts + Tm, :, None] == False,
                partial_xs_pred_next,
                xs_pred[:, Ts:Ts + Tm],
            )

            # Ensure all context tokens use ground truth
            pointer_A = torch.where(context_mask[:, Ts + Tcc:Ts + Tm, 0] == True)[-1]
            pointer_A = pointer_A.max().item() if pointer_A.numel() > 0 else 0
            if pointer_A > 0:
                xs_pred[:, Ts + Tcc:Ts + Tm, 0][:, pointer_A] = motion_tokens_gt[:, Ts + Tcc:Ts + Tm, 0][:, pointer_A]
                context_mask[:, Ts + Tcc:Ts + Tm, 0][:, pointer_A] = True
                curr_noise_level[:, pointer_A, 0] = -1
                next_noise_level[:, pointer_A, 0] = -1

            xs_pred = torch.where(context_mask[..., None], motion_tokens_gt, xs_pred)

        self.diffuser.eval()
        with torch.no_grad():
            Ts = 0

            for m in range(sampling_time):
                curr_noise_level = noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                next_noise_level = noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                sampling_step(Ts, Tc, curr_noise_level, next_noise_level)
                pbar.update(1)

            if sconfig.is_update_history_transforms:
                xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
                    xs_pred[:, Ts:Ts + Tm], Tc, sconfig.root_transform_mode)

            for ar in range(ar_itrs):
                Ts = Ts + ar_stride
                for m in range(ar_sampling_time):
                    curr_noise_level = ar_noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level = ar_noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step(Ts, Tca, curr_noise_level, next_noise_level)
                    pbar.update(1)

                if sconfig.is_update_history_transforms:
                    xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
                        xs_pred[:, Ts:Ts + Tm], Tca, sconfig.root_transform_mode)

        return xs_pred

    def _ar_sample_sequence_motion_control_live(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
        motion_tokens_gt: Float[Tensor, "B T P D"] | None = None,
    ):
        """
        Motion Control Live: Given Agent A's full GT, denoise Agent B in rolling
        64-frame windows, each with the FULL noise schedule.

        Instead of AR scheduling (which gives boundary tokens fewer denoising steps
        and causes jitter at frame 64), every chunk runs the complete noise schedule
        for uniform quality across all frames:

        1. Agent A is ALWAYS clean (GT available for the full sequence)
        2. Rolling window: each chunk is a fresh full-quality Tm-token denoising pass
        3. The last Tc tokens from the previous chunk become context for the next
        4. Stride = Tm - Tc: maximum efficiency, minimum overlap
        5. Every frame gets identical denoising quality (no boundary artifacts)
        6. GT injection only for Agent A + initial B context (never overwrites generated B)

        This maps to: P(B_t | A_{0:T}, B_{0:t-1}) with uniform quality across all frames.
        """
        sconfig = sampling_config
        S, Tc, P, D = context.shape
        T_gt = motion_tokens_gt.shape[1]

        Tm, stride, n_chunks, T = self._setup_rolling_params(sconfig, Tc, T_gt)

        context, context_mask, motion_tokens_gt = self._pad_context(context, context_mask, T, motion_tokens_gt)

        # Live motion control: Agent A is ALWAYS clean (fully observed)
        context_mask[:, :, 0] = True
        context_mask[:, :Tc, 1] = True

        # FIXED mask for GT injection -- only Agent A and initial Agent B context.
        gt_inject_mask = torch.zeros(S, T, P, device=self.device, dtype=torch.bool)
        gt_inject_mask[:, :, 0] = True
        gt_inject_mask[:, :Tc, 1] = True

        xs_pred = torch.randn(S, T, P, D, device=self.device)
        xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)

        # Noise schedule: FULL schedule reused for every chunk
        noise_sch_matrix = self._create_schedule_matrix(
            sampling_schedule=sconfig.sampling_schedule,
            sampling_steps=sconfig.sampling_steps,
            token_len=Tm * P,
            sub_token_len=P,
            offset_proportion=sconfig.offset_proportion,
        )
        noise_sch_matrix = noise_sch_matrix.to(self.device)
        noise_sch_matrix[:, :Tc * P] = -1
        noise_sch_matrix = self.prune_noise_sch_matrix(noise_sch_matrix)

        # Agent A is always clean
        noise_sch_matrix_ = noise_sch_matrix.reshape(-1, Tm, P)
        noise_sch_matrix_[:, :, 0] = -1
        noise_sch_matrix = noise_sch_matrix_.reshape(-1, Tm * P)

        sampling_time = noise_sch_matrix.shape[0] - 1
        whole_sampling_time = sampling_time * n_chunks

        pbar = tqdm(
            total=whole_sampling_time,
            desc=f"Live Motion Control (Rolling {self.config.sequence_length}f, {n_chunks} chunks)",
        )
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        use_temporal_smoothing = sconfig.use_temporal_smoothing
        smoothing_strength = sconfig.temporal_smoothing_strength
        smoothing_start_step = sconfig.temporal_smoothing_start_step

        def sampling_step(
            Ts: int,
            Tcc: int,
            curr_noise_level: Float[Tensor, "S T P"],
            next_noise_level: Float[Tensor, "S T P"],
        ):
            nonlocal xs_pred, context_mask

            self._update_context_mask(context_mask, curr_noise_level, Ts, Tcc, Tm)

            xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)
            partial_curr_xs_pred = xs_pred[:, Ts:Ts + Tm]

            partial_xs_pred_next = guidance_sampler(
                x_t=partial_curr_xs_pred,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                context_token_len=Tcc * P,
            )

            xs_pred[:, Ts:Ts + Tm] = torch.where(
                context_mask[:, Ts:Ts + Tm, :, None] == False,
                partial_xs_pred_next,
                xs_pred[:, Ts:Ts + Tm],
            )

            xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)

        self.diffuser.eval()
        with torch.no_grad():
            for chunk_idx in range(n_chunks):
                Ts = chunk_idx * stride

                if chunk_idx > 0:
                    context_mask[:, Ts:Ts + Tc, 1] = True

                for m in range(sampling_time):
                    curr_noise_level = noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level = noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step(Ts, Tc, curr_noise_level, next_noise_level)
                    pbar.update(1)

                if use_temporal_smoothing:
                    xs_pred[:, Ts:Ts + Tm], _ = apply_temporal_smoothing(
                        xs_pred[:, Ts:Ts + Tm], chunk_idx,
                        strength=smoothing_strength, start_step=smoothing_start_step)

                if sconfig.is_update_history_transforms:
                    xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
                        xs_pred[:, Ts:Ts + Tm], Tc, sconfig.root_transform_mode)

        return xs_pred[:, :T_gt]

    def _ar_sample_sequence_partner_prediction(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
        motion_tokens_gt: Float[Tensor, "B T P D"] | None = None,
    ):
        """
        Partner prediction with noised-GT Agent A in rolling windows.

        Both agents share the SAME noise schedule (in-distribution for the
        model).  The key difference: after every denoising step, Agent A's
        values are replaced with q_sample(GT_A, next_noise_level) -- the
        real GT noised to the schedule's current level.  This means:

        - The model always sees both agents at matching noise levels (training
          distribution), so it produces coherent joint predictions.
        - Agent B denoises against the REAL A signal (noised GT), not the
          model's own A prediction, so the final B motion is aligned with
          the actual GT A.
        - Agent A is never "predicted" -- it's always GT at the appropriate
          noise level, converging to clean GT as denoising finishes.
        - Generated Agent B tokens are NEVER overwritten with GT.

        Models: P(B_t | A_noised_GT, B_{0:t-1})

        Uses rolling windows for uniform quality across all frames.
        """
        sconfig = sampling_config
        S, Tc, P, D = context.shape
        T_gt = motion_tokens_gt.shape[1]

        Tm, stride, n_chunks, T = self._setup_rolling_params(sconfig, Tc, T_gt)

        context, context_mask, motion_tokens_gt = self._pad_context(context, context_mask, T, motion_tokens_gt)

        # Initial context: both agents clean for first Tc tokens
        context_mask[:, :Tc, 0] = True
        context_mask[:, :Tc, 1] = True

        # GT inject mask: Agent B ONLY gets GT for initial context.
        gt_inject_mask = torch.zeros(S, T, P, device=self.device, dtype=torch.bool)
        gt_inject_mask[:, :Tc, 0] = True
        gt_inject_mask[:, :Tc, 1] = True

        # Pre-generate fixed noise for Agent A's q_sample
        noise_a = torch.randn(S, T, 1, D, device=self.device).expand(S, T, P, D).clone()

        xs_pred = torch.randn(S, T, P, D, device=self.device)
        xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)

        # Noise schedule: SAME for both agents
        noise_sch_matrix = self._create_schedule_matrix(
            sampling_schedule=sconfig.sampling_schedule,
            sampling_steps=sconfig.sampling_steps,
            token_len=Tm * P,
            sub_token_len=P,
            offset_proportion=sconfig.offset_proportion,
        )
        noise_sch_matrix = noise_sch_matrix.to(self.device)
        noise_sch_matrix[:, :Tc * P] = -1
        noise_sch_matrix = self.prune_noise_sch_matrix(noise_sch_matrix)

        sampling_time = noise_sch_matrix.shape[0] - 1
        whole_sampling_time = sampling_time * n_chunks

        pbar = tqdm(
            total=whole_sampling_time,
            desc=f"Partner Prediction (Rolling {self.config.sequence_length}f, {n_chunks} chunks)",
        )
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        def sampling_step(
            Ts: int,
            Tcc: int,
            curr_noise_level: Float[Tensor, "S T P"],
            next_noise_level: Float[Tensor, "S T P"],
        ):
            nonlocal xs_pred, context_mask

            self._update_context_mask(context_mask, curr_noise_level, Ts, Tcc, Tm)

            xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)
            partial_curr_xs_pred = xs_pred[:, Ts:Ts + Tm]

            partial_xs_pred_next = guidance_sampler(
                x_t=partial_curr_xs_pred,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                context_token_len=Tcc * P,
            )

            xs_pred[:, Ts:Ts + Tm] = torch.where(
                context_mask[:, Ts:Ts + Tm, :, None] == False,
                partial_xs_pred_next,
                xs_pred[:, Ts:Ts + Tm],
            )

            # Agent A: replace with noised GT at the next noise level
            next_a_level = next_noise_level[:, :, 0:1]  # (S, Tm, 1)
            a_gt_window = motion_tokens_gt[:, Ts:Ts + Tm, 0:1, :]
            a_noise_window = noise_a[:, Ts:Ts + Tm, 0:1, :]

            a_level_clamped = next_a_level.clamp(min=0)
            a_noised = self.diffuser.q_sample(
                x_start=a_gt_window,
                k=a_level_clamped,
                noise=a_noise_window,
            )

            a_is_clean = (next_a_level == -1).unsqueeze(-1)
            a_value = torch.where(a_is_clean, a_gt_window, a_noised)

            a_context = context_mask[:, Ts:Ts + Tm, 0:1].unsqueeze(-1)
            xs_pred[:, Ts:Ts + Tm, 0:1, :] = torch.where(
                a_context, motion_tokens_gt[:, Ts:Ts + Tm, 0:1, :], a_value,
            )

            xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)

        self.diffuser.eval()
        with torch.no_grad():
            for chunk_idx in range(n_chunks):
                Ts = chunk_idx * stride

                if chunk_idx > 0:
                    context_mask[:, Ts:Ts + Tc, :] = True
                    gt_inject_mask[:, Ts:Ts + Tc, 0] = True

                for m in range(sampling_time):
                    curr_noise_level = noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level = noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step(Ts, Tc, curr_noise_level, next_noise_level)
                    pbar.update(1)

        return xs_pred[:, :T_gt]


    def _ar_sample_sequence_partner_inpainting(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
        motion_tokens_gt: Float[Tensor, "B T P D"] | None = None,
    ):
        """
        Partner inpainting with rolling windows (full schedule per chunk).

        Agent A is fully GT. Agent B is denoised in rolling Tm-token windows,
        each receiving the COMPLETE noise schedule for uniform quality:

        1. Agent A is ALWAYS clean (GT for the full sequence)
        2. Each chunk is a fresh full-quality denoising pass
        3. The last Tc tokens of Agent B from the previous chunk become context
        4. Stride = Tm - Tc for maximum coverage with minimal overlap
        5. Every frame gets identical denoising quality (no AR degradation)
        """
        sconfig = sampling_config
        S, Tc, P, D = context.shape
        T_gt = motion_tokens_gt.shape[1]

        Tm = self.config.sequence_length // VQVAE_TEMPORAL_FACTOR

        # Rolling window: stride = Tm - Tc
        stride = Tm - Tc
        n_chunks = max(1, int(np.ceil((T_gt - Tm) / stride)) + 1) if T_gt > Tm else 1
        T = (n_chunks - 1) * stride + Tm if n_chunks > 1 else max(T_gt, Tm)

        # Pad if needed
        if T > T_gt:
            pad = T - T_gt
            motion_tokens_gt = torch.cat([
                motion_tokens_gt,
                torch.zeros(S, pad, P, D, device=self.device),
            ], dim=1)

        context, context_mask = self._pad_context(context, context_mask, T)

        # Agent A is always GT; Agent B context is only initial frames
        context_mask[:, :, 0] = True
        context_mask[:, :Tc, 1] = True

        # Fixed GT injection mask — only Agent A + initial Agent B context
        gt_inject_mask = torch.zeros(S, T, P, device=self.device, dtype=torch.bool)
        gt_inject_mask[:, :, 0] = True       # Agent A: always GT
        gt_inject_mask[:, :Tc, 1] = True      # Agent B: only initial context

        xs_pred = torch.randn(S, T, P, D, device=self.device)
        xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)

        # Full noise schedule reused for every chunk
        noise_sch_matrix = self._create_schedule_matrix(
            sampling_schedule=sconfig.sampling_schedule,
            sampling_steps=sconfig.sampling_steps,
            token_len=Tm * P,
            sub_token_len=P,
            offset_proportion=sconfig.offset_proportion,
        )
        noise_sch_matrix = noise_sch_matrix.to(self.device)
        noise_sch_matrix[:, :Tc * P] = -1
        noise_sch_matrix = self.prune_noise_sch_matrix(noise_sch_matrix)

        # Agent A is always clean
        noise_sch_matrix_ = noise_sch_matrix.reshape(-1, Tm, P)
        noise_sch_matrix_[:, :, 0] = -1
        noise_sch_matrix = noise_sch_matrix_.reshape(-1, Tm * P)

        sampling_time = noise_sch_matrix.shape[0] - 1
        whole_sampling_time = sampling_time * n_chunks

        pbar = tqdm(
            total=whole_sampling_time,
            desc=f"Partner Inpainting (Rolling {self.config.sequence_length}f, {n_chunks} chunks)",
        )
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        use_temporal_smoothing = sconfig.use_temporal_smoothing
        smoothing_strength = sconfig.temporal_smoothing_strength
        smoothing_start_step = sconfig.temporal_smoothing_start_step

        def sampling_step(
            Ts: int,
            Tcc: int,
            curr_noise_level: Float[Tensor, "S T P"],
            next_noise_level: Float[Tensor, "S T P"],
        ):
            nonlocal xs_pred, context_mask

            # Update context_mask for newly clean tokens
            context_mask[:, Ts + Tcc:Ts + Tm] = torch.where(
                torch.logical_and(
                    context_mask[:, Ts + Tcc:Ts + Tm] == False,
                    curr_noise_level[:, Tcc:] == -1,
                ),
                True,
                context_mask[:, Ts + Tcc:Ts + Tm],
            )

            # Re-inject Agent A GT + initial Agent B context (never overwrite generated B)
            xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)

            partial_xs_pred_next = guidance_sampler(
                x_t=xs_pred[:, Ts:Ts + Tm],
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                context_token_len=Tcc * P,
            )

            # Only update non-context tokens (Agent B being denoised)
            xs_pred[:, Ts:Ts + Tm] = torch.where(
                context_mask[:, Ts:Ts + Tm, :, None] == False,
                partial_xs_pred_next,
                xs_pred[:, Ts:Ts + Tm],
            )

            # Protect Agent A
            xs_pred = torch.where(gt_inject_mask[..., None], motion_tokens_gt, xs_pred)

        self.diffuser.eval()
        with torch.no_grad():
            for chunk_idx in range(n_chunks):
                Ts = chunk_idx * stride

                # For chunks after the first: mark previous Agent B output as context
                if chunk_idx > 0:
                    context_mask[:, Ts:Ts + Tc, 1] = True

                # Run FULL noise schedule — every chunk gets identical quality
                for m in range(sampling_time):
                    curr_noise_level = noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level = noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step(Ts, Tc, curr_noise_level, next_noise_level)
                    pbar.update(1)

                # Post-processing after each chunk
                if use_temporal_smoothing:
                    xs_pred[:, Ts:Ts + Tm], _ = apply_temporal_smoothing(
                        xs_pred[:, Ts:Ts + Tm], chunk_idx,
                        strength=smoothing_strength, start_step=smoothing_start_step)
                if sconfig.is_update_history_transforms:
                    xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
                        xs_pred[:, Ts:Ts + Tm], Tc, sconfig.root_transform_mode)

        return xs_pred[:, :T_gt]

    def _ar_sample_sequence_inbetweening(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
    ):
        sconfig = sampling_config
        S, Tc, P, D = context.shape

        Tm = self.config.sequence_length // 4
        ar_stride = sconfig.ar_seq_stride // 4
        ar_itrs = int(np.ceil((max(sconfig.sampling_seq_len // 4, Tm) - Tm) / ar_stride))

        T = context_mask.shape[1]

        padding = T - Tc
        context = torch.cat([context, torch.zeros(S, padding, P, D, device=self.device)], dim=1)
        context_mask = torch.cat([
            context_mask,
            torch.zeros(S, padding, P, device=self.device, dtype=torch.bool),
        ], dim=1)

        xs_pred = torch.randn(S, T, P, D, device=self.device)
        xs_pred = torch.where(context_mask[..., None], context, xs_pred)

        # Generate sampling schedule
        noise_sch_matrix = self._create_schedule_matrix(
            sampling_schedule=sconfig.sampling_schedule,
            sampling_steps=sconfig.sampling_steps,
            token_len=Tm * P,
            sub_token_len=P,
            offset_proportion=sconfig.offset_proportion,
        )
        noise_sch_matrix = noise_sch_matrix.to(self.device)

        noise_sch_matrix = self.prune_noise_sch_matrix(noise_sch_matrix)

        # Make noise_sch_matrix -1 for key frame indices within the model window
        noise_sch_matrix_ = noise_sch_matrix.clone().reshape(-1, Tm, P)
        noise_sch_matrix_ = torch.where(context_mask[:1, :Tm], -1, noise_sch_matrix_)
        noise_sch_matrix = noise_sch_matrix_.reshape(-1, Tm * P)

        ar_token_stride = ar_stride * P
        Tca = Tm - ar_token_stride
        ar_noise_sch_matrix = self._create_ar_schedule_matrix(ar_token_stride, noise_sch_matrix)

        sampling_time = noise_sch_matrix.shape[0] - 1
        ar_sampling_time = ar_noise_sch_matrix.shape[0] - 1
        whole_sampling_time = sampling_time + ar_sampling_time * ar_itrs

        pbar = tqdm(total=whole_sampling_time, desc="Sampling")
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        use_temporal_smoothing = sconfig.use_temporal_smoothing
        smoothing_strength = sconfig.temporal_smoothing_strength
        smoothing_start_step = sconfig.temporal_smoothing_start_step

        def sampling_step(
            Ts: int,
            Tcc: int,
            curr_noise_level: Float[Tensor, "S T P"],
            next_noise_level: Float[Tensor, "S T P"],
        ):
            partial_curr_xs_pred = xs_pred[:, Ts:Ts + Tm]

            self._update_context_mask(context_mask, curr_noise_level, Ts, Tcc, Tm)

            partial_xs_pred_next = guidance_sampler(
                x_t=partial_curr_xs_pred,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                context_token_len=Tcc * P,
            )

            xs_pred[:, Ts:Ts + Tm] = torch.where(
                context_mask[:, Ts:Ts + Tm, :, None] == False,
                partial_xs_pred_next,
                xs_pred[:, Ts:Ts + Tm],
            )

        self.diffuser.eval()
        with torch.no_grad():
            Ts = 0

            for m in range(sampling_time):
                curr_noise_level = noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                next_noise_level = noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                sampling_step(Ts, Tc, curr_noise_level, next_noise_level)
                pbar.update(1)

            if use_temporal_smoothing:
                xs_pred[:, Ts:Ts + Tm], _ = apply_temporal_smoothing(
                    xs_pred[:, Ts:Ts + Tm], 0,
                    strength=smoothing_strength, start_step=smoothing_start_step)

            # if sconfig.is_update_history_transforms:
            #     xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
            #         xs_pred[:, Ts:Ts + Tm], Tc, sconfig.root_transform_mode)

            for ar in range(ar_itrs):
                Ts = Ts + ar_stride
                for m in range(ar_sampling_time):
                    curr_noise_level = ar_noise_sch_matrix[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level = ar_noise_sch_matrix[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step(Ts, Tca, curr_noise_level, next_noise_level)
                    pbar.update(1)

                if use_temporal_smoothing:
                    xs_pred[:, Ts:Ts + Tm], _ = apply_temporal_smoothing(
                        xs_pred[:, Ts:Ts + Tm], ar + 1,
                        strength=smoothing_strength, start_step=smoothing_start_step)

                if sconfig.is_update_history_transforms:
                    xs_pred[:, Ts:Ts + Tm] = self._postprocess_transforms(
                        xs_pred[:, Ts:Ts + Tm], Tca, sconfig.root_transform_mode)

        return xs_pred

    def _ar_agentic_sync_sequence(
        self,
        sampling_config: DFOTSamplingConfig,
        context: Float[Tensor, "B T P D"],
        context_mask: Bool[Tensor, "B T P"] | None = None,
        betas: Float[Tensor, "B P 10"] | None = None,
        gt_seq_len: int = None,
    ):
        sconfig = sampling_config
        S, Tc, P, D = context.shape

        Tm, ar_stride, ar_itrs, T, ar_token_stride, Tca = self._setup_ar_params(sconfig, Tc, P, T_gt=gt_seq_len)

        context, context_mask = self._pad_context(context, context_mask, T)
        context_mask_A = context_mask
        context_mask_B = torch.clone(context_mask_A)

        xs_pred_A = torch.randn(S, T, P, D, device=self.device)
        xs_pred_B = torch.randn(S, T, P, D, device=self.device)
        xs_pred_A = torch.where(context_mask_A[..., None], context, xs_pred_A)
        xs_pred_B = torch.where(context_mask_B[..., None], context, xs_pred_B)

        noise_sch_matrix_A = self.generate_noise_sch_matrix(sconfig, token_len=Tm * P, context_token_len=Tc * P, P=P)
        noise_sch_matrix_B = self.generate_noise_sch_matrix(sconfig, token_len=Tm * P, context_token_len=Tc * P, P=P)

        ar_noise_sch_matrix_A = self._create_ar_schedule_matrix(ar_token_stride, noise_sch_matrix_A)
        ar_noise_sch_matrix_B = self._create_ar_schedule_matrix(ar_token_stride, noise_sch_matrix_B)

        sampling_time = noise_sch_matrix_A.shape[0] - 1
        ar_sampling_time = ar_noise_sch_matrix_A.shape[0] - 1
        whole_sampling_time = sampling_time + ar_sampling_time * ar_itrs

        pbar = tqdm(total=whole_sampling_time, desc="Sampling")
        guidance_sampler = self._create_guidance_sampler(sconfig, betas, whole_sampling_time)

        def sampling_step_tt(
            Ts: int,
            Tcc: int,
            curr_noise_level_A: Float[Tensor, "S T P"],
            next_noise_level_A: Float[Tensor, "S T P"],
            curr_noise_level_B: Float[Tensor, "S T P"],
            next_noise_level_B: Float[Tensor, "S T P"],
        ):
            """Denoise Agent A and B simultaneously, then cross-update."""
            nonlocal context_mask_A, context_mask_B, xs_pred_A, xs_pred_B

            S_, T_, P_, D_ = xs_pred_A.shape
            xs_pred_A_ = xs_pred_A.reshape(S_, T_ * P_, D_)
            xs_pred_B_ = xs_pred_B.reshape(S_, T_ * P_, D_)
            end_token = (Ts + Tm) * P_
            start_token = Ts * P_
            start_context = (Ts + Tcc) * P_

            partial_curr_xs_pred_A = xs_pred_A_[:, start_token:end_token].reshape(S_, -1, P_, D_)
            partial_curr_xs_pred_B = xs_pred_B_[:, start_token:end_token].reshape(S_, -1, P_, D_)

            curr_noise_level_A_ = curr_noise_level_A.clone().reshape(S_, -1)
            next_noise_level_A_ = next_noise_level_A.clone().reshape(S_, -1)
            curr_noise_level_B_ = curr_noise_level_B.clone().reshape(S_, -1)
            next_noise_level_B_ = next_noise_level_B.clone().reshape(S_, -1)

            context_mask_A_ = context_mask_A.clone().reshape(S_, -1)

            # Update context mask with denoised tokens
            new_context_mask = context_mask_A_.clone()
            new_context_mask[:, start_context:end_token] = torch.where(
                torch.logical_and(
                    context_mask_A_[:, start_context:end_token] == False,
                    curr_noise_level_A_[:, P_ * Tcc:] == -1,
                ),
                True,
                context_mask_A_[:, start_context:end_token],
            )

            context_mask_A_ = new_context_mask.clone()
            context_mask_B_ = new_context_mask.clone()

            # Step 1: Sample next tokens for both Agent A and Agent B
            partial_xs_pred_next_A = guidance_sampler(
                x_t=partial_curr_xs_pred_A,
                curr_noise_level=curr_noise_level_A,
                next_noise_level=next_noise_level_A,
                context_token_len=Tcc * P,
            )

            partial_xs_pred_next_B = guidance_sampler(
                x_t=partial_curr_xs_pred_B,
                curr_noise_level=curr_noise_level_B,
                next_noise_level=next_noise_level_B,
                context_token_len=Tcc * P,
            )

            # Update xs_pred_A and xs_pred_B with denoised tokens
            xs_pred_A_[:, start_token:end_token] = torch.where(
                context_mask_A_[:, start_token:end_token, None] == False,
                partial_xs_pred_next_A.reshape(S_, -1, D_),
                xs_pred_A_[:, start_token:end_token],
            )

            xs_pred_B_[:, start_token:end_token] = torch.where(
                context_mask_B_[:, start_token:end_token, None] == False,
                partial_xs_pred_next_B.reshape(S_, -1, D_),
                xs_pred_B_[:, start_token:end_token],
            )

            # Cross-update: exchange predictions between agents
            xs_pred_A_ = xs_pred_A_.reshape(S_, T_, P_, D_)
            xs_pred_B_ = xs_pred_B_.reshape(S_, T_, P_, D_)

            partial_xs_pred_next_A_reshaped = partial_xs_pred_next_A.reshape(S_, -1, P_, D_)
            partial_xs_pred_next_B_reshaped = partial_xs_pred_next_B.reshape(S_, -1, P_, D_)

            start_t = start_token // P_
            end_t = end_token // P_

            context_mask_B_slice = context_mask_B_[:, start_token:end_token].reshape(S_, -1, P_)
            context_mask_A_slice = context_mask_A_[:, start_token:end_token].reshape(S_, -1, P_)

            xs_pred_A_[:, start_t:end_t, 1] = torch.where(
                context_mask_B_slice[:, :, 1:2] == False,
                partial_xs_pred_next_A_reshaped[:, :, 1],
                xs_pred_B_[:, start_t:end_t, 1],
            )

            xs_pred_B_[:, start_t:end_t, 0] = torch.where(
                context_mask_A_slice[:, :, 0:1] == False,
                partial_xs_pred_next_B_reshaped[:, :, 0],
                xs_pred_A_[:, start_t:end_t, 0],
            )

            # Step 2: Sample next token for Agent B (follower) with updated context
            partial_xs_pred_next_B = guidance_sampler(
                x_t=partial_curr_xs_pred_B,
                curr_noise_level=curr_noise_level_B_.reshape(S_, -1, P_),
                next_noise_level=next_noise_level_B_.reshape(S_, -1, P_),
                context_token_len=Tcc * P,
            )

            xs_pred_A = xs_pred_A_.reshape(S_, T_, P_, D_)
            xs_pred_B = xs_pred_B_.reshape(S_, T_, P_, D_)

        self.diffuser.eval()
        with torch.no_grad():
            Ts = 0

            for m in range(sampling_time):
                curr_noise_level_A = noise_sch_matrix_A[m].reshape(1, Tm, P).expand(S, Tm, P)
                next_noise_level_A = noise_sch_matrix_A[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                curr_noise_level_B = noise_sch_matrix_B[m].reshape(1, Tm, P).expand(S, Tm, P)
                next_noise_level_B = noise_sch_matrix_B[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                sampling_step_tt(Ts, Tc, curr_noise_level_A, next_noise_level_A, curr_noise_level_B, next_noise_level_B)
                pbar.update(1)

            for ar in range(ar_itrs):
                Ts = Ts + ar_stride
                for m in range(ar_sampling_time):
                    curr_noise_level_A = ar_noise_sch_matrix_A[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level_A = ar_noise_sch_matrix_A[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    curr_noise_level_B = ar_noise_sch_matrix_B[m].reshape(1, Tm, P).expand(S, Tm, P)
                    next_noise_level_B = ar_noise_sch_matrix_B[m + 1].reshape(1, Tm, P).expand(S, Tm, P)
                    sampling_step_tt(Ts, Tca, curr_noise_level_A, next_noise_level_A, curr_noise_level_B, next_noise_level_B)
                    pbar.update(1)

        return xs_pred_A

    # ------------------------------------------------------------------ #
    #  Schedule matrix construction                                       #
    # ------------------------------------------------------------------ #

    def _create_schedule_matrix(
        self,
        sampling_schedule: Literal["full_sequence", "causal_uncertainty", "autoregressive"],
        sampling_steps: int,
        token_len: int,
        sub_token_len: int | None = None,
        offset_proportion: float = 0.25,
    ) -> Float[Tensor, "sampling_steps token_len"]:
        noise_sch_matrix = None
        if sampling_schedule == "full_sequence":
            noise_sch_matrix = torch.arange(sampling_steps, -1, -1)
            noise_sch_matrix = noise_sch_matrix[:, None].expand(-1, token_len)
        elif sampling_schedule == "causal_uncertainty":
            noise_sch_matrix = self._create_pyramid_schedule_matrix(sampling_steps, token_len, sub_token_len)
        elif sampling_schedule == "autoregressive":
            noise_sch_matrix = self._create_offset_sampling_matrix(sampling_steps, token_len, sub_token_len, offset_proportion)
        else:
            raise ValueError(f"Invalid sampling schedule: {sampling_schedule}")

        real_steps = torch.linspace(-1, self.max_noise, sampling_steps + 1).long()
        noise_sch_matrix = real_steps[noise_sch_matrix]
        return noise_sch_matrix

    def _create_offset_sampling_matrix(
        self,
        sampling_steps: int,
        token_len: int,
        subseq_len: int,
        offset_proportion: float = 0.25,
    ) -> Float[Tensor, "height token_len"]:
        offset_by = int(sampling_steps * (1 - offset_proportion))
        group_count = int(np.ceil(token_len / subseq_len))
        offset_per_group = max(sampling_steps - offset_by, 0)
        max_group_index = max(group_count - 1, 0)
        height = sampling_steps + max_group_index * offset_per_group + 1
        scheduling_matrix = torch.zeros((height, token_len), dtype=torch.int64)
        for m in range(height):
            for t in range(token_len):
                group_index = int(t / subseq_len)
                scheduling_matrix[m, t] = sampling_steps + group_index * offset_per_group - m
        return torch.clamp(scheduling_matrix, 0, sampling_steps)

    def _create_pyramid_schedule_matrix(
        self,
        sampling_steps: int,
        token_len: int,
        sub_token_len: int,
    ) -> Float[Tensor, "sampling_steps token_len"]:
        height = sampling_steps + int(token_len / sub_token_len)
        scheduling_matrix = torch.zeros((height, token_len), dtype=torch.int64)
        for m in range(height):
            for t in range(token_len):
                scheduling_matrix[m, t] = sampling_steps + int(t / sub_token_len) - m
        return torch.clamp(scheduling_matrix, 0, sampling_steps)

    def _create_ar_schedule_matrix(
        self,
        ar_token_stride: int,
        noise_sch_matrix: Float[Tensor, "steps tokens"],
    ) -> Float[Tensor, "steps tokens"]:
        for itr_start_idx, noise_level in enumerate(noise_sch_matrix):
            if noise_level[-ar_token_stride] != self.max_noise:
                break
        itr_start_idx -= 1
        ar_noise_sch_matrix = noise_sch_matrix[itr_start_idx:].clone()
        ar_noise_sch_matrix[:, :-ar_token_stride] = -1
        return ar_noise_sch_matrix

    def compute_T_world_root(self, pred_motion_tokens: MotionToken, betas: Float[Tensor, "B P 10"]):
        with torch.no_grad():
            B, P = betas.shape[:2]

            tm1_t_pred_raw = pred_motion_tokens.canonical_tm1_t_transforms
            Tz = tm1_t_pred_raw.shape[1]
            dec_cond = torch.cat([betas[:, None, :].expand(B, Tz, P, -1), tm1_t_pred_raw], dim=-1)
            tm1_t_pred_raw = rearrange(tm1_t_pred_raw, "b tz s (f d) -> b (tz f) s d", f=4)
            T = tm1_t_pred_raw.shape[1]

            enc_cond = torch.cat([betas[:, None, :].expand(B, T, P, -1), tm1_t_pred_raw], dim=-1)
            pose_z = pred_motion_tokens.pose_latent
            pose_val = self.decode(pose_z, dec_cond)

            pose_tokens = MultiPoseToken.unpack(pose_val, enc_cond)

            multi_pred_motion = MultiPoseToken.denormalize(pose_tokens, self.mean, self.std)

            root_processor = RootTransformProcessor()

            T_world_root = root_processor.convert_root_transform(
                pred_motion=multi_pred_motion,
                mode="temporal",
            )

        return T_world_root, multi_pred_motion


class DFOTNetwork(DFOTBase):
    def __init__(self, config: DFOTConfig, device: torch.device):
        print('DFOTNetwork Cond')
        super(DFOTNetwork, self).__init__(config, device)

        self._vqvae = PoseNetwork.load(
            config.vqvae_model_path / "model.safetensors",
            config.vqvae_cfg,
            device)

        # set vqvae frozen
        self._vqvae.eval()
        for p in self._vqvae.parameters():
            p.requires_grad = False

        self._loss_weight = {
            "pose_latent": 1.0,
        }

        if config.loss_weight is not None:
            self._loss_weight.update(config.loss_weight)
            if config.without_self_partner or config.person_num <= 1:
                self._loss_weight.pop("self_partner_transforms", None)
                self._loss_weight.pop("canonical_consistency_translations", None)
                self._loss_weight.pop("canonical_consistency_rotations", None)

        if config.loss_func == 'l1':
            self._loss_func = F.l1_loss
        elif config.loss_func == 'mse':
            self._loss_func = F.mse_loss
        elif config.loss_func == 'smooth_l1':
            self._loss_func = F.smooth_l1_loss
        else:
            raise ValueError(f"Invalid loss function: {config.loss_func}")

        self.body_model = SmplModel.load(config.smpl_model_path).to(device)
        ms = np.load(config.mean_std_path)
        self.mean = torch.from_numpy(ms["mean"]).to(torch.float32).to(device)
        self.std = torch.from_numpy(ms["std"]).to(torch.float32).to(device)
        self.foot_idx = [6, 7, 9, 10]
        self.d_latent = config.vqvae_cfg.d_latent

    def init_model(self):
        self.diffuser = DiscreteDiffusion(self.config, TransformerModel(self.config))
        if self.config.diffusion_noise_rand_type == "uniform":
            self.rand_fn = partial(torch.randint, 0, self.config.max_t)
        elif self.config.diffusion_noise_rand_type == "logsnr":
            self.rand_fn = make_logsnr_rand_fn(
                self.diffuser.alphas_cumprod, self.config.diffusion_noise_logsnr_mu, self.config.diffusion_noise_logsnr_sigma)
        self.max_noise = self.config.max_t - 1

    def _setup_guidance_decode(self, guidance_sampler, betas):
        """Set up VQ-VAE decode function for x0-space guidance (smoothing + foot skating)."""
        needs_decode = (
            guidance_sampler.use_foot_skating_guidance
            or guidance_sampler.use_smoothing_guidance
        )
        if needs_decode and betas is not None:
            def _decode_fn(pose_latent, dec_cond):
                B, Tz, P, _ = pose_latent.shape
                z = rearrange(pose_latent, "b t p d -> (b p) t d")
                dc = rearrange(dec_cond, "b t p d -> (b p) t d")
                val = self._vqvae.decode_wo_quantize(z, dc)
                return rearrange(val, "(b p) t d -> b t p d", b=B, p=P)

            guidance_sampler.decode_fn = _decode_fn
            guidance_sampler.betas = betas

    def _postprocess_transforms(self, chunk, context_len, root_transform_mode):
        """Re-orthogonalize transforms and recompute partner transforms after each AR chunk.

        The swarm MotionToken has [pose_latent, canonical_tm1_t(4x9), canonical_self_partner((P-1)x4x9)].
        This method:
          1. Re-orthogonalizes SE3 transforms (prevents numerical drift).
          2. Recomputes canonical_self_partner_transforms from the temporal chain
             so inter-person transforms stay consistent with accumulated root trajectory.

        Args:
            chunk: [S, Tm, P, D] packed motion tokens (normalized)
            context_len: number of context tokens in this chunk
            root_transform_mode: "temporal" or "temporal_partner"
        Returns:
            Updated chunk [S, Tm, P, D]
        """
        token = MotionToken.unpack(chunk)
        token = MotionToken.denormalize(token, self.mean, self.std)

        S, T, P, _ = token.canonical_tm1_t_transforms.shape

        # 1. Re-orthogonalize SE3 transforms
        tm1_t = token.canonical_tm1_t_transforms.reshape(S, T, P, 4, 9)
        tm1_t[..., 7] = 0.0
        token.canonical_tm1_t_transforms = SE3.from_9d(tm1_t.reshape(S, T, P, 4*9).reshape(S, T, P, 4, 9)).as_9d().reshape(S, T, P, 4*9)

        sp = token.canonical_self_partner_transforms.reshape(S, T, P, P-1, 4, 9)
        sp[..., 7] = 0.0
        token.canonical_self_partner_transforms = SE3.from_9d(sp.reshape(S, T, P, P-1, 4, 9)).as_9d().reshape(S, T, P, P-1, 4*9)

        if P == 1:
            return MotionToken.normalize(token, self.mean, self.std).pack()

        if root_transform_mode == "temporal":
            # 2. Recompute partner transforms from temporal chain
            T_world_first_partner_canonical = SE3.from_9d(
                token.canonical_self_partner_transforms[:, context_len-1, 0].reshape(S, P-1, 4, 9)[:, :, 0, :]
            ).wxyz_xyz
            T_world_first_canonical = torch.cat([
                SE3.identity(chunk.device, chunk.dtype).wxyz_xyz[None, None, :].expand(S, 1, 7),
                T_world_first_partner_canonical,
            ], dim=1)

            T_canonical_tm1_t = SE3.from_9d(
                token.canonical_tm1_t_transforms[:, context_len:].reshape(S, T-context_len, P, 4, 9)[:, :, :, 0, :]
            ).wxyz_xyz
            T_world_canonical = RootTransformProcessor.calc_canonical_trans_using_temporal_trans(
                T_canonical_tm1_t=T_canonical_tm1_t,
                T_world_first_canonical=T_world_first_canonical,
            )

            T_sp = (SE3(T_world_canonical[:, :, :, None, :]).inverse() @ SE3(T_world_canonical[:, :, None, :, :])).wxyz_xyz
            T_sp_list = []
            for i in range(P):
                rel_i = T_sp[:, :, i, torch.arange(P) != i, :]
                T_sp_list.append(rel_i)
            T_self_partner = torch.stack(T_sp_list, dim=2)  # [S, T', P, P-1, 7]

            sp_updated = token.canonical_self_partner_transforms[:, context_len:].reshape(S, T-context_len, P, P-1, 4, 9)
            sp_updated[:, :, :, :, 0, :] = SE3(T_self_partner).as_9d()
            token.canonical_self_partner_transforms[:, context_len:] = sp_updated.reshape(S, T-context_len, P, P-1, 4*9)

        return MotionToken.normalize(token, self.mean, self.std).pack()

    def encode(
        self,
        batch: TrainingData,
    ) -> Tuple[Float[Tensor, "B Tz P D"], Float[Tensor, "B Tz P"]]:
        B, T, P, _ = batch.betas.shape
        pose_tokens = PoseToken.convert_from_training_data(batch)
        val = pose_tokens.pack_val()
        enc_cond = pose_tokens.pack_enc_cond()
        mask = pose_tokens.mask
        z, mask_z = self._vqvae.encode(val, enc_cond, mask)
        z = rearrange(z, "(b p) t d -> b t p d", b=B, p=P)
        mask_z = rearrange(mask_z, "(b p) t d -> b t p d", b=B, p=P)
        pose_latent = z.detach()
        mask = mask_z.squeeze(-1)
        return pose_latent, mask

    def decode(
        self,
        pose_latent: Float[Tensor, "B Tz P D"],
        dec_cond: Float[Tensor, "B Tz P D"]
    ) -> Float[Tensor, "B T P D"]:
        B, Tz, P, _ = pose_latent.shape
        pose_z = rearrange(pose_latent, "b t p d -> (b p) t d")
        dec_cond = rearrange(dec_cond, "b t p d -> (b p) t d")
        pose_val = self._vqvae.decode(pose_z, dec_cond)
        pose_val = rearrange(pose_val, "(b p) t d -> b t p d", b=B, p=P)
        return pose_val

    def decode_wo_quantize(
        self,
        pose_latent: Float[Tensor, "B Tz P D"],
        dec_cond: Float[Tensor, "B Tz P D"]
    ) -> Float[Tensor, "B T P D"]:
        B, Tz, P, _ = pose_latent.shape
        pose_z = rearrange(pose_latent, "b t p d -> (b p) t d")
        dec_cond = rearrange(dec_cond, "b t p d -> (b p) t d")
        pose_val = self._vqvae.decode_wo_quantize(pose_z, dec_cond)
        pose_val = rearrange(pose_val, "(b p) t d -> b t p d", b=B, p=P)
        return pose_val

    def get_cond(
        self,
        batch: TrainingData,
    ) -> Dict[str, Float[Tensor, "B P D"]]:
        token_kwargs = {}
        raw_tm1_t_data = getattr(batch, "T_canonical_tm1_canonical_t")
        raw_tm1_t_data = rearrange(raw_tm1_t_data, "b (tz f) s d -> b tz s (f d)", f=4)
        token_kwargs["canonical_tm1_t_transforms"] = raw_tm1_t_data
        raw_self_partner_data = getattr(batch, "T_self_canonical_partner_canonical")
        raw_self_partner_data = rearrange(raw_self_partner_data, "b (tz f) s p d -> b tz s p (f d)", f=4)
        token_kwargs["canonical_self_partner_transforms"] = raw_self_partner_data
        return token_kwargs

    def training_step(
        self,
        batch: TrainingData,
        training_step: int,
    ) -> Tuple[Float[Tensor, "B T D"], Float[Tensor, "B T"]]:
        token_kwargs = self.get_cond(batch)

        with torch.no_grad():
            pose_latent, mask = self.encode(batch)
        token_kwargs["pose_latent"] = pose_latent

        motion_tokens = MotionToken(**token_kwargs)
        val = motion_tokens.pack()

        train_kwargs = {}
        train_kwargs["is_token_cat"] = motion_tokens.is_cat
        train_kwargs["foot_contact"] = batch.body_contacts[:, :, :, self.foot_idx]
        train_kwargs["betas"] = batch.betas[:, 0, :, :]
        train_kwargs["T_world_canonical"] = self._denormalize(batch.T_world_canonical, "T_world_canonical")
        train_kwargs["T_self_canonical_partner_canonical"] = self._denormalize(batch.T_self_canonical_partner_canonical, "T_self_canonical_partner_canonical")
        train_kwargs["warmup_w"] = torch.tensor(training_step / float(self.config.warmup_steps), device=self.device).clamp(0., 1.)

        return super()._training_step(val, mask, **train_kwargs)

    def calc_body_joint_positions(
        self,
        betas: Float[Tensor, "B P 10"],
        body_6d: Float[Tensor, "B T P J 6"],
        T_world_root: Float[Tensor, "B T P 7"]
    ) -> Float[Tensor, "B T P J 3"]:
        B, T, P, J, _ = body_6d.shape
        shaped_model = self.body_model.with_shape(betas[:, None, :, :].expand(-1, T, -1, -1))
        posed_model = shaped_model.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=SO3.from_6d(body_6d).wxyz,
            is_only_body=True)
        return posed_model.Ts_world_joint[..., 4:]

    def geodesic_loss(
        self,
        pred: Float[Tensor, "... 4"],
        target: Float[Tensor, "... 4"],
        eps: float = 1e-7
    ) -> Float[Tensor, "..."]:
        R1 = SO3(pred).as_matrix()
        R2 = SO3(target).as_matrix()
        Rt = R1.transpose(-1, -2) @ R2
        cos_t = ((Rt[..., 0, 0] + Rt[..., 1, 1] + Rt[..., 2, 2]) - 1.0) * 0.5
        cos_t = torch.clamp(cos_t, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_t)
        return theta

    def geodesic_chordal_loss(
        self,
        pred: Float[Tensor, "... 4"],
        target: Float[Tensor, "... 4"],
    ) -> Float[Tensor, "..."]:
        R1 = SO3(pred).as_matrix()
        R2 = SO3(target).as_matrix()
        Rt = R1.transpose(-1, -2) @ R2
        tr = Rt[..., 0, 0] + Rt[..., 1, 1] + Rt[..., 2, 2]
        c = (tr - 1.0) * 0.5
        return 1.0 - c

    def _calc_loss(
        self,
        model_output: Float[Tensor, "B T P D"],
        target: Float[Tensor, "B T P D"],
        **kwargs
    ) -> Float[Tensor, "B T P"]:
        B, T, P, _ = model_output.shape
        loss = {}
        loss_mask = {}

        model_output_unpacked = MotionToken.unpack(model_output, is_cat=kwargs["is_token_cat"])
        target_unpacked = MotionToken.unpack(target, is_cat=kwargs["is_token_cat"])
        x_start_pred_unpacked = MotionToken.unpack(kwargs["x_start_pred"], is_cat=kwargs["is_token_cat"])

        loss["pose_latent"] = self._loss_func(
            model_output_unpacked.pose_latent, target_unpacked.pose_latent, reduction="none"
        ).mean(dim=(-1))
        loss_mask["pose_latent"] = None

        loss["self_partner_transforms"] = self._loss_func(
            model_output_unpacked.canonical_self_partner_transforms,
            target_unpacked.canonical_self_partner_transforms,
            reduction="none"
        ).mean(dim=(-1, -2))
        loss_mask["self_partner_transforms"] = None

        loss["tm1_t_transforms"] = self._loss_func(
            model_output_unpacked.canonical_tm1_t_transforms,
            target_unpacked.canonical_tm1_t_transforms,
            reduction="none"
        ).mean(dim=(-1))
        loss_mask["tm1_t_transforms"] = None

        if self._loss_weight.get("canonical_consistency_translations", 0) > 0 or self._loss_weight.get("canonical_consistency_rotations", 0) > 0:
            def pool_T_to_Tz(x_T, reduce='mean'):
                B, T, P = x_T.shape
                Tz = T // 4
                x_T = x_T.view(B, Tz, 4, P)
                if reduce == 'mean':
                    return x_T.mean(dim=2)
                elif reduce == 'sum':
                    return x_T.sum(dim=2)
                elif reduce == 'max':
                    return x_T.max(dim=2).values

            self_partner_transform_pred_raw = x_start_pred_unpacked.canonical_self_partner_transforms
            self_partner_transform_pred_raw = rearrange(self_partner_transform_pred_raw, "b tz s p (f d) -> b (tz f) s p d", f=4)
            self_partner_transform_pred = self._denormalize(self_partner_transform_pred_raw, "T_self_canonical_partner_canonical")
            self_partner_transform_pred = SE3.from_9d(self_partner_transform_pred).wxyz_xyz

            tm1_t_transform_pred_raw = x_start_pred_unpacked.canonical_tm1_t_transforms
            tm1_t_transform_pred_raw = rearrange(tm1_t_transform_pred_raw, "b tz s (f d) -> b (tz f) s d", f=4)
            tm1_t_transform_pred = self._denormalize(tm1_t_transform_pred_raw, "T_canonical_tm1_canonical_t")
            tm1_t_transform_pred = SE3.from_9d(tm1_t_transform_pred).wxyz_xyz
            self_partner_transform_from_tm1_t_list = []
            for i in range(P):
                sp_transfrom = (SE3(tm1_t_transform_pred[:, 1:, i].unsqueeze(2)).inverse() @ SE3(self_partner_transform_pred[:, :-1, i]) @ SE3(tm1_t_transform_pred[:, 1:, torch.arange(P) != i])).wxyz_xyz
                self_partner_transform_from_tm1_t_list.append(sp_transfrom)
            self_partner_transform_from_tm1_t = torch.stack(self_partner_transform_from_tm1_t_list, dim=2)

            transl_loss = self._loss_func(
                self_partner_transform_from_tm1_t[..., 4:],
                self_partner_transform_pred[:, 1:, :, :, 4:],
                reduction="none"
            ).mean(dim=(-1, -2))

            rot_loss = self.geodesic_chordal_loss(
                self_partner_transform_from_tm1_t[..., :4],
                self_partner_transform_pred[:, 1:, :, :, :4]
            ).mean(dim=(-1))

            transl_loss = torch.cat([torch.zeros_like(transl_loss[:, :1]), transl_loss], dim=1)
            rot_loss = torch.cat([torch.zeros_like(rot_loss[:, :1]), rot_loss], dim=1)

            c = torch.ones_like(transl_loss)
            c[:, 0] = 0.
            c = pool_T_to_Tz(c, reduce='sum')
            mask = (c > 0).float()

            loss["canonical_consistency_translations"] = pool_T_to_Tz(transl_loss, reduce="sum") / c
            loss["canonical_consistency_rotations"] = pool_T_to_Tz(rot_loss, reduce="sum") / c
            loss_mask["canonical_consistency_translations"] = mask
            loss_mask["canonical_consistency_rotations"] = mask

        loss_weight = {}
        for k, v in self._loss_weight.items():
            loss_weight[k] = v if k in ["pose_latent", "self_partner_transforms", "tm1_t_transforms"] else v * kwargs["warmup_w"]

        loss["warmup_w"] = kwargs["warmup_w"]

        return loss, loss_mask, loss_weight

    def _denormalize(self, x: Float[Tensor, "..."], elem_name: str):
        idx = StdMeanIdx()
        idx_list = getattr(idx, elem_name)
        x_denom = x * self.std[idx_list] + self.mean[idx_list]
        return x_denom

    def _normalize(self, x: Float[Tensor, "..."], elem_name):
        idx = StdMeanIdx()
        idx_list = getattr(idx, elem_name)
        x_norm = (x - self.mean[idx_list]) / self.std[idx_list]
        return x_norm

    # ------------------------------------------------------------------ #
    #  Helper: prepare data for sampling tasks                            #
    # ------------------------------------------------------------------ #

    def _prepare_sampling_data(self, sampling_config, data):
        """Build context_data, gt_data, token_kwargs, token_kwargs_gt for the given task.

        Returns a dict with keys depending on the task.  Always contains
        'context_data' and 'token_kwargs'.  Tasks that need GT also return
        'gt_data' and 'token_kwargs_gt'.
        """
        sampling_num = sampling_config.sampling_num
        context_seq_len = sampling_config.context_seq_len
        if context_seq_len % 4 != 0:
            context_seq_len = context_seq_len - context_seq_len % 4
        task = sampling_config.sampling_task

        result = {}

        if task in ("motion_control", "partner_inpainting"):
            T, P = data.betas.shape[:2]
            T = min(T, sampling_config.sampling_seq_len)
            if T % 4 != 0:
                T = T - T % 4

            kwargs_for_data, kwargs_for_gt_data = {}, {}
            for field in dataclasses.fields(data):
                value = getattr(data, field.name)
                value_gt = torch.clone(value[:T])
                value_ctx = value[:context_seq_len]
                kwargs_for_gt_data[field.name] = value_gt[None].expand((sampling_num,) + (-1,) * value_gt.ndim)
                kwargs_for_data[field.name] = value_ctx[None].expand((sampling_num,) + (-1,) * value_ctx.ndim)

            result["context_data"] = TrainingData(**kwargs_for_data)
            result["gt_data"] = TrainingData(**kwargs_for_gt_data)
            result["token_kwargs"] = self.get_cond(result["context_data"])
            result["token_kwargs_gt"] = self.get_cond(result["gt_data"])

        elif task in ("motion_control_live", "archumanoid", "partner_prediction"):
            P = data.betas.shape[1]
            T = min(sampling_config.sampling_seq_len, data.betas.shape[0])
            if T % 4 != 0:
                T = T - T % 4

            min_ctx = 4
            if context_seq_len < min_ctx:
                raise ValueError(
                    f"context_seq_len={context_seq_len} is too short for VQ-VAE encoder "
                    f"(minimum {min_ctx}). Check your config."
                )

            kwargs_for_data, kwargs_for_gt_data = {}, {}
            for field in dataclasses.fields(data):
                value = getattr(data, field.name)
                value_gt = torch.clone(value[:T])
                value_ctx = value[:context_seq_len]
                kwargs_for_gt_data[field.name] = value_gt[None].expand((sampling_num,) + (-1,) * value_gt.ndim)
                kwargs_for_data[field.name] = value_ctx[None].expand((sampling_num,) + (-1,) * value_ctx.ndim)

            result["context_data"] = TrainingData(**kwargs_for_data)
            result["gt_data"] = TrainingData(**kwargs_for_gt_data)
            result["token_kwargs"] = self.get_cond(result["context_data"])
            result["token_kwargs_gt"] = self.get_cond(result["gt_data"])

        elif task == "inbetweening":
            kwargs_for_data, kwargs_for_gt_data = {}, {}
            T = sampling_config.sampling_seq_len
            if T % 4 != 0:
                T = T - T % 4

            key_frame_indices = sampling_config.inbetweening_key_frame_indices
            if not all(0 <= idx < T for idx in key_frame_indices):
                raise ValueError(f"Key frame indices must be in the range of T: {T}")
            if len(key_frame_indices) != len(set(key_frame_indices)):
                raise ValueError("Key frame indices must be unique")

            device = data.betas.device
            Tz = T // 4
            P = data.betas.shape[1]

            # GT data
            for field in dataclasses.fields(data):
                value = getattr(data, field.name)
                value_gt = torch.clone(value[:T])
                kwargs_for_gt_data[field.name] = value_gt[None].expand((sampling_num,) + (-1,) * value_gt.ndim)
            gt_data = TrainingData(**kwargs_for_gt_data)

            # Encode each key frame chunk separately
            encoded_tokens = {}
            for key_frame_idx in key_frame_indices:
                frame_chunk_indices = list(range(key_frame_idx, key_frame_idx + 4))
                token_position = key_frame_idx // 4

                chunk_kwargs = {}
                for field in dataclasses.fields(data):
                    value = getattr(data, field.name)
                    chunk_value = value[frame_chunk_indices]
                    if not isinstance(chunk_value, torch.Tensor):
                        chunk_value = torch.tensor(chunk_value, device=device)
                    chunk_kwargs[field.name] = chunk_value[None].expand((sampling_num,) + (-1,) * chunk_value.ndim)
                chunk_data = TrainingData(**chunk_kwargs)

                with torch.no_grad():
                    chunk_pose_latent, _ = self.encode(chunk_data)
                    encoded_tokens[token_position] = chunk_pose_latent

            B = sampling_num
            D = encoded_tokens[list(encoded_tokens.keys())[0]].shape[-1]
            full_pose_latent = torch.full((B, Tz, P, D), -1.0, device=device, dtype=torch.float32)
            full_mask = torch.zeros((B, Tz, P), device=device, dtype=torch.bool)

            for token_pos, token_value in encoded_tokens.items():
                full_pose_latent[:, token_pos, :, :] = token_value.squeeze(1)
                full_mask[:, token_pos, :] = True

            # Full-sequence context data for get_cond
            for field in dataclasses.fields(data):
                value = getattr(data, field.name)[:T]
                kwargs_for_data[field.name] = value[None].expand((sampling_num,) + (-1,) * value.ndim)
            context_data = TrainingData(**kwargs_for_data)

            result["context_data"] = context_data
            result["gt_data"] = gt_data
            result["token_kwargs"] = self.get_cond(context_data)
            result["token_kwargs_gt"] = self.get_cond(gt_data)
            result["inbetweening_pose_latent"] = full_pose_latent
            result["inbetweening_mask"] = full_mask
            result["inbetweening_gt_data"] = gt_data
            result["inbetweening_key_frame_indices"] = key_frame_indices

        else:
            # Default: joint, next_token, agentic_turn_taking, agentic_sync
            kwargs_for_data = {}
            T, P = data.betas.shape[:2]
            gt_seq_len = sampling_config.sampling_seq_len
            actual_ctx_len = min(context_seq_len, T)
            actual_ctx_len = (actual_ctx_len // 4) * 4
            min_ctx = 16
            if actual_ctx_len < min_ctx:
                raise ValueError(
                    f"Sequence too short for VQ-VAE encoder: T={T}, "
                    f"actual_ctx_len={actual_ctx_len} < {min_ctx}"
                )
            for field in dataclasses.fields(data):
                value = getattr(data, field.name)[:actual_ctx_len]
                kwargs_for_data[field.name] = value[None].expand((sampling_num,) + (-1,) * value.ndim)
            result["context_data"] = TrainingData(**kwargs_for_data)
            result["token_kwargs"] = self.get_cond(result["context_data"])
            result["gt_seq_len"] = gt_seq_len

        result["context_seq_len"] = context_seq_len
        return result

    def sample_sequence(
        self,
        sampling_config: DFOTSamplingConfig,
        data: TrainingData | None = None,
    ):
        context_seq_len = sampling_config.context_seq_len
        if context_seq_len % 4 != 0:
            context_seq_len = context_seq_len - context_seq_len % 4
        context_latent_seq_len = context_seq_len // 4
        print(f'context_seq_len: {context_seq_len}')

        prep = self._prepare_sampling_data(sampling_config, data)
        context_data = prep["context_data"]
        token_kwargs = prep["token_kwargs"]
        task = sampling_config.sampling_task

        with torch.no_grad():
            if task == "inbetweening":
                pose_latent = prep["inbetweening_pose_latent"]
                mask = prep["inbetweening_mask"]
            else:
                pose_latent, mask = self.encode(context_data)

            token_kwargs["pose_latent"] = pose_latent
            motion_tokens = MotionToken(**token_kwargs)
            val = motion_tokens.pack()

            P = context_data.betas.shape[2]

            # Encode GT for tasks that need it
            def _encode_gt():
                gt_data = prep["gt_data"]
                token_kwargs_gt = prep["token_kwargs_gt"]
                pose_latent_gt, _ = self.encode(gt_data)
                token_kwargs_gt["pose_latent"] = pose_latent_gt
                return MotionToken(**token_kwargs_gt).pack()

            if task in ("joint", "next_token"):
                x_pred = super()._ar_sample_sequence(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                )
            elif task == "agentic_turn_taking":
                x_pred = self._ar_agentic_turn_taking_sequence(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                    gt_seq_len=prep["gt_seq_len"],
                )
            elif task == "agentic_sync":
                x_pred = self._ar_agentic_sync_sequence(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                    gt_seq_len=prep["gt_seq_len"],
                )
            elif task == "motion_control":
                x_pred = super()._ar_sample_sequence_motion_control(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                    motion_tokens_gt=_encode_gt(),
                )
            elif task == "motion_control_live":
                x_pred = super()._ar_sample_sequence_motion_control_live(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                    motion_tokens_gt=_encode_gt(),
                )
            elif task == "partner_prediction":
                x_pred = super()._ar_sample_sequence_partner_prediction(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                    motion_tokens_gt=_encode_gt(),
                )
            
            elif task == "inbetweening":
                x_pred = super()._ar_sample_sequence_inbetweening(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                )
            elif task == "partner_inpainting":
                x_pred = super()._ar_sample_sequence_partner_inpainting(
                    sampling_config=sampling_config,
                    context=val,
                    context_mask=mask,
                    betas=context_data.betas[:, 0],
                    motion_tokens_gt=_encode_gt(),
                )
            else:
                raise ValueError(f"Unknown sampling task: {task}")

            pred_motion_tokens = MotionToken.unpack(x_pred)
            pred_motion_tokens_dn = MotionToken.denormalize(pred_motion_tokens, self.mean, self.std)

            if sampling_config.root_transform_mode == "temporal":
                tm1_t_pred_raw = pred_motion_tokens.canonical_tm1_t_transforms
                Tz = tm1_t_pred_raw.shape[1]
                dec_cond = torch.cat([context_data.betas[:, :1].expand(-1, Tz, P, -1), tm1_t_pred_raw], dim=-1)
                tm1_t_pred_raw = rearrange(tm1_t_pred_raw, "b tz s (f d) -> b (tz f) s d", f=4)
                T = tm1_t_pred_raw.shape[1]
                enc_cond = torch.cat([context_data.betas[:, :1].expand(-1, T, P, -1), tm1_t_pred_raw], dim=-1)

            elif sampling_config.root_transform_mode == "temporal_partner":
                tm1_t_pred_dn = pred_motion_tokens_dn.canonical_tm1_t_transforms
                tm1_t_pred_dn = rearrange(tm1_t_pred_dn, "b tz s (f d) -> b (tz f) s d", f=4)
                self_partner_pred_dn = pred_motion_tokens_dn.canonical_self_partner_transforms
                self_partner_pred_dn = rearrange(self_partner_pred_dn, "b t s p (f d) -> b (t f) s p d", f=4)
                partner_tm1_t_dn = (SE3.from_9d(self_partner_pred_dn[:, :-1, 0]).inverse() @ SE3.from_9d(tm1_t_pred_dn[:, 1:, :1]) @ SE3.from_9d(self_partner_pred_dn[:, 1:, 0])).as_9d()
                partner_tm1_t_dn = torch.cat([tm1_t_pred_dn[:, :1, 1:], partner_tm1_t_dn], dim=1)

                # Restore Agent B's original GT root for context frames to avoid numerical drift
                Tc_frames = context_latent_seq_len * 4
                partner_tm1_t_dn[:, :Tc_frames] = tm1_t_pred_dn[:, :Tc_frames, 1:]

                tm1_t_pred_tp_dn = torch.cat([tm1_t_pred_dn[:, :, :1], partner_tm1_t_dn], dim=2)
                tm1_t_pred_tp = self._normalize(tm1_t_pred_tp_dn, "T_canonical_tm1_canonical_t")
                tm1_t_pred_tp_r = rearrange(tm1_t_pred_tp, "b (t f) s d -> b t s (f d)", f=4)

                # Use the model's own predicted tm1_t for Agent B as decode conditioning.
                # The VQ-VAE was trained with Agent B's own tm1_t, not the temporal_partner-
                # derived value. Using the temporal_partner formula introduces a
                # denormalize->SE3->normalize round-trip that changes dec_cond enough for
                # the decoder to snap to wrong codebook entries (causing noisy output in the
                # first predicted chunk). By using the original normalized values for ALL
                # Agent B frames, the decoder gets conditioning consistent with training.
                # The temporal_partner-derived tm1_t is still used for root trajectory
                # reconstruction via enc_cond.
                original_tm1_t_norm = pred_motion_tokens.canonical_tm1_t_transforms
                # Token space (dec_cond): restore ALL Agent B tokens, not just context
                tm1_t_pred_tp_r[:, :, 1:] = original_tm1_t_norm[:, :, 1:]
                # Frame space (enc_cond): restore ALL Agent B frames, not just context
                original_tm1_t_norm_frames = rearrange(
                    original_tm1_t_norm[:, :, 1:],
                    "b tz s (f d) -> b (tz f) s d", f=4
                )
                tm1_t_pred_tp[:, :, 1:] = original_tm1_t_norm_frames

                Tz = tm1_t_pred_tp_r.shape[1]
                dec_cond = torch.cat([context_data.betas[:, :1].expand(-1, Tz, P, -1), tm1_t_pred_tp_r], dim=-1)
                T = tm1_t_pred_tp.shape[1]
                enc_cond = torch.cat([context_data.betas[:, :1].expand(-1, T, P, -1), tm1_t_pred_tp], dim=-1)
            else:
                raise ValueError(f"Unknown root_transform_mode: {sampling_config.root_transform_mode}")

            pose_z = pred_motion_tokens.pose_latent
            pose_val = self.decode(pose_z, dec_cond)
            pose_tokens = MultiPoseToken.unpack(pose_val, enc_cond)

            # For inbetweening, decode keyframes separately without quantization to ensure exact match
            if task == "inbetweening":
                _inbetweening_gt_data = prep["inbetweening_gt_data"]
                _inbetweening_pose_latent = prep["inbetweening_pose_latent"]
                _inbetweening_key_frame_indices = prep["inbetweening_key_frame_indices"]

                original_pose_tokens_single = PoseToken.convert_from_training_data(_inbetweening_gt_data)
                P_inb = _inbetweening_gt_data.betas.shape[2]
                original_pose_tokens_single.convert_to_mutli_pose_token(person_num=P_inb)

                for key_frame_idx in _inbetweening_key_frame_indices:
                    token_position = key_frame_idx // 4
                    frame_slice = slice(key_frame_idx, key_frame_idx + 4)

                    original_keyframe_latent = _inbetweening_pose_latent[:, token_position:token_position + 1, :, :]
                    keyframe_dec_cond = dec_cond[:, token_position:token_position + 1, :, :]
                    keyframe_pose_val = self.decode_wo_quantize(original_keyframe_latent, keyframe_dec_cond)

                    keyframe_enc_cond = enc_cond[:, frame_slice, :, :]
                    keyframe_pose_tokens = MultiPoseToken.unpack(keyframe_pose_val, keyframe_enc_cond)

                    for field in dataclasses.fields(pose_tokens):
                        field_name = field.name
                        if not hasattr(keyframe_pose_tokens, field_name):
                            continue
                        keyframe_value = getattr(keyframe_pose_tokens, field_name)
                        decoded_value = getattr(pose_tokens, field_name)

                        if (isinstance(decoded_value, torch.Tensor) and isinstance(keyframe_value, torch.Tensor)
                                and decoded_value.shape[1] >= key_frame_idx + 4
                                and keyframe_value.shape[1] == 4
                                and field_name != "betas"):
                            decoded_value[:, frame_slice, ...] = keyframe_value[:, :, ...]

            self_partner_transform = pred_motion_tokens_dn.canonical_self_partner_transforms
            self_partner_transform = rearrange(self_partner_transform, "b t s p (f d) -> b (t f) s p d", f=4)

        return pose_tokens, self_partner_transform


# ------------------------------------------------------------------ #
#  Standalone utility (not called during sampling)                     #
# ------------------------------------------------------------------ #

def viz_noise(schedule: Float[Tensor, "height token_len"], title: str = ""):
    """Visualize a noise schedule matrix and save to ./tmp/."""
    import matplotlib.pyplot as plt

    schedule = schedule.detach().cpu().numpy() if isinstance(schedule, torch.Tensor) else schedule
    plt.figure(figsize=(12, 5))
    plt.imshow(schedule, aspect="auto", cmap="viridis")
    plt.colorbar(label="Noise Level")
    plt.xlabel("Token Index (A/B alternating)")
    plt.ylabel("Denoising Step")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"./tmp/noise_schedule_{title.replace(' ', '_')}.png")
    plt.close()
    print(f"Saved noise schedule to ./tmp/noise_schedule_{title.replace(' ', '_')}.png")