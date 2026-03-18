import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int, Bool
from typing import Literal, Dict, Tuple

from libs.model.dfot.config import DFOTConfig
from libs.model.dfot.diffusion_transformer import TransformerModel
from libs.utils.noise_schedule import cosine_schedule


def extract(a, t, x_shape):
    t = t.long()
    shape = t.shape
    out = a[t]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))

class DiscreteDiffusion(nn.Module):
    def __init__(
        self, 
        config: DFOTConfig, 
        model: TransformerModel,
    ):
        super(DiscreteDiffusion, self).__init__()

        self.config = config
        self.model = model
        self._make_schedule()

    def _reg(self, name, t): self.register_buffer(name, t.float(), persistent=False)

    def _make_schedule(self):
        sch = cosine_schedule(timesteps=self.config.max_t)

        self._reg('betas', sch['betas'])
        self._reg('alphas', sch['alphas'])
        self._reg('alphas_cumprod', sch['alphas_cumprod'])
        self._reg('alphas_cumprod_prev', F.pad(sch['alphas_cumprod'][:-1], (1, 0), value=1.0))

        self._reg('sqrt_alphas_cumprod', torch.sqrt(sch['alphas_cumprod']))
        self._reg('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - sch['alphas_cumprod']))
        self._reg('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / sch['alphas_cumprod']))
        self._reg('sqrt_one_minus_times_sqrt_recip', 
            self.sqrt_recip_alphas_cumprod * self.sqrt_one_minus_alphas_cumprod)


        self._reg('posterior_variance', self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self._reg('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min=1e-20)))
        self._reg('posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self._reg('posterior_mean_coef2', (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

        self._reg('snr', self.alphas_cumprod / (1.0 - self.alphas_cumprod))
        self._reg('min_snr_weight', torch.minimum(self.snr, torch.tensor(5.)) / (self.snr + 1))


    def model_predictions(
        self, 
        x: Float[Tensor, "B T P D"], 
        k: Int[Tensor, "B T P"],
        cond: Float[Tensor, "B T P C"] | None = None,
    ) -> Dict[str, Float[Tensor, "B T P D"]]:
        model_output = self.model(x, k, cond)
        return self.model_output_process(x, k, model_output)
    
    def model_output_process(
        self,
        x: Float[Tensor, "B T P D"],
        k: Int[Tensor, "B T P"],
        model_output: Float[Tensor, "B T P D"],
    ) -> Dict[str, Float[Tensor, "B T P D"]]:
        if self.config.diffusion_objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.config.diffusion_objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)
        
        elif self.config.diffusion_objective == "pred_v":
            x_start = self.predict_start_from_v(x, k, model_output)
            pred_noise = self.predict_noise_from_v(x, k, model_output)

        return {'model_output': model_output, 'x_start': x_start, 'noise': pred_noise}
    

    def forward(
        self, 
        x: Float[Tensor, "B T P D"], 
        k: Int[Tensor, "B T P"],
        cond: Float[Tensor, "B T P C"] | None = None):

        noise = torch.randn_like(x)

        noised_x = self.q_sample(x_start=x, k=k, noise=noise)

        model_pred = self.model_predictions(x=noised_x, k=k, cond=cond)

        if self.config.diffusion_objective == "pred_noise":
            target = noise
        elif self.config.diffusion_objective == "pred_x0":
            target = x
        elif self.config.diffusion_objective == "pred_v":
            target = self.calc_v_from_x0_and_noise(x, k, noise)
        else:
            raise ValueError(f"Invalid diffusion objective: {self.config.diffusion_objective}")

        snr_weight = extract(self.min_snr_weight, k, x.shape)
        
        return {'model_output': model_pred['model_output'], 'target': target.detach(), 'x_start_pred': model_pred['x_start'], 'snr_weight': snr_weight}

    def forward_mixin(
        self, 
        x: Float[Tensor, "B T P D"], 
        k: Int[Tensor, "B T P"],
        mix: Bool[Tensor, "B T P"]):

        noise = torch.randn_like(x)

        noised_x = self.q_sample(x_start=x, k=k, noise=noise)
        noised_x = torch.where(mix[..., None], x, noised_x)

        model_pred = self.model_predictions(x=noised_x, k=k)

        if self.config.diffusion_objective == "pred_noise":
            target = noise
        elif self.config.diffusion_objective == "pred_x0":
            target = x
        elif self.config.diffusion_objective == "pred_v":
            target = self.calc_v_from_x0_and_noise(x, k, noise)
        else:
            raise ValueError(f"Invalid diffusion objective: {self.config.diffusion_objective}")

        snr_weight = extract(self.min_snr_weight, k, x.shape)
        
        return {'model_output': model_pred['model_output'], 'target': target.detach(), 'x_start_pred': model_pred['x_start'], 'snr_weight': snr_weight}

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_one_minus_times_sqrt_recip, k, x_k.shape) * noise
        )
    
    def predict_noise_from_start(self, x_k, k, x0):
        return (x_k - extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0) / extract(
            self.sqrt_one_minus_alphas_cumprod, k, x_k.shape
        )

    def predict_start_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
         + extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
        )

    def calc_v_from_x0_and_noise(self, x0, k, noise):
        return (
            extract(self.sqrt_alphas_cumprod, k, x0.shape).to(x0.dtype) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x0.shape).to(x0.dtype) * x0
        )

    def q_sample(
        self, 
        x_start: Float[Tensor, "B T P D"], 
        k: Int[Tensor, "B T P"],
        noise: Float[Tensor, "B T P D"],
    ) -> Float[Tensor, "B T P D"]:
        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )


    @torch.inference_mode()
    def sample_step(
        self,
        x: Float[Tensor, "B T P D"],
        curr_noise_level: Int[Tensor, "B T P"],
        next_noise_level: Int[Tensor, "B T P"],
        model_output: Float[Tensor, "B T P D"] | None = None,
        denoising_process: Literal["ddim", "ddpm"] = "ddim",
        ddim_eta: float = 0.0,
        noise_tau: float = 1.0,
    ) -> Float[Tensor, "B T P D"]:
        if denoising_process == "ddim":
            return self.ddim_sample_step(x, curr_noise_level, next_noise_level, model_output, ddim_eta, noise_tau)
        elif denoising_process == "ddpm":
            return self.ddpm_sample_step(x, curr_noise_level, model_output)
        else:
            raise ValueError(f"Invalid denoising process: {denoising_process}")


    def ddpm_sample_step(
        self,
        x: Float[Tensor, "B T P D"],
        curr_noise_level: Int[Tensor, "B T P"],
        model_output: Float[Tensor, "B T P D"] | None = None,
    ) -> Float[Tensor, "B T P D"]:
        
        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            k=clipped_curr_noise_level,
            model_output=model_output,
        )

        noise = torch.where(
            clipped_curr_noise_level[..., None] > 0,
            torch.randn_like(x),
            0.,
        )
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(curr_noise_level[..., None] == -1, x, x_pred)

    def ddim_sample_step(
        self,
        x: Float[Tensor, "B T P D"],
        curr_noise_level: Int[Tensor, "B T P"],
        next_noise_level: Int[Tensor, "B T P"],
        model_output: Float[Tensor, "B T P D"] | None = None,
        ddim_eta: float = 0.0,
        noise_tau: float = 1.0,
    ) -> Float[Tensor, "B T P D"]:

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level.long()]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level, dtype=self.alphas_cumprod.dtype),
            self.alphas_cumprod[next_noise_level.long()],
        )
        # eps = torch.finfo(self.alphas_cumprod.dtype).eps
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level, dtype=self.alphas_cumprod.dtype),
            ddim_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
            # * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha + eps)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).clamp(min=0.0).sqrt()

        if model_output is None:
            model_pred = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level
            )
        else:
            model_pred = self.model_output_process(
                x=x,
                k=clipped_curr_noise_level,
                model_output=model_output,
            )

        x_start = model_pred['x_start']
        pred_noise = model_pred['noise']

        noise = torch.randn_like(x)
        noise = noise * noise_tau

        x_pred = x_start * alpha_next[..., None].sqrt() + pred_noise * c[..., None] + sigma[..., None] * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            mask[..., None],
            x,
            x_pred,
        )

        return x_pred

    def p_mean_variance(
        self, 
        x: Float[Tensor, "B T P D"], 
        k: Int[Tensor, "B T P"],
        model_output: Float[Tensor, "B T P D"] | None = None,
    ) -> Tuple[Float[Tensor, "B T P D"], Float[Tensor, "B T P D"], Float[Tensor, "B T P D"]]:
        if model_output is None:
            model_pred = self.model_predictions(x, k)
        else:
            model_pred = self.model_output_process(x, k, model_output)

        x_start = model_pred['x_start']
        return self.q_posterior(x_start=x_start, x_k=x, k=k)
    

    def q_posterior(
        self, 
        x_start: Float[Tensor, "B T P D"], 
        x_k: Float[Tensor, "B T P D"], 
        k: Int[Tensor, "B T P"]
    ) -> Tuple[Float[Tensor, "B T P D"], Float[Tensor, "B T P D"], Float[Tensor, "B T P D"]]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = extract(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, k, x_k.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


class ContinuousDiffusion(DiscreteDiffusion):
    def __init__(
        self, 
        config: DFOTConfig,
        model: TransformerModel,
    ):
        super(ContinuousDiffusion, self).__init__(config, model)
        pass

    def forward(
        self, 
        x: Float[Tensor, "B T P D"], 
        k: Float[Tensor, "B T P"]):
        pass

