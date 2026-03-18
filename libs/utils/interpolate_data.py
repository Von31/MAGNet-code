from jaxtyping import Float, Bool
import torch
from torch import Tensor
import libs.utils.transforms as tf

def interpolate_rotation(
    rotations: tf.SO3, 
    src_fps: float, 
    tgt_fps: float, 
    tpose_mask: Bool[Tensor, "T"]| None = None
) -> tf.SO3:
    device = rotations.wxyz.device
    T = rotations.wxyz.shape[0]
    dim = rotations.wxyz.dim()-1
    tgt_T = int(T * tgt_fps / src_fps)
    
    # Time steps
    org_times = torch.arange(T, device=device) / src_fps
    new_times = torch.linspace(0, org_times[-1], tgt_T, device=device)

    if tpose_mask is not None:
        org_times = org_times[tpose_mask]

    # Prepare interpolation indices
    idxs = torch.searchsorted(org_times, new_times, right=True).clamp(1, T-1)
    t0 = org_times[idxs - 1]
    t1 = org_times[idxs]
    alpha = ((new_times - t0) / (t1 - t0)).reshape(*((-1,) + (1,) * dim))  # Shape: [target_T, ..., 1]

    # Select corresponding quaternions
    q0 = tf.SO3(rotations.wxyz[idxs - 1]) # [target_T, ..., 4]
    q1 = tf.SO3(rotations.wxyz[idxs])     # [target_T, ..., 4]

    # Perform SLERP
    interp_quats = tf.SO3.slerp(alpha, q0, q1)

    return interp_quats

def interpolate_translation(
    translations: Float[Tensor, "T ... 3"], 
    src_fps: float, 
    tgt_fps: float,
    tpose_mask: Bool[Tensor, "T"]| None = None
) -> Float[Tensor, "T ... 3"]:
    device = translations.device
    T = translations.shape[0]
    dim = translations.dim()-1
    tgt_T = int(T * tgt_fps / src_fps)
    
    # Time steps
    org_times = torch.arange(T, device=device) / src_fps
    new_times = torch.linspace(0, org_times[-1], tgt_T, device=device)

    if tpose_mask is not None:
        org_times = org_times[tpose_mask]

    # Prepare interpolation indices
    idxs = torch.searchsorted(org_times, new_times, right=True).clamp(1, T-1)
    t0 = org_times[idxs - 1]
    t1 = org_times[idxs]
    alpha = ((new_times - t0) / (t1 - t0)).reshape(*((-1,) + (1,) * dim)) # Shape: [target_T, ..., 1]

    # Select corresponding translations
    t0 = translations[idxs - 1]  # [target_T, ..., 3]
    t1 = translations[idxs]      # [target_T, ..., 3]

    interp_trans = t0 + alpha * (t1 - t0)

    return interp_trans
    


    