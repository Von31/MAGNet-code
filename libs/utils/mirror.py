from jaxtyping import Float
from typing import Literal
from torch import Tensor

BODY_ROTATION_PERMUTATION = [1, 0, 2, 4, 3,
                            5, 7, 6, 8, 10,
                            9, 11, 13, 12,
                            14, 16, 15, 18, 17,
                            20, 19]
HAND_ROTATION_PERMUTATION = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

def mirror_augment(
  joint_rotations: Float[Tensor, "... 21 4"],
  T_world_root: Float[Tensor, "... 7"],
) -> tuple[Float[Tensor, "... 21 4"], Float[Tensor, "... 7"]]:
  aug_joint_rotations = joint_rotations[..., BODY_ROTATION_PERMUTATION, :]
  aug_joint_rotations[..., 2:4] *= -1

  aug_T_world_root = T_world_root.clone()
  aug_T_world_root[..., 2:5] *= -1

  return aug_joint_rotations, aug_T_world_root

def mirror_rotations_6d(
  joint_rotations: Float[Tensor, "... 21 6"],
  mode: Literal["body", "hand"] = "body",
) -> Float[Tensor, "... 12 6"]:
  aug_joint_rotations = joint_rotations.clone()
  if mode == "body":
    aug_joint_rotations = aug_joint_rotations[..., BODY_ROTATION_PERMUTATION, :]
  elif mode == "hand":
    aug_joint_rotations = aug_joint_rotations[..., HAND_ROTATION_PERMUTATION, :]
  aug_joint_rotations[..., 1:4] *= -1
  return aug_joint_rotations

def mirror_transforms_9d(
  transforms: Float[Tensor, "... 9"]
) -> Float[Tensor, "... 9"]:
  aug_transforms = transforms.clone()
  aug_transforms[..., 1:4] *= -1
  aug_transforms[..., 6] *= -1
  return aug_transforms