from enum import Enum
from pathlib import Path

from typing import List
import dataclasses
from libs.utils.tensor_dataclass import TensorDataclass 

class DataType(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class DatasetName(Enum):
    # boxing
    DUOBOX = 0

    # dancing
    DD100 = 1
    REMOCAP = 2

    # social interaction
    INTERX = 3
    EMBODY3DSOLO = 4
    EMBODY3DDUO = 5
    EMBODY3DTRIO = 6
    EMBODY3DQUAD = 7



DATASET_NPZ_FILE_DICT = {
    DatasetName.DUOBOX: Path("/home/kieh/dataset/duobox/processed/"),
    DatasetName.DD100: Path("/home/kieh/dataset/dd100/motion/smplx/processed/"),
    DatasetName.REMOCAP: Path("/home/kieh/workspace/ReMoS/data/Lindyhop/smplx_data_pose_prior_001"),
    DatasetName.INTERX: Path("/datasets/inter_x/current/smplx/motions/"),
    DatasetName.EMBODY3DSOLO: Path("/datasets/embody-3d/current/datasets_processed/solo/"),
    DatasetName.EMBODY3DDUO: Path("/datasets/embody-3d/current/datasets_processed/duo/"),
    DatasetName.EMBODY3DTRIO: Path("/datasets/embody-3d/current/datasets_processed/trio/"),
    DatasetName.EMBODY3DQUAD: Path("/datasets/embody-3d/current/datasets_processed/quad/"),
}

DATASET_FPS = {
    DatasetName.DUOBOX: 30,
    DatasetName.DD100: 30,
    DatasetName.REMOCAP: 50,
    DatasetName.INTERX: 120,
    DatasetName.EMBODY3DSOLO: 30,
    DatasetName.EMBODY3DDUO: 30,
    DatasetName.EMBODY3DTRIO: 30,
    DatasetName.EMBODY3DQUAD: 30,
}

class StdMeanIdx(TensorDataclass):
    betas: List[int] = dataclasses.field(
        default_factory=lambda: list(range(0, 10)))
    body_joint_rotations: List[int] = dataclasses.field(
        default_factory=lambda: list(range(10, 10 + 21 * 6)))
    hand_joint_rotations: List[int] = dataclasses.field(
        default_factory=lambda: list(range(10 + 21 * 6, 10 + 51 * 6)))
    T_canonical_tm1_canonical_t: List[int] = dataclasses.field(
        default_factory=lambda: list(range(10 + 51 * 6, 10 + 51 * 6 + 9)))
    T_canonical_root: List[int] = dataclasses.field(
        default_factory=lambda: list(range(10 + 51 * 6 + 9, 10 + 51 * 6 + 9 + 9)))
    T_world_root: List[int] = dataclasses.field(
        default_factory=lambda: list(range(10 + 51 * 6 + 9 + 9, 10 + 51 * 6 + 9 + 9 + 9)))
    T_world_canonical: List[int] = dataclasses.field(
        default_factory=lambda: list(range(10 + 51 * 6 + 9 + 9 + 9, 10 + 51 * 6 + 9 + 9 + 9 + 9)))
    T_self_canonical_partner_canonical: List[int] = dataclasses.field(
        default_factory=lambda: list(range(10 + 51 * 6 + 9 + 9 + 9 + 9, 10 + 51 * 6 + 9 + 9 + 9 + 9 + 9)))