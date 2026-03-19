import torch
from torch import Tensor
import numpy as np
from sklearn.cluster import DBSCAN

from pathlib import Path
from typing import List
from jaxtyping import Float, Bool

from libs.dataloaders import DatasetName, TrainingData, DATASET_FPS
from libs.utils.interpolate_data import interpolate_rotation, interpolate_translation
from libs.utils.transforms import SO3
from libs.utils import mirror
from libs.utils import fncsmpl

FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
CONTACT_VEL_THRESH = 0.005
CONTACT_HEIGHT_THRESH = 0.04

VIS_CONTACT = False
VIS_COORD = False

# rotate global transform for z up
def rotate_global_up_rotation(
    dataset_name: DatasetName,
    T_world_root: Float[Tensor, "T P 7"],
) -> Float[Tensor, "T P 7"]:
    rad_p_2 = torch.ones_like(T_world_root[..., 0]) * np.pi / 2

    if dataset_name in [DatasetName.DD100]:
        rot_org = SO3(wxyz=T_world_root[..., :4])
        rot_x = SO3.from_x_radians(-rad_p_2)
        rot = rot_x.multiply(rot_org).wxyz
        trans = rot_x @ T_world_root[..., 4:7]
        T_world_root = torch.cat([rot, trans], axis=-1)
        
    elif dataset_name in [DatasetName.REMOCAP, DatasetName.INTERX, DatasetName.EMBODY3DSOLO, DatasetName.EMBODY3DDUO, DatasetName.EMBODY3DTRIO, DatasetName.EMBODY3DQUAD]:
        pass

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    return T_world_root


def determine_floor_height_and_contacts(
    smpl_path: Path,
    betas: Float[Tensor, "T P 10"],
    T_world_root: Float[Tensor, "T P 7"],
    body_joint_rotations: Float[Tensor, "T P 21 4"],
) -> (float, Bool[Tensor, "T P 21"]):
    T, P, _ = T_world_root.shape

    body_model = fncsmpl.SmplModel.load(smpl_path).to(betas.device)

    toes_translation = np.empty((T, P, 2, 3), dtype=np.float32)
    all_translation = np.empty((T, P, 21, 3), dtype=np.float32)
    for i in range(P):
        shaped = body_model.with_shape(betas[:, i])
        fk_outputs = shaped.with_pose_decomposed(
            T_world_root=T_world_root[:, i],
            body_quats=body_joint_rotations[:, i],
        )

        # all (use 21 joints ignoring hand index 15*2)
        all_transform = fk_outputs.Ts_world_joint.numpy(force=True)
        all_translation[:, i] = all_transform[..., :21, 4:7]

        # smpl toe idx; 9, 10 (ignoring root)
        toes_translation[:, i] = all_transform[..., 9:11, 4:7]

    # floor height
    toe_heights_raw = toes_translation[..., 1].reshape(-1, 1)
    cluster_heights = []
    cluster_sizes = []
    clustering = DBSCAN(eps=0.005, min_samples=3).fit(toe_heights_raw)
    all_labels = np.unique(clustering.labels_)

    min_median = float('inf')
    for cur_label in all_labels:
        cur_clust = toe_heights_raw[clustering.labels_ == cur_label]
        cur_median = np.median(cur_clust)
        cluster_heights.append(cur_median)
        cluster_sizes.append(cur_clust.shape[0])

        if cur_median < min_median:
            min_median = cur_median

    floor_height = min_median
    offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

    # contacts
    all_velocities = np.linalg.norm(all_translation[1:] - all_translation[:-1], axis=-1)
    contacts = (all_velocities < CONTACT_VEL_THRESH) & (all_translation[1:, :, :, 1]-offset_floor_height < CONTACT_HEIGHT_THRESH)
    contacts = torch.tensor(contacts)

    if VIS_CONTACT:
        all_translation[..., 1] -= offset_floor_height

        T_world_root_ = T_world_root.clone()
        T_world_root_[..., 5] -= offset_floor_height
        T_world_root_ = T_world_root_.numpy(force=True)

        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        for j in range(P):
            for i in range(2):
                a_toes_translation = all_translation[1:, j,9+i]
                a_t_contact =  a_toes_translation[contacts[:, j, 9+i]]
                a_t_free = a_toes_translation[~contacts[:, j, 9+i]]
                ax.scatter(a_t_contact[:, 0], a_t_contact[:, 2], a_t_contact[:, 1], c='red', s=2.)
                ax.scatter(a_t_free[:, 0], a_t_free[:, 2], a_t_free[:, 1], c='blue', s=1.)

            ax.scatter(T_world_root_[:, j, 4], T_world_root_[:, j, 6], T_world_root_[:, j, 5], c='green', s=0.2)
        ax.set_aspect('equal', 'box')
        ax.view_init(elev=7., azim=25, roll=0)

    return offset_floor_height, contacts

def process_global_and_canonical_coord(
    T_world_root: Float[Tensor, "T P 7"],
    floor_height: float
) -> (Float[Tensor, "T P 7"], Float[Tensor, "T P 7"]):
    # global coord
    T_world_root[..., 5] -= floor_height
    
    # canonical coord
    mat = SO3(wxyz=T_world_root[..., :4]).as_matrix()
    ex = mat @ torch.tensor([1., 0., 0.], device=T_world_root.device, dtype=T_world_root.dtype)
    rad = -torch.atan2(ex[..., 2], ex[..., 0])
    rot = SO3.from_y_radians(rad)
    translate = T_world_root[..., 4:].clone()
    translate[..., 1] = 0.
    T_world_canonical = torch.cat([
        rot.wxyz,
        translate
    ], dim=-1)

    if VIS_COORD:
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        T_world_root_ = T_world_root.clone()
        T_world_canonical_ = T_world_canonical.clone()
        if T_world_root.ndim == 3:
            T_world_root_ = T_world_root_[:, 0]
            T_world_canonical_ = T_world_canonical_[:, 0]

        for T_world in [T_world_root_, T_world_canonical_]:
            mat = SO3(wxyz=T_world[..., :4])
            trans = T_world[..., 4:].numpy(force=True)
            for i in range(3):
                ee = torch.tensor([0., 0., 0.], device=T_world_root.device, dtype=T_world_root.dtype)
                ee[i] = 0.1
                eee = mat @ ee
                eee = eee.numpy(force=True)
                ee = ee.numpy(force=True)
                for j in range(5):
                    cl = ["red", "green", "blue"]
                    interval = 10
                    ax.plot([trans[j*interval, 0], trans[j*interval, 0] + eee[j*interval, 0]],
                            [trans[j*interval, 2], trans[j*interval, 2] + eee[j*interval, 2]],
                            [trans[j*interval, 1], trans[j*interval, 1] + eee[j*interval, 1]], c=cl[i])
                ax.plot(trans[:j*interval+1, 0], trans[:j*interval+1, 2], trans[:j*interval+1, 1], c='grey', lw=0.5)
        ax.set_aspect('equal', 'box')
        ax.view_init(elev=30., azim=120., roll=0)

    return T_world_root, T_world_canonical
  
  
def load_from_np(
    dataset_name: DatasetName,
    path: Path,
    smpl_path: Path,
    device_idx: int,
    is_mirror_augment: bool = False,
    data_fps: int = 30,
) -> List[TrainingData]:
    """Load motion data from an NPZ file.
    
    Args:
        dataset_name: Type of dataset to load
        path: Path to the NPZ file
        smpl_path: Path to SMPL model data
        device_idx: CUDA device index
        
    Returns:
        List of TrainingData
    """
    
    # Load and process raw data
    ext = path.suffix
    if ext == '.npy':
        raw_data = np.load(path, allow_pickle=True).item()
    elif ext == '.npz':
        raw_data = np.load(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    raw_fields = {
        k: torch.from_numpy(v.astype(np.float32) if v.dtype == np.float64 else v)
        for k, v in raw_data.items()
        if v.dtype in (np.float32, np.float64)
    }

    # process person_dim
    for field in raw_fields:
        field_data = raw_fields[field]
        if dataset_name in [DatasetName.EMBODY3DSOLO]:
            if field_data.ndim == 0:
                continue
            raw_fields[field] = rearrange(field_data, "t ... -> t 1 ...")
        elif field_data.ndim >= 3:         
            raw_fields[field] = rearrange(field_data, "p t ... -> t p ...")
        else:
            pass

    # remap key names
    if dataset_name in [DatasetName.DUOBOX]:
        raw_fields["pose_body"] = raw_fields["body_pose"]
        raw_fields["root_orient"] = raw_fields["global_orient"]
        raw_fields["trans"] = raw_fields["transl"]
    elif dataset_name in [DatasetName.REMOCAP]:
        raw_fields["pose_body"] = raw_fields["body_pose"]
        raw_fields["root_orient"] = raw_fields["orient"]
    elif dataset_name in [DatasetName.INTERX]:
        raw_fields["pose_body"] = raw_fields["body_pose"]
    elif dataset_name in [DatasetName.EMBODY3DSOLO, DatasetName.EMBODY3DDUO, DatasetName.EMBODY3DTRIO, DatasetName.EMBODY3DQUAD]:
        raw_fields["pose_body"] = raw_fields["body_pose"]
        raw_fields["root_orient"] = raw_fields["global_orient"]
        raw_fields["trans"] = raw_fields["transl"]
        raw_fields["pose_lhand"] = raw_fields["left_hand_pose"]
        raw_fields["pose_rhand"] = raw_fields["right_hand_pose"]
    elif dataset_name in [DatasetName.DD100]:
        raw_fields["pose_body"] = raw_fields["poses"][..., 3:66]
        raw_fields["pose_hand"] = raw_fields["poses"][..., 75:165]
        raw_fields["trans"] = raw_fields["transl"]
        raw_fields["root_orient"] = raw_fields["global_orient"]
    else:
        pass

    T, P, _ = raw_fields["trans"].shape

    # remove data including tpose
    is_no_tpose = torch.all(raw_fields["pose_body"].abs().sum(dim=-1) > 0)
    if not is_no_tpose:
        print(f'!!!! {path.stem} includes tpose mask')
        return None
    # mocap dataset has no tpose
    tpose_mask = torch.ones((T, P), dtype=torch.bool, device=f'cuda:{device_idx}')

    #cut off betas dim
    raw_fields["betas"] = raw_fields["betas"][0, :, :10]
        

    # pose_body
    if dataset_name in [DatasetName.REMOCAP]:
        raw_fields["pose_body"] = raw_fields["pose_body"][:, :, :63].reshape(-1, P, 21, 3)
    elif dataset_name in [DatasetName.DD100]:
        raw_fields["pose_body"] = raw_fields["pose_body"][:, :, 3:66].reshape(-1, P, 21, 3)
    else:
        raw_fields["pose_body"] = raw_fields["pose_body"].reshape(-1, P, 21, 3)

    # pose_hand
    if dataset_name in [DatasetName.INTERX, DatasetName.EMBODY3DSOLO, DatasetName.EMBODY3DDUO, DatasetName.EMBODY3DTRIO, DatasetName.EMBODY3DQUAD]:
        raw_fields["pose_hand"] = torch.cat([
            raw_fields["pose_lhand"].reshape(T, P, 15, 3), 
            raw_fields["pose_rhand"].reshape(T, P, 15, 3)], dim=2)
    elif dataset_name in [DatasetName.DD100]:
        raw_fields["pose_hand"] = raw_fields["pose_hand"].reshape(T, P, 30, 3)
    else:
        raw_fields["pose_hand"] = torch.zeros((T, P, 30, 3), device=f'cuda:{device_idx}', dtype=torch.float32)
              

    # sanity check
    expected_shapes = {
        "root_orient": torch.Size([T, P, 3]),
        "trans": torch.Size([T, P, 3]),
        "pose_body": torch.Size([T, P, 21, 3]),
        "pose_hand": torch.Size([T, P, 30, 3]),
        "betas": torch.Size([P, 10]),
    }
    for field, expected_shape in expected_shapes.items():
        if raw_fields[field].shape != expected_shape:
            print(f"Invalid shape for {field}: expected {expected_shape}, got {raw_fields[field].shape}")
            return None
    

    # downsample to data fps
    if DATASET_FPS[dataset_name] is not None:
        org_fps = int(DATASET_FPS[dataset_name])
    else:
        org_fps = int(raw_fields["mocap_frame_rate"].item())

    if org_fps != data_fps:
        print(f"Downsampling from {org_fps} to {data_fps} fps")
        body_joint_rotations = interpolate_rotation(
            SO3.exp(raw_fields["pose_body"].to(f'cuda:{device_idx}')), org_fps, data_fps).wxyz
        hand_joint_rotations = interpolate_rotation(
            SO3.exp(raw_fields["pose_hand"].to(f'cuda:{device_idx}')), org_fps, data_fps).wxyz
        root_orientations = interpolate_rotation(SO3.exp(raw_fields["root_orient"].to(f'cuda:{device_idx}')), org_fps, data_fps).wxyz
        root_translations = interpolate_translation(raw_fields["trans"].to(f'cuda:{device_idx}'), org_fps, data_fps)

        T = root_translations.shape[0]
        raw_fields["betas"] = raw_fields["betas"][None, :, :].repeat(T, 1, 1)
        tpose_mask = torch.ones((T, P), dtype=torch.bool)

    else:
        body_joint_rotations = SO3.exp(raw_fields["pose_body"].to(f'cuda:{device_idx}')).wxyz
        hand_joint_rotations = SO3.exp(raw_fields["pose_hand"].to(f'cuda:{device_idx}')).wxyz
        root_orientations = SO3.exp(raw_fields["root_orient"].to(f'cuda:{device_idx}')).wxyz
        root_translations = raw_fields["trans"].to(f'cuda:{device_idx}')
        raw_fields["betas"] = raw_fields["betas"][None, :, :].repeat(T, 1, 1)

    T_world_root = torch.cat(
        [
            root_orientations,
            root_translations
        ],
        dim=-1,
    )
    T_world_root = rotate_global_up_rotation(dataset_name, T_world_root)
    
    floor_height, body_contacts = determine_floor_height_and_contacts(
        smpl_path=smpl_path,
        betas=raw_fields["betas"].to(f'cuda:{device_idx}'),
        T_world_root=T_world_root,
        body_joint_rotations=body_joint_rotations,
    )
    T_world_root, T_world_canonical = process_global_and_canonical_coord(
        T_world_root=T_world_root,
        floor_height=floor_height
    )

    # move world coordinate to the center of persons' first canonical frames
    new_world_translation = T_world_canonical[0, :, [4, 6]].mean(dim=0)
    T_world_root[:, :, [4, 6]] -= new_world_translation
    T_world_canonical[:, :, [4, 6]] -= new_world_translation

    # calculate self to partner canonical transform
    if P > 1:
        T_self_canonical_partner_canonical_ = (SE3(T_world_canonical[:, :, None, :]).inverse() @ SE3(T_world_canonical[:, None, :, :])).wxyz_xyz
        T_self_canonical_partner_canonical_list = []
        for i in range(P):
            rel_i = T_self_canonical_partner_canonical_[:, i, torch.arange(P) != i, :]
            T_self_canonical_partner_canonical_list.append(rel_i)
        T_self_canonical_partner_canonical = torch.stack(T_self_canonical_partner_canonical_list, dim=1)
    else:
        T_self_canonical_partner_canonical = torch.empty((T, 1, 0, 7), dtype=torch.float32)

    # calculate T_canonical_tm1_canonical_t
    T_canonical_tm1_canonical_t = torch.cat(
        [
            SE3.identity(device=T_world_canonical.device, dtype=T_world_canonical.dtype).wxyz_xyz[None, None, :].expand(1, P, 7),
            (SE3(T_world_canonical[:-1]).inverse() @ SE3(T_world_canonical[1:])).wxyz_xyz
        ],
        dim=0)

    # calculate T_canonical_root
    T_canonical_root = (SE3(T_world_canonical).inverse() @ SE3(T_world_root)).wxyz_xyz

    # expand contacts
    body_contacts = torch.cat([body_contacts[:1], body_contacts], dim=0)
    

    data = TrainingData(
        betas=raw_fields["betas"],
        body_joint_rotations=SO3(body_joint_rotations).as_6d().cpu(),
        hand_joint_rotations=SO3(hand_joint_rotations).as_6d().cpu(),
        T_canonical_root=SE3(T_canonical_root).as_9d().cpu(),
        T_canonical_tm1_canonical_t = SE3(T_canonical_tm1_canonical_t).as_9d().cpu(),
        T_world_root=SE3(T_world_root).as_9d().cpu(),
        T_world_canonical=SE3(T_world_canonical).as_9d().cpu(),
        T_self_canonical_partner_canonical=SE3(T_self_canonical_partner_canonical).as_9d().cpu(),
        body_contacts = body_contacts.cpu(),
        tpose_mask = tpose_mask.cpu(),
    )

    data_list = [data.pack()]


    if is_mirror_augment:
        m_data =  TrainingData(
            betas=raw_fields["betas"],
            body_joint_rotations=mirror.mirror_rotations_6d(data.body_joint_rotations, mode="body"),
            hand_joint_rotations=mirror.mirror_rotations_6d(data.hand_joint_rotations, mode="hand"),
            T_canonical_root=mirror.mirror_transforms_9d(data.T_canonical_root),
            T_canonical_tm1_canonical_t = mirror.mirror_transforms_9d(data.T_canonical_tm1_canonical_t),
            T_world_root=mirror.mirror_transforms_9d(data.T_world_root),
            T_world_canonical=mirror.mirror_transforms_9d(data.T_world_canonical),
            T_self_canonical_partner_canonical=mirror.mirror_transforms_9d(data.T_self_canonical_partner_canonical),
            body_contacts = body_contacts.cpu(),
            tpose_mask = tpose_mask.cpu(),
        )

        data_list.append(m_data.pack())

    return data_list
