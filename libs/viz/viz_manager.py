import uuid
from typing import List, Literal, Tuple

import numpy as np
import torch
import viser
from viser import transforms as tf

from color_gradation import SLAHMR_COLORS, make_oklab_shade_fn, rgb_to_oklab, oklab_to_rgb
from libs.utils import fncsmpl
from libs.utils.transforms import SO3


class VizTheme:
    # Ground Truth Colors
    GT_COLORS = [(128, 128, 128)] * 4

    # Prediction Shades
    PRED_STARTS = [
        (226, 87, 73),   # Red
        (44, 108, 161),  # Blue
        (46, 92, 73),    # Green
        (106, 58, 102),  # Purple
    ]
    PRED_SHADES = [
        make_oklab_shade_fn(start, limit=0.8, curve=0.8)
        for start in PRED_STARTS
    ]

    # Context Shades
    CONTEXT_STARTS = (255, 0, 255)  # Magenta
    CONTEXT_SHADES = make_oklab_shade_fn(CONTEXT_STARTS, limit=1.0, curve=1.0)

    # Dataset Colors
    DATASET_COLORS = PRED_STARTS
    DATASET_MIRROR_COLORS = [
        (242, 160, 150),
        (130, 180, 220),
        (120, 160, 145),
        (170, 130, 165),
    ]

    PERSON_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]

    @staticmethod
    def get_palette_pred(person_num: int, sample_num: int) -> List[List[Tuple[int, int, int]]]:
        """Generates colors for Prediction samples."""
        num_shades = len(VizTheme.PRED_SHADES)
        return [
            [
                VizTheme.PRED_SHADES[j % num_shades](0.1 * i)
                for j in range(person_num)
            ]
            for i in range(sample_num)
        ]

    @staticmethod
    def get_palette_gt(person_num: int) -> List[Tuple[int, int, int]]:
        """Returns gray colors for GT."""
        return VizTheme.GT_COLORS[:person_num]

    @staticmethod
    def get_palette_context(person_num: int) -> List[Tuple[int, int, int]]:
        """Generates magenta-based colors for context."""
        return [VizTheme.CONTEXT_SHADES(i / 8.0) for i in range(person_num)]

    @staticmethod
    def get_palette_dataset(person_num: int, is_mirror: bool) -> List[Tuple[int, int, int]]:
        if is_mirror:
            return VizTheme.DATASET_MIRROR_COLORS[:person_num]
        else:
            return VizTheme.DATASET_COLORS[:person_num]

    @staticmethod
    def get_person_name_gt(person_num: int) -> List[str]:
        return [f"GT/{VizTheme.PERSON_NAMES[i % len(VizTheme.PERSON_NAMES)]}" for i in range(person_num)]

    @staticmethod
    def get_person_name_pred(person_num: int, sample_idx_list: list[int]) -> List[List[str]]:
        names = VizTheme.PERSON_NAMES
        n_len = len(names)
        return [
            [f"Pred/Sample_{i}/{names[j % n_len]}" for j in range(person_num)]
            for i in sample_idx_list
        ]

    @staticmethod
    def get_person_name_dataset(dataset_name: str, person_num: int, is_mirror: bool) -> List[str]:
        if is_mirror:
            return [f"{dataset_name}_Mirror/{VizTheme.PERSON_NAMES[i % len(VizTheme.PERSON_NAMES)]}" for i in range(person_num)]
        else:
            return [f"{dataset_name}/{VizTheme.PERSON_NAMES[i % len(VizTheme.PERSON_NAMES)]}" for i in range(person_num)]

    @staticmethod
    def get_color_gradient(
        base_color: Tuple[int, int, int],
        num_steps: int,
        start_lightness: float = 1.0,
        end_lightness: float = 0.2,
    ) -> List[Tuple[int, int, int]]:
        """Generate a color gradient adjusting lightness in OKLAB space for perceptually uniform transitions.

        Args:
            base_color: Base RGB color tuple
            num_steps: Number of gradient steps
            start_lightness: Lightness factor for oldest position (0.0 = black, 1.0 = original, 2.0 = white)
            end_lightness: Lightness factor for newest position (0.0 = black, 1.0 = original, 2.0 = white)
        """
        L_base, a_base, b_base = rgb_to_oklab(*base_color)

        def factor_to_L(factor: float, L_base: float) -> float:
            if factor <= 1.0:
                return L_base * factor
            else:
                alpha = factor - 1.0
                return L_base + (1.0 - L_base) * alpha

        L_start = factor_to_L(start_lightness, L_base)
        L_end = factor_to_L(end_lightness, L_base)

        colors = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0
            L = L_start + (L_end - L_start) * t

            if L_base > 0:
                if L <= L_base:
                    chroma_scale = L / L_base if L_base > 0 else 0
                else:
                    chroma_scale = (1.0 - L) / (1.0 - L_base) if L_base < 1.0 else 0
                a = a_base * chroma_scale
                b = b_base * chroma_scale
            else:
                a = 0
                b = 0

            colors.append(oklab_to_rgb(L, a, b))

        return colors


class VizData:
    body_model: fncsmpl.SmplModel = fncsmpl.SmplModel.load(
        "./body_model/smplx/SMPLX_NEUTRAL.npz"
    ).to("cuda")

    def __init__(self) -> None:
        # --- SMPL data (GT) ---
        self.smpl_verts_zero_gt: np.ndarray | None = None
        self.smpl_joints_zero_gt: np.ndarray | None = None
        self.T_world_root_gt: np.ndarray | None = None
        self.Ts_world_joint_gt: np.ndarray | None = None

        # --- SMPL data (Pred) ---
        self.smpl_verts_zero_pred: np.ndarray | None = None
        self.smpl_joints_zero_pred: np.ndarray | None = None
        self.T_world_root_pred: np.ndarray | None = None
        self.Ts_world_joint_pred: np.ndarray | None = None

        # --- Mesh handles ---
        self.curr_body_handles_pred: List[List[viser.MeshSkinnedHandle]] = []  # [sample_idx][person_idx]
        self.prev_body_handles_pred: List[List[viser.MeshSkinnedHandle]] = []  # [sample_idx][person_idx]
        self.curr_body_handles_gt: List[viser.MeshSkinnedHandle] = []  # [person_idx]
        self.prev_body_handles_gt: List[viser.MeshSkinnedHandle] = []  # [person_idx]
        self.multi_timestep_handles: List[List[List[viser.MeshSkinnedHandle]]] = []  # [timestep_idx][sample_idx][person_idx]

        # --- Skin weights (lazily computed) ---
        self.skin_weights: np.ndarray | None = None

        # --- Timing / counts ---
        self.timesteps: int = 10
        self.context_timesteps: int = 0
        self.gt_timesteps: int = 0
        self.sample_num: int = 0
        self.person_num: int = 0

        # --- Visibility ---
        self.visible_sample_idx_list: List[int] = [0]

        # --- Colors ---
        self.colors_gt: List[Tuple[int, int, int]] = []
        self.colors_context: List[Tuple[int, int, int]] = []
        self.colors_pred: List[List[Tuple[int, int, int]]] = []

        # --- Names ---
        self.person_names_gt: List[str] = []
        self.person_names_pred: List[List[str]] = []

        # --- Mode ---
        self.hand_pose_type: Literal["Zero", "Fist"] = "Zero"
        self.mode: Literal["Playback", "Multitimestep"] = "Playback"
        self.visible_timesteps: int = 16
        self.selected_timesteps: List[int] = []
        self.task_mode: str = ""

    # ------------------------------------------------------------------
    # Skin-weight helpers
    # ------------------------------------------------------------------

    def _ensure_skin_weights(self) -> np.ndarray:
        if self.skin_weights is None:
            self.skin_weights = self._collapse_weights_to_top3_parent_pool(
                self.body_model.weights.numpy(force=True),
                list(self.body_model.parent_indices),
            )
        return self.skin_weights

    @staticmethod
    def _collapse_weights_to_top3_parent_pool(W, parents, eps=1e-8):
        parents = [0] + [i + 1 for i in parents]
        V, B = W.shape
        W_new = W.copy()
        for v in range(V):
            active = np.nonzero(W_new[v] > eps)[0].tolist()
            while len(active) > 3:
                j = min(active, key=lambda i: W_new[v, i])
                p = parents[j]
                if p == -1:
                    p = np.argmax(W_new[v])
                W_new[v, p] += W_new[v, j]
                W_new[v, j] = 0.0
                active.remove(j)
            s = W_new[v].sum()
            if s > 0:
                W_new[v] /= s
        return W_new

    # ------------------------------------------------------------------
    # Handle visibility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_handles_visible(handles: List[viser.MeshSkinnedHandle], visible: bool) -> None:
        for h in handles:
            h.visible = visible

    def remove_prev_data(self) -> None:
        for handle_list in self.prev_body_handles_pred:
            for h in handle_list:
                h.remove()
        self.prev_body_handles_pred = []
        for h in self.prev_body_handles_gt:
            h.remove()
        self.prev_body_handles_gt = []

    def invisible_all_data(self) -> None:
        self._set_handles_visible(self.prev_body_handles_gt, False)
        self._set_handles_visible(self.curr_body_handles_gt, False)
        for handle_list in self.curr_body_handles_pred:
            self._set_handles_visible(handle_list, False)
        for handle_list in self.prev_body_handles_pred:
            self._set_handles_visible(handle_list, False)

    def change_visible_pred_samples(self) -> None:
        for idx, body_handles in enumerate(self.curr_body_handles_pred):
            visible = idx in self.visible_sample_idx_list
            self._set_handles_visible(body_handles, visible)

    def visible_gt_data(self, visible: bool = True) -> None:
        self._set_handles_visible(self.curr_body_handles_gt, visible)

    def change_pred_color(self, is_context: bool = False) -> None:
        for idx in self.visible_sample_idx_list:
            if idx >= len(self.curr_body_handles_pred):
                continue
            for person_id, body_handle in enumerate(self.curr_body_handles_pred[idx]):
                if is_context:
                    body_handle.color = self.colors_context[person_id]
                else:
                    body_handle.color = self.colors_pred[idx][person_id]

    def change_gt_color(self, is_context: bool = False) -> None:
        for person_id, body_handle in enumerate(self.curr_body_handles_gt):
            if is_context:
                body_handle.color = self.colors_context[person_id]
            else:
                body_handle.color = self.colors_gt[person_id]

    # ------------------------------------------------------------------
    # Hand pose creation
    # ------------------------------------------------------------------

    def create_hand_poses(
        self,
        B: int,
        T: int,
        P: int,
        device: torch.device,
        pose_type: Literal["Zero", "Fist"] = "Fist",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Create left and right hand poses from predefined axis-angle rotations.

        Right hand is mirrored by negating the Z component of the axis-angle.

        Args:
            B: Batch size
            T: Timesteps
            P: Number of persons
            device: Device to create tensors on
            pose_type: Type of hand pose - "Zero" or "Fist"

        Returns:
            Tuple of (left_hand_quats, right_hand_quats), or (None, None) if pose_type is "Zero"
        """
        if pose_type == "Zero":
            return None, None

        # Joint angles in axis-angle format (SMPLX joints 25-39 -> hand indices 0-14)
        left_hand_axis_angles = torch.tensor(
            [
                [0.11, 0.17, -1.16],  # joint 25 -> hand idx 0
                [-0.00, -0.00, -0.97],  # joint 26 -> hand idx 1
                [0.00, 0.00, -1.12],  # joint 27 -> hand idx 2
                [-0.26, 0.05, -1.54],  # joint 28 -> hand idx 3
                [0.00, 0.00, 0.00],  # joint 29 -> hand idx 4
                [-0.13, -0.10, -1.77],  # joint 30 -> hand idx 5
                [-0.42, -0.56, -1.16],  # joint 31 -> hand idx 6
                [-0.20, -0.45, -0.58],  # joint 32 -> hand idx 7
                [-1.35, 0.04, -0.39],  # joint 33 -> hand idx 8
                [-0.20, -0.28, -1.34],  # joint 34 -> hand idx 9
                [-0.18, -0.00, -0.54],  # joint 35 -> hand idx 10
                [-0.50, -0.13, -1.25],  # joint 36 -> hand idx 11
                [1.23, 0.32, 0.33],  # joint 37 -> hand idx 12
                [0.08, -0.72, -0.49],  # joint 38 -> hand idx 13
                [-0.44, 1.29, -1.15],  # joint 39 -> hand idx 14
            ],
            device=device,
            dtype=torch.float32,
        )  # Shape: (15, 3)

        # Mirror for right hand by negating Y and Z components
        right_hand_axis_angles = left_hand_axis_angles.clone()
        right_hand_axis_angles[:, 1] *= -1  # Negate Y
        right_hand_axis_angles[:, 2] *= -1  # Negate Z

        # Convert all axis-angles to quaternions in one batch operation
        left_hand_quats_flat = SO3.exp(left_hand_axis_angles).wxyz  # Shape: (15, 4)
        right_hand_quats_flat = SO3.exp(right_hand_axis_angles).wxyz  # Shape: (15, 4)

        # Expand to (B, T, P, 15, 4)
        left_hand_quats = (
            left_hand_quats_flat.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, T, P, 15, 4)
        )
        right_hand_quats = (
            right_hand_quats_flat.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, T, P, 15, 4)
        )

        return left_hand_quats, right_hand_quats

    # ------------------------------------------------------------------
    # SMPL data update
    # ------------------------------------------------------------------

    def update_smpl_data(
        self,
        betas: torch.Tensor,
        T_world_root: torch.Tensor,
        body_joint_rotations: torch.Tensor,
        context_timesteps: int = 0,
        gt_timesteps: int = 0,
        max_sample_num: int = 10,
    ) -> None:
        print("updating smpl data")
        B, T, P = T_world_root.shape[:3]
        if B > max_sample_num + 1:
            betas = betas[:max_sample_num + 1]
            T_world_root = T_world_root[:max_sample_num + 1]
            body_joint_rotations = body_joint_rotations[:max_sample_num + 1]
            B = max_sample_num + 1

        smpl_shaped_all = self.body_model.with_shape(betas)
        self.smpl_verts_zero_gt = smpl_shaped_all.verts_zero[0, 0].numpy(force=True)
        self.smpl_verts_zero_pred = smpl_shaped_all.verts_zero[1:, 0].numpy(force=True)
        self.smpl_joints_zero_gt = smpl_shaped_all.joints_zero[0, 0].numpy(force=True)
        self.smpl_joints_zero_pred = smpl_shaped_all.joints_zero[1:, 0].numpy(force=True)

        # Create hand poses (right hand is mirrored version of left)
        left_hand_quats, right_hand_quats = self.create_hand_poses(
            B, T, P, T_world_root.device, self.hand_pose_type
        )

        smpl_posed_all = smpl_shaped_all.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_joint_rotations,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
        )

        self.T_world_root_gt = smpl_posed_all.T_world_root[0].numpy(force=True)
        self.Ts_world_joint_gt = smpl_posed_all.Ts_world_joint[0].numpy(force=True)
        self.Ts_world_joint_gt[..., 4:] -= self.T_world_root_gt[..., None, 4:]

        self.T_world_root_pred = smpl_posed_all.T_world_root[1:].numpy(force=True)
        self.Ts_world_joint_pred = smpl_posed_all.Ts_world_joint[1:].numpy(force=True)
        self.Ts_world_joint_pred[..., 4:] -= self.T_world_root_pred[..., None, 4:]

        self.timesteps = T
        self.context_timesteps = context_timesteps
        self.gt_timesteps = gt_timesteps
        self.sample_num = B - 1
        self.person_num = P

        self.person_names_gt = VizTheme.get_person_name_gt(self.person_num)
        self.colors_gt = VizTheme.get_palette_gt(self.person_num)

        self.person_names_pred = VizTheme.get_person_name_pred(
            self.person_num, list(range(self.sample_num))
        )
        self.colors_pred = VizTheme.get_palette_pred(self.person_num, self.sample_num)

        self.colors_context = VizTheme.get_palette_context(self.person_num)
        print("updated smpl data")

    # ------------------------------------------------------------------
    # Visibility / color updates
    # ------------------------------------------------------------------

    def update_visible_sample_idx_list(
        self,
        visible_sample_idx_list: List[int],
    ) -> None:
        print(f"update visible sample idx list: {visible_sample_idx_list}")
        self.visible_sample_idx_list = visible_sample_idx_list
        self.colors_pred = VizTheme.get_palette_pred(self.person_num, len(visible_sample_idx_list))
        if self.task_mode in ("inpainting", "motion_inpainting", "motion_prediction", "partner_prediction") and self.colors_context:
            for sample_colors in self.colors_pred:
                sample_colors[0] = self.colors_context[0]
        self.change_visible_pred_samples()

    def update_gt_color(self, colors: List[Tuple[int, int, int]]) -> None:
        self.colors_gt = colors

    def update_pred_color(self, colors: List[List[Tuple[int, int, int]]]) -> None:
        self.colors_pred = colors

    # ------------------------------------------------------------------
    # Scene handle creation
    # ------------------------------------------------------------------

    def add_body_handles_to_scene(
        self,
        scene: viser.SceneApi,
        show_gt: bool = False,
    ) -> None:
        self.invisible_all_data()

        # Move current handles to prev
        self.prev_body_handles_gt.extend(self.curr_body_handles_gt)
        self.prev_body_handles_pred.extend(self.curr_body_handles_pred)

        # Create GT handles
        self.curr_body_handles_gt = self._create_body_handles_gt(scene, show_gt)

        # Create Pred handles
        self.curr_body_handles_pred = [
            self._create_body_handles_pred(sample_idx, scene)
            for sample_idx in range(self.sample_num)
        ]

    def _create_body_handles_gt(
        self,
        scene: viser.SceneApi,
        show_gt: bool = False,
    ) -> List[viser.MeshSkinnedHandle]:
        skin_weights = self._ensure_skin_weights()
        handles = [
            scene.add_mesh_skinned(
                f"/persons/{self.person_names_gt[person_idx]}/{uuid.uuid4()}",
                vertices=self.smpl_verts_zero_gt[person_idx, :],
                faces=self.body_model.faces.numpy(force=True),
                bone_wxyzs=tf.SO3.identity(
                    batch_axes=(self.body_model.get_num_joints() + 1,)
                ).wxyz,
                bone_positions=np.concatenate(
                    [
                        np.zeros((1, 3)),
                        self.smpl_joints_zero_gt[person_idx, :],
                    ],
                    axis=0,
                ),
                skin_weights=skin_weights,
                visible=show_gt,
            )
            for person_idx in range(self.person_num)
        ]
        return handles

    def _create_body_handles_pred(
        self,
        sample_idx: int,
        scene: viser.SceneApi,
    ) -> List[viser.MeshSkinnedHandle]:
        skin_weights = self._ensure_skin_weights()
        is_visible = sample_idx in self.visible_sample_idx_list

        handles = [
            scene.add_mesh_skinned(
                f"/persons/{self.person_names_pred[sample_idx][person_idx]}/{uuid.uuid4()}",
                vertices=self.smpl_verts_zero_pred[sample_idx, person_idx, :],
                faces=self.body_model.faces.numpy(force=True),
                bone_wxyzs=tf.SO3.identity(
                    batch_axes=(self.body_model.get_num_joints() + 1,)
                ).wxyz,
                bone_positions=np.concatenate(
                    [
                        np.zeros((1, 3)),
                        self.smpl_joints_zero_pred[sample_idx, person_idx, :],
                    ],
                    axis=0,
                ),
                skin_weights=skin_weights,
                visible=is_visible,
            )
            for person_idx in range(self.person_num)
        ]
        return handles

    # ------------------------------------------------------------------
    # Multi-timestep handle management
    # ------------------------------------------------------------------

    def create_multitimestep_handles(
        self,
        num_timesteps: int,
        scene: viser.SceneApi,
    ) -> None:
        """Create body handles for multitimestep mode.

        Args:
            num_timesteps: Number of timestep slots to create.
            scene: The viser scene to add handles to.
        """
        print("creating multitimestep handles")
        # Clear old handles
        self.remove_multitimestep_handles()

        skin_weights = self._ensure_skin_weights()

        for timestep_idx in range(num_timesteps):
            timestep_handles: List[List[viser.MeshSkinnedHandle]] = []

            # GT slot
            gt_handles = [
                scene.add_mesh_skinned(
                    f"/multitimestep/{timestep_idx}/{self.person_names_gt[person_idx]}/{uuid.uuid4()}",
                    vertices=self.smpl_verts_zero_gt[person_idx, :],
                    faces=self.body_model.faces.numpy(force=True),
                    bone_wxyzs=tf.SO3.identity(
                        batch_axes=(self.body_model.get_num_joints() + 1,)
                    ).wxyz,
                    bone_positions=np.concatenate(
                        [
                            np.zeros((1, 3)),
                            self.smpl_joints_zero_gt[person_idx, :],
                        ],
                        axis=0,
                    ),
                    skin_weights=skin_weights,
                    visible=False,
                )
                for person_idx in range(self.person_num)
            ]
            timestep_handles.append(gt_handles)

            # Pred slots
            for sample_idx in range(self.sample_num):
                pred_handles = [
                    scene.add_mesh_skinned(
                        f"/multitimestep/{timestep_idx}/{self.person_names_pred[sample_idx][person_idx]}/{uuid.uuid4()}",
                        vertices=self.smpl_verts_zero_pred[sample_idx, person_idx, :],
                        faces=self.body_model.faces.numpy(force=True),
                        bone_wxyzs=tf.SO3.identity(
                            batch_axes=(self.body_model.get_num_joints() + 1,)
                        ).wxyz,
                        bone_positions=np.concatenate(
                            [
                                np.zeros((1, 3)),
                                self.smpl_joints_zero_pred[sample_idx, person_idx, :],
                            ],
                            axis=0,
                        ),
                        skin_weights=skin_weights,
                        visible=False,
                    )
                    for person_idx in range(self.person_num)
                ]
                timestep_handles.append(pred_handles)

            self.multi_timestep_handles.append(timestep_handles)
        print("created multitimestep handles")

    def remove_multitimestep_handles(self) -> None:
        for timestep_handles_list in self.multi_timestep_handles:
            for sample_handles in timestep_handles_list:
                for handle in sample_handles:
                    try:
                        handle.remove()
                    except Exception:
                        pass
        self.multi_timestep_handles.clear()

    def hide_multitimestep_handles(self) -> None:
        for timestep_handles_list in self.multi_timestep_handles:
            for sample_handles in timestep_handles_list:
                for handle in sample_handles:
                    handle.visible = False