import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import torch
import viser

from color_gradation import SLAHMR_COLORS
from libs.viz.viz_manager import VizData, VizTheme


def _pp2joint_partner_phase_end_exclusive(viz_data: VizData) -> int:
    """Last frame index excluded from partner coloring (first joint-only frame).

    Partner prediction follows GT only for frames where GT is actually used in the
    pipeline. We take the minimum of (1) the saved joint-phase boundary from
    inference and (2) ``gt_timesteps`` (length of GT aligned with the saved
    sequence: ``min(clip, pred_len)``), so colors never treat joint-generated
    frames as partner-phase when GT does not cover them.
    """
    if viz_data.task_mode != "pp2joint":
        return -1
    j = int(viz_data.joint_phase_start_timesteps)
    g = int(viz_data.gt_timesteps)
    if j > 0 and g > 0:
        return min(j, g)
    if j > 0:
        return j
    if g > 0:
        return g
    return -1


def _is_pp2joint_partner_phase_agent_a(t: int, viz_data: VizData) -> bool:
    """Agent A is GT-conditioned in partner prediction; joint phase is fully generated."""
    end = _pp2joint_partner_phase_end_exclusive(viz_data)
    if end < 0:
        return False
    ctx = int(viz_data.context_timesteps)
    return ctx <= t < end


def load_visualizer(server, data_root_dir):
    raw_data = None
    is_update_ok = False
    viz_data = VizData()

    def get_subdir_list():
        subdirs = []
        for i in data_root_dir.glob("*"):
            if i.is_dir():
                subdirs.append(str(i.relative_to(data_root_dir)))
        return ["None"] + sorted(subdirs)

    def get_file_list(dir=data_root_dir):
        return ["None"] + sorted(str(p.relative_to(dir)) for p in dir.glob("**/*.npz"))

    # add ground and lights
    server.scene.set_up_direction("+y")
    server.scene.add_grid(
        "/ground",
        plane="xz",
        width=100,
        height=100,
        cell_size=0.5,
        cell_thickness=0.5,
        section_size=1.0,
        section_thickness=0.7,
        position=(0.0, 0.0, 0.0),
        infinite_grid=False,
        fade_distance=30.0,
    )
    server.scene.add_light_hemisphere(
        name="light_hemisphere",
        sky_color=(255, 255, 255),
        ground_color=(200, 200, 200),
        intensity=1.0,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        visible=True,
    )

    # add timestep handles
    timestep_handles = [server.scene.add_frame("timesteps/0", show_axes=False)]

    # Create tabs for organization
    file_tab = server.gui.add_tab_group()

    # FILE TAB
    with file_tab.add_tab("File"):
        subdir_dropdown = server.gui.add_dropdown("Dir", options=get_subdir_list())
        file_dropdown = server.gui.add_dropdown("File", options=get_file_list())
        refresh_file_list = server.gui.add_button("Refresh File List")
        sample_num_dropdown = server.gui.add_dropdown(
            "Sample Num", initial_value=str(1), options=[str(i) for i in range(0, 11)]
        )

        sample_idx_folder = server.gui.add_folder("Sample Idx", visible=True)
        sample_check_buttons = []
        with sample_idx_folder:
            for sample_idx in range(10):
                if sample_idx == 0:
                    is_visible = True
                else:
                    is_visible = False
                sample_check_buttons.append(
                    server.gui.add_checkbox(
                        f"idx_{sample_idx + 1:02d}", is_visible, visible=is_visible
                    )
                )

    # DISPLAY TAB
    with file_tab.add_tab("Display"):
        mode_dropdown = server.gui.add_dropdown(
            "Mode",
            initial_value="Playback",
            options=["Playback", "Multitimestep"],
        )

        sample_color_mode_dropdown = server.gui.add_dropdown(
            "Sample color mode",
            initial_value="Uniform",
            options=(
                "Uniform",
                "Trajectory (rgb)",
                "Trajectory (idx)",
            ),
        )

        # Manual color selection folder
        idx_color_folder = server.gui.add_folder("Trajectory Colors", visible=False)
        idx_color_sliders = {}  # Will store sliders as {(sample_idx, person_idx): slider}

        # Trajectory color selection folder
        trajectory_colors_folder = server.gui.add_folder(
            "Trajectory Colors", visible=False
        )
        trajectory_rgb_pickers = {}  # {(sample_idx, person_idx): rgb_picker}

        # GT color selection folder
        gt_colors_folder = server.gui.add_folder("GT Colors", visible=True)
        gt_rgb_pickers = {}  # {person_idx: rgb_picker}

        @sample_color_mode_dropdown.on_update
        def _(_):
            idx_color_folder.visible = (
                sample_color_mode_dropdown.value == "Trajectory (idx)"
            )
            trajectory_colors_folder.visible = (
                sample_color_mode_dropdown.value == "Trajectory (rgb)"
            )

        show_gt_checkbox = server.gui.add_checkbox("Show GT", True)
        show_samples_checkbox = server.gui.add_checkbox("Show samples", True)
        hand_pose_dropdown = server.gui.add_dropdown(
            "Hand pose", initial_value="Zero", options=["Zero", "Fist"]
        )

        # Create meshes for all timesteps
        playback_folder = server.gui.add_folder("Playback", visible=True)
        with playback_folder:
            gui_timestep = server.gui.add_slider(
                "Timestep",
                min=0,
                max=viz_data.timesteps - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            gui_start_end = server.gui.add_multi_slider(
                "Start/end",
                min=0,
                max=viz_data.timesteps - 1,
                initial_value=(0, viz_data.timesteps - 1),
                step=1,
            )
            gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
            gui_playing = server.gui.add_checkbox("Playing", False)
            gui_framerate = server.gui.add_slider(
                "FPS", min=1, max=120, step=1, initial_value=30
            )
            gui_framerate_options = server.gui.add_button_group(
                "FPS options", ("15", "30", "60", "120")
            )
            gui_gt_shift_slider = server.gui.add_slider(
                "GT distance",
                min=0,
                max=8,
                step=0.5,
                initial_value=2.5,
            )
            gui_shift_slider = server.gui.add_slider(
                "Sample distance",
                min=0,
                max=8,
                step=0.5,
                initial_value=2.5,
            )
            gui_motion_trails_playback = server.gui.add_checkbox("Motion trails", False)

        # Multitimestep folder
        multitimestep_folder = server.gui.add_folder("Multitimestep", visible=False)
        multitimestep_sliders = []
        multitimestep_checkboxes = []
        timestep_sliders_folder = None  # Will be created dynamically
        with multitimestep_folder:
            gui_num_visible_timesteps = server.gui.add_number(
                "# visible timesteps",
                initial_value=16,
                min=1,
                max=100,
                step=1,
            )
            gui_gt_shift_vector = server.gui.add_vector3(
                "GT offset",
                initial_value=(2.0, 0.0, 0.0),
                step=0.1,
            )
            gui_shift_vector = server.gui.add_vector3(
                "Sample offset",
                initial_value=(0.0, 0.0, 0.0),
                step=0.1,
            )
            gui_motion_trails_multi = server.gui.add_checkbox("Motion trails", False)

            # Timestep offset control
            gui_timestep_offset_vector = server.gui.add_vector3(
                "Offset per timestep",
                initial_value=(0.0, 0.0, 0.0),
                step=0.001,
            )

            # Lightness controls for gradient
            lightness_folder = server.gui.add_folder("Lightness / fade with timestep")
            with lightness_folder:
                gui_trail_start_brightness = server.gui.add_number(
                    "Trail start (oldest)",
                    initial_value=1.3,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )
                gui_trail_end_brightness = server.gui.add_number(
                    "Trail end (newest)",
                    initial_value=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )
                gui_mesh_start_brightness = server.gui.add_number(
                    "Mesh start (oldest)",
                    initial_value=1.3,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )
                gui_mesh_end_brightness = server.gui.add_number(
                    "Mesh end (newest)",
                    initial_value=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )

            # timestep_sliders_folder will be created dynamically in create_timestep_sliders()

    # ------------------------------------------------------------------
    # Helper: create multitimestep sliders
    # ------------------------------------------------------------------

    def create_timestep_sliders():
        """Create or recreate timestep sliders based on # visible timesteps."""
        nonlocal multitimestep_sliders
        nonlocal multitimestep_checkboxes
        nonlocal timestep_sliders_folder

        # Remove old folder entirely (this removes all children automatically)
        if timestep_sliders_folder is not None:
            try:
                timestep_sliders_folder.remove()
            except Exception:
                pass

        multitimestep_sliders.clear()
        multitimestep_checkboxes.clear()

        # Create new sliders and checkboxes
        num_sliders = int(gui_num_visible_timesteps.value)
        viz_data.visible_timesteps = num_sliders

        # Recreate the folder inside multitimestep_folder
        with multitimestep_folder:
            timestep_sliders_folder = server.gui.add_folder("Timestep Sliders")

        with timestep_sliders_folder:
            for i in range(num_sliders):
                # Calculate uniformly spaced initial value
                if num_sliders == 1:
                    initial_value = 0
                else:
                    initial_value = int(
                        i * (viz_data.timesteps - 1) / (num_sliders - 1)
                    )

                # Create checkbox
                checkbox = server.gui.add_checkbox(f"Show {i + 1}", True)
                multitimestep_checkboxes.append(checkbox)

                # Create slider
                slider = server.gui.add_slider(
                    f"Timestep {i + 1}",
                    min=0,
                    max=viz_data.timesteps - 1,
                    step=1,
                    initial_value=initial_value,
                )
                multitimestep_sliders.append(slider)

        # Update selected timesteps
        viz_data.selected_timesteps = [slider.value for slider in multitimestep_sliders]

    # ------------------------------------------------------------------
    # Helper: update color controls
    # ------------------------------------------------------------------

    def update_color_controls():
        """Update color controls based on current visible_sample_idx_list."""
        person_names = VizTheme.PERSON_NAMES

        # Update manual color sliders
        for slider in idx_color_sliders.values():
            slider.remove()
        idx_color_sliders.clear()

        with idx_color_folder:
            for sample_idx in viz_data.visible_sample_idx_list:
                for person_idx in range(viz_data.person_num):
                    slider = server.gui.add_slider(
                        f"Sample {sample_idx + 1} - {person_names[person_idx % len(person_names)]}",
                        min=0,
                        max=len(SLAHMR_COLORS) - 1,
                        step=1,
                        initial_value=(sample_idx * 7 + person_idx * 13)
                        % len(SLAHMR_COLORS),
                    )
                    idx_color_sliders[(sample_idx, person_idx)] = slider

        # Update trajectory color controls (one per sample per person)
        for picker in trajectory_rgb_pickers.values():
            picker.remove()
        trajectory_rgb_pickers.clear()

        with trajectory_colors_folder:
            for list_pos, sample_idx in enumerate(viz_data.visible_sample_idx_list):
                for person_idx in range(viz_data.person_num):
                    # Use colors from viz_data.colors_pred
                    color_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                    if color_idx >= 0 and color_idx < len(viz_data.colors_pred):
                        initial_trajectory_color = viz_data.colors_pred[color_idx][person_idx]
                    else:
                        initial_trajectory_color = (200, 200, 200)

                    rgb_picker = server.gui.add_rgb(
                        f"Sample {sample_idx + 1} - {person_names[person_idx % len(person_names)]}",
                        initial_value=initial_trajectory_color,
                    )

                    trajectory_rgb_pickers[(sample_idx, person_idx)] = rgb_picker

    # ------------------------------------------------------------------
    # Helper: load data
    # ------------------------------------------------------------------

    def load_data():
        nonlocal raw_data
        if raw_data is None:
            return

        nonlocal viz_data

        max_sample = 10
        if "body_joint_rotations" in raw_data:
            body_joint_rotations = (
                torch.from_numpy(raw_data["body_joint_rotations"][: max_sample + 1])
                .float()
                .to("cuda")
            )
        else:
            body_joint_rotations = (
                torch.from_numpy(raw_data["joint_rotations"][: max_sample + 1])
                .float()
                .to("cuda")
            )

        T_world_root = (
            torch.from_numpy(raw_data["T_world_root"][: max_sample + 1])
            .float()
            .to("cuda")
        )
        betas = torch.from_numpy(raw_data["betas"][: max_sample + 1]).float().to("cuda")
        context_timesteps = int(np.asarray(raw_data["context_timesteps"]).squeeze())
        if "joint_phase_start_timesteps" in raw_data:
            joint_phase_start_timesteps = int(np.asarray(raw_data["joint_phase_start_timesteps"]).squeeze())
        else:
            joint_phase_start_timesteps = -1
        if "gt_timesteps" in raw_data:
            gt_timesteps = int(np.asarray(raw_data["gt_timesteps"]).squeeze())
        else:
            gt_timesteps = 0

        # Read task mode from npz if available, otherwise try inf_cfg.yaml
        if "mode" in raw_data:
            viz_data.task_mode = str(raw_data["mode"])
        else:
            viz_data.task_mode = ""
            # Fallback: read from inf_cfg.yaml in the same directory
            if subdir_dropdown.value != "None":
                inf_cfg_path = data_root_dir / subdir_dropdown.value / "inf_cfg.yaml"
            else:
                inf_cfg_path = data_root_dir / "inf_cfg.yaml"
            if inf_cfg_path.exists():
                try:
                    with open(inf_cfg_path, "r") as f:
                        for line in f:
                            stripped = line.strip()
                            if stripped.startswith("task_mode:"):
                                viz_data.task_mode = stripped.split(":", 1)[1].strip()
                                break
                except Exception:
                    pass
        print(f"task mode: {viz_data.task_mode}")

        viz_data.update_smpl_data(
            betas,
            T_world_root,
            body_joint_rotations,
            context_timesteps,
            gt_timesteps,
            joint_phase_start_timesteps=joint_phase_start_timesteps,
        )

        pe = _pp2joint_partner_phase_end_exclusive(viz_data)
        if pe >= 0 and viz_data.task_mode == "pp2joint":
            print(
                f"pp2joint coloring: context [0..{viz_data.context_timesteps}], "
                f"partner Agent A [{viz_data.context_timesteps}..{pe}), "
                f"joint from {pe} "
                f"(joint_boundary={viz_data.joint_phase_start_timesteps}, gt_timesteps={viz_data.gt_timesteps})"
            )

        # For inpainting / motion_prediction, set first person's color to context color
        if viz_data.task_mode in ("inpainting", "motion_inpainting", "motion_prediction", "partner_prediction"):
            viz_data.colors_gt[0] = viz_data.colors_context[0]
            for sample_colors in viz_data.colors_pred:
                sample_colors[0] = viz_data.colors_context[0]

        viz_data.add_body_handles_to_scene(server.scene, show_gt=show_gt_checkbox.value)

        for sample_idx, sample_check_button in enumerate(sample_check_buttons):
            if sample_idx < viz_data.sample_num:
                sample_check_button.visible = True
            else:
                break

        # Create GT color controls
        for picker in gt_rgb_pickers.values():
            picker.remove()
        gt_rgb_pickers.clear()

        with gt_colors_folder:
            for person_idx in range(viz_data.person_num):
                initial_gt_color = viz_data.colors_context[person_idx]

                rgb_picker = server.gui.add_rgb(
                    f"{VizTheme.PERSON_NAMES[person_idx % len(VizTheme.PERSON_NAMES)]} GT",
                    initial_value=initial_gt_color,
                )

                def make_gt_callback(pidx, picker):
                    @picker.on_update
                    def _(_):
                        new_colors = list(viz_data.colors_gt)
                        new_colors[pidx] = picker.value
                        viz_data.update_gt_color(new_colors)

                    return _

                make_gt_callback(person_idx, rgb_picker)
                gt_rgb_pickers[person_idx] = rgb_picker

        # Update color controls for visible samples
        update_color_controls()

        gui_timestep.max = viz_data.timesteps - 1
        gui_start_end.max = viz_data.timesteps - 1
        gui_start_end.value = (0, viz_data.timesteps - 1)

        # Initialize multitimestep sliders and handles
        create_timestep_sliders()

    # ------------------------------------------------------------------
    # Body update at timestep (Playback mode)
    # ------------------------------------------------------------------

    # Track trail handles and previous state to avoid unnecessary updates
    trail_handles = {}
    prev_multitimestep_state = None

    def update_body_at_timestep_gt(person_idx, t, offset):
        """Update a GT body handle to show pose at timestep t."""
        if not viz_data.curr_body_handles_gt:
            return
        body_handle = viz_data.curr_body_handles_gt[person_idx]

        if not body_handle.visible:
            return
        if viz_data.timesteps <= t:
            return

        zero_pos = np.zeros(3, dtype=np.float32)
        T_world_root = viz_data.T_world_root_gt[t, person_idx]
        body_handle.bones[0].position = zero_pos
        body_handle.bones[0].wxyz = T_world_root[:4]
        Ts_world_joint = viz_data.Ts_world_joint_gt[t, person_idx]
        for b, bone_handle in enumerate(body_handle.bones[1:]):
            bone_transform = Ts_world_joint[b]
            bone_handle.position = bone_transform[4:7]
            bone_handle.wxyz = bone_transform[:4]
        body_handle.position = offset + T_world_root[4:7]

        # GT color: use context color during context period, else GT gray
        if t <= viz_data.context_timesteps:
            if person_idx in gt_rgb_pickers:
                body_handle.color = gt_rgb_pickers[person_idx].value
            else:
                body_handle.color = viz_data.colors_gt[person_idx]
        else:
            body_handle.color = viz_data.colors_gt[person_idx]

    def update_body_at_timestep_pred(sample_idx, list_pos, person_idx, t, offset):
        """Update a Pred body handle to show pose at timestep t."""
        if sample_idx >= len(viz_data.curr_body_handles_pred):
            return
        body_handle = viz_data.curr_body_handles_pred[sample_idx][person_idx]

        if not body_handle.visible:
            return
        if viz_data.timesteps <= t:
            return

        zero_pos = np.zeros(3, dtype=np.float32)
        T_world_root = viz_data.T_world_root_pred[sample_idx, t, person_idx]
        body_handle.bones[0].position = zero_pos
        body_handle.bones[0].wxyz = T_world_root[:4]
        Ts_world_joint = viz_data.Ts_world_joint_pred[sample_idx, t, person_idx]
        for b, bone_handle in enumerate(body_handle.bones[1:]):
            bone_transform = Ts_world_joint[b]
            bone_handle.position = bone_transform[4:7]
            bone_handle.wxyz = bone_transform[:4]
        body_handle.position = offset + T_world_root[4:7]

        # Color logic
        if t < viz_data.context_timesteps:
            # Samples in context period: use GT color from RGB picker
            if person_idx in gt_rgb_pickers:
                body_handle.color = gt_rgb_pickers[person_idx].value
            else:
                body_handle.color = viz_data.colors_gt[person_idx]
        elif _is_pp2joint_partner_phase_agent_a(t, viz_data) and person_idx == 0:
            # pp2joint partner phase: Agent A follows GT (show as context-conditioned)
            if 0 in gt_rgb_pickers:
                body_handle.color = gt_rgb_pickers[0].value
            else:
                body_handle.color = viz_data.colors_context[0]
        else:
            color_mode = sample_color_mode_dropdown.value
            if color_mode == "Uniform":
                c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                body_handle.color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
            elif color_mode == "Trajectory (idx)":
                slider_key = (sample_idx, person_idx)
                if slider_key in idx_color_sliders:
                    color_idx = int(idx_color_sliders[slider_key].value)
                    body_handle.color = SLAHMR_COLORS[color_idx]
                else:
                    c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                    body_handle.color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
            elif color_mode == "Trajectory (rgb)":
                s_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                body_handle.color = viz_data.colors_pred[s_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
            else:
                assert False

    # ------------------------------------------------------------------
    # Motion trails
    # ------------------------------------------------------------------

    def _get_T_world_root(sample_idx, t, person_idx, is_gt=False):
        """Get T_world_root for either GT or Pred."""
        if is_gt:
            return viz_data.T_world_root_gt[t, person_idx]
        else:
            return viz_data.T_world_root_pred[sample_idx, t, person_idx]

    def draw_motion_trails(
        start_t, end_t, gt_shift, sample_shift, active_timesteps=None
    ):
        """Draw motion trails from start_t to end_t.

        Args:
            start_t: Starting timestep
            end_t: Ending timestep
            gt_shift: GT offset (scalar for Playback, vector for Multitimestep)
            sample_shift: Sample offset (scalar for Playback, vector for Multitimestep)
            active_timesteps: List of visible timestep values for interpolating frame indices (Multitimestep only)
        """
        nonlocal trail_handles

        # Remove old trails
        for handle in trail_handles.values():
            try:
                handle.remove()
            except Exception:
                pass
        trail_handles.clear()

        # Build the list of (is_gt, sample_idx, list_pos) to iterate
        entries = []
        if show_gt_checkbox.value:
            entries.append((True, 0, 0))  # GT entry
        if show_samples_checkbox.value:
            for list_pos, sample_idx in enumerate(viz_data.visible_sample_idx_list):
                entries.append((False, sample_idx, list_pos))

        for is_gt, sample_idx, list_pos in entries:
            if is_gt:
                offset = np.array([0.0, 0.0, 0.0])
            elif list_pos == 0:
                offset = (
                    np.array([gt_shift, 0.0, 0.0])
                    if np.isscalar(gt_shift)
                    else np.array(gt_shift)
                )
            else:
                if np.isscalar(gt_shift):
                    offset = np.array(
                        [gt_shift + sample_shift * float(list_pos), 0.0, 0.0]
                    )
                else:
                    offset = np.array(gt_shift) + np.array(sample_shift) * float(list_pos)

            for person_idx in range(viz_data.person_num):
                # Get base color for this person based on color mode
                color_mode = sample_color_mode_dropdown.value

                if is_gt:
                    if person_idx in gt_rgb_pickers:
                        base_color = gt_rgb_pickers[person_idx].value
                    else:
                        base_color = viz_data.colors_gt[person_idx]
                elif color_mode == "Uniform":
                    c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                    base_color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
                elif color_mode == "Trajectory (idx)":
                    slider_key = (sample_idx, person_idx)
                    if slider_key in idx_color_sliders:
                        color_idx = int(idx_color_sliders[slider_key].value)
                        base_color = SLAHMR_COLORS[color_idx]
                    else:
                        c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                        base_color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
                elif color_mode == "Trajectory (rgb)":
                    trajectory_key = (sample_idx, person_idx)
                    if trajectory_key in trajectory_rgb_pickers:
                        base_color = trajectory_rgb_pickers[trajectory_key].value
                    else:
                        c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                        base_color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
                else:
                    assert False

                # Create positions for trail
                num_points = end_t - start_t + 1
                if num_points < 2:
                    continue

                points = []
                for t in range(start_t, end_t + 1):
                    if t < viz_data.timesteps:
                        T_world_root = _get_T_world_root(sample_idx, t, person_idx, is_gt)

                        # Calculate frame index for timestep offset
                        if active_timesteps is not None and len(active_timesteps) > 1:
                            frame_idx = 0.0
                            for i in range(len(active_timesteps) - 1):
                                if active_timesteps[i] <= t <= active_timesteps[i + 1]:
                                    t_range = (
                                        active_timesteps[i + 1] - active_timesteps[i]
                                    )
                                    if t_range > 0:
                                        alpha = (t - active_timesteps[i]) / t_range
                                        frame_idx = i + alpha
                                    else:
                                        frame_idx = i
                                    break
                            else:
                                if t >= active_timesteps[-1]:
                                    frame_idx = len(active_timesteps) - 1
                            timestep_offset = (
                                np.array(gui_timestep_offset_vector.value) * frame_idx
                            )
                        else:
                            timestep_offset = np.array([0.0, 0.0, 0.0])

                        pos = offset + T_world_root[4:7] + timestep_offset
                        points.append(pos)

                if len(points) < 2:
                    continue

                # Apply Gaussian smoothing to trajectory
                points_array = np.array(points)
                if points_array.shape[0] >= 3:
                    points_smoothed = np.zeros_like(points_array)
                    for i in range(3):  # x, y, z
                        points_smoothed[:, i] = gaussian_filter1d(
                            points_array[:, i], sigma=2.0, mode="nearest"
                        )
                else:
                    points_smoothed = points_array

                # Create line segments
                starts = points_smoothed[:-1]
                ends = points_smoothed[1:]

                # Generate color gradient
                trail_start_bright = gui_trail_start_brightness.value
                trail_end_bright = gui_trail_end_brightness.value
                colors_list = VizTheme.get_color_gradient(
                    base_color, len(starts), trail_start_bright, trail_end_bright
                )

                colors = np.array(colors_list)  # Shape: (N, 3)
                colors_per_vertex = np.repeat(
                    colors[:, np.newaxis, :], 2, axis=1
                )  # Shape: (N, 2, 3)

                # Add line segments
                trail_key = f"trail_{'gt' if is_gt else sample_idx}_{person_idx}"
                trail_handles[trail_key] = server.scene.add_line_segments(
                    f"/trails/{trail_key}",
                    points=np.concatenate([starts, ends], axis=1).reshape(-1, 2, 3),
                    colors=colors_per_vertex,
                    line_width=3.0,
                )

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def do_update():
        if not viz_data.curr_body_handles_pred and not viz_data.curr_body_handles_gt:
            return
        if not is_update_ok:
            return

        # Get shift values based on mode
        if viz_data.mode == "Playback":
            gt_shift = gui_gt_shift_slider.value
            sample_shift = gui_shift_slider.value
            motion_trails_enabled = gui_motion_trails_playback.value
        else:  # Multitimestep
            gt_shift = np.array(gui_gt_shift_vector.value)
            sample_shift = np.array(gui_shift_vector.value)
            motion_trails_enabled = gui_motion_trails_multi.value

        if viz_data.mode == "Playback":
            # Playback mode: show single timestep
            t = gui_timestep.value

            # Update GT
            if show_gt_checkbox.value:
                viz_data.visible_gt_data(True)
                gt_offset = np.array([0.0, 0.0, 0.0])
                for person_idx in range(viz_data.person_num):
                    update_body_at_timestep_gt(person_idx, t, gt_offset)
            else:
                viz_data.visible_gt_data(False)

            # Update Pred samples
            if show_samples_checkbox.value:
                for list_pos, sample_idx in enumerate(viz_data.visible_sample_idx_list):
                    if list_pos == 0:
                        offset = np.array([gt_shift, 0.0, 0.0])
                    else:
                        offset = np.array(
                            [gt_shift + sample_shift * float(list_pos), 0.0, 0.0]
                        )

                    for person_idx in range(viz_data.person_num):
                        update_body_at_timestep_pred(
                            sample_idx, list_pos, person_idx, t, offset
                        )
            else:
                # Hide all pred sample bodies
                for sample_idx in range(len(viz_data.curr_body_handles_pred)):
                    VizData._set_handles_visible(viz_data.curr_body_handles_pred[sample_idx], False)

            # Update timestep frames
            for ii, timestep_frame in enumerate(timestep_handles):
                timestep_frame.visible = t == ii

            # Draw motion trails from 0 to current timestep
            if motion_trails_enabled:
                draw_motion_trails(0, t, gt_shift, sample_shift, active_timesteps=None)
            else:
                for handle in trail_handles.values():
                    try:
                        handle.remove()
                    except Exception:
                        pass
                trail_handles.clear()

        else:  # Multitimestep mode
            nonlocal prev_multitimestep_state

            # Skip if no sliders or no data
            if not multitimestep_sliders or viz_data.skin_weights is None:
                return

            # Lazily create multitimestep handles if needed
            if len(viz_data.multi_timestep_handles) != len(multitimestep_sliders):
                viz_data.create_multitimestep_handles(len(multitimestep_sliders), server.scene)

            # Get current state
            current_selected_timesteps = tuple(
                slider.value for slider in multitimestep_sliders
            )
            current_checkbox_states = tuple(
                checkbox.value for checkbox in multitimestep_checkboxes
            )
            num_visible = len(multitimestep_sliders)

            # Include manual color slider values if in Manual mode
            if sample_color_mode_dropdown.value == "Trajectory (idx)":
                idx_colors = tuple(
                    slider.value for slider in idx_color_sliders.values()
                )
            else:
                idx_colors = ()

            # Include GT color values
            gt_colors = tuple(picker.value for picker in gt_rgb_pickers.values())

            # Include trajectory color values if in Trajectory mode
            if sample_color_mode_dropdown.value == "Trajectory (rgb)":
                trajectory_colors = tuple(
                    picker.value for picker in trajectory_rgb_pickers.values()
                )
            else:
                trajectory_colors = ()

            current_state = (
                current_selected_timesteps,
                current_checkbox_states,
                tuple(gt_shift),
                tuple(sample_shift),
                motion_trails_enabled,
                show_gt_checkbox.value,
                show_samples_checkbox.value,
                gui_trail_start_brightness.value,
                gui_trail_end_brightness.value,
                gui_mesh_start_brightness.value,
                gui_mesh_end_brightness.value,
                sample_color_mode_dropdown.value,
                num_visible,
                idx_colors,
                gt_colors,
                trajectory_colors,
                gui_timestep_offset_vector.value,
            )

            # Only check state if we have valid handles with matching count
            if len(viz_data.multi_timestep_handles) == len(multitimestep_sliders):
                if prev_multitimestep_state == current_state:
                    return
            else:
                return

            # Update state
            prev_multitimestep_state = current_state

            # Hide all regular body handles
            viz_data.invisible_all_data()

            # Hide all multitimestep handles first
            viz_data.hide_multitimestep_handles()

            # Update selected timesteps from sliders
            viz_data.selected_timesteps = list(current_selected_timesteps)

            # Show and update bodies at each selected timestep
            zero_pos = np.zeros(3, dtype=np.float32)
            active_timesteps = []

            num_visible = min(
                num_visible,
                len(viz_data.selected_timesteps),
                len(multitimestep_checkboxes),
            )

            for timestep_idx in range(num_visible):
                if timestep_idx >= len(viz_data.multi_timestep_handles):
                    continue
                if timestep_idx >= len(viz_data.selected_timesteps):
                    continue
                if timestep_idx >= len(multitimestep_checkboxes):
                    continue

                t = viz_data.selected_timesteps[timestep_idx]

                if not multitimestep_checkboxes[timestep_idx].value:
                    continue

                active_timesteps.append(t)

                # In multi_timestep_handles: [timestep_idx][0=GT, 1+=pred_sample][person_idx]
                # Build entries: (is_gt, handle_slot_idx, sample_idx_for_data, list_pos)
                entries = []
                if show_gt_checkbox.value:
                    entries.append((True, 0, 0, 0))
                if show_samples_checkbox.value:
                    for list_pos, sample_idx in enumerate(viz_data.visible_sample_idx_list):
                        # handle slot for pred is sample_idx + 1 (0 is GT)
                        handle_slot = sample_idx + 1
                        entries.append((False, handle_slot, sample_idx, list_pos))

                for entry_num, (is_gt, handle_slot, sample_idx, list_pos) in enumerate(entries):
                    if handle_slot >= len(viz_data.multi_timestep_handles[timestep_idx]):
                        continue

                    if is_gt:
                        offset = np.array([0.0, 0.0, 0.0])
                    elif list_pos == 0:
                        offset = np.array(gt_shift)
                    else:
                        offset = np.array(gt_shift) + np.array(sample_shift) * float(list_pos)

                    for person_idx in range(viz_data.person_num):
                        if person_idx >= len(viz_data.multi_timestep_handles[timestep_idx][handle_slot]):
                            continue

                        body_handle = viz_data.multi_timestep_handles[timestep_idx][handle_slot][person_idx]
                        body_handle.visible = True

                        if viz_data.timesteps <= t:
                            continue

                        # Update pose
                        if is_gt:
                            T_world_root = viz_data.T_world_root_gt[t, person_idx]
                            Ts_world_joint = viz_data.Ts_world_joint_gt[t, person_idx]
                        else:
                            T_world_root = viz_data.T_world_root_pred[sample_idx, t, person_idx]
                            Ts_world_joint = viz_data.Ts_world_joint_pred[sample_idx, t, person_idx]

                        body_handle.bones[0].position = zero_pos
                        body_handle.bones[0].wxyz = T_world_root[:4]
                        for b, bone_handle in enumerate(body_handle.bones[1:]):
                            bone_transform = Ts_world_joint[b]
                            bone_handle.position = bone_transform[4:7]
                            bone_handle.wxyz = bone_transform[:4]

                        # Calculate position with timestep offset
                        timestep_offset = (
                            np.array(gui_timestep_offset_vector.value) * timestep_idx
                        )
                        body_handle.position = (
                            offset + T_world_root[4:7] + timestep_offset
                        )

                        # Color logic
                        color_mode = sample_color_mode_dropdown.value
                        apply_lightness = True

                        if is_gt:
                            if person_idx in gt_rgb_pickers:
                                base_color = gt_rgb_pickers[person_idx].value
                            else:
                                base_color = viz_data.colors_gt[person_idx]
                            apply_lightness = False
                        elif t < viz_data.context_timesteps:
                            if person_idx in gt_rgb_pickers:
                                base_color = gt_rgb_pickers[person_idx].value
                            else:
                                base_color = viz_data.colors_gt[person_idx]
                            apply_lightness = False
                        elif _is_pp2joint_partner_phase_agent_a(t, viz_data) and person_idx == 0:
                            if 0 in gt_rgb_pickers:
                                base_color = gt_rgb_pickers[0].value
                            else:
                                base_color = viz_data.colors_context[0]
                            apply_lightness = False
                        elif color_mode == "Uniform":
                            c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                            base_color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
                        elif color_mode == "Trajectory (idx)":
                            slider_key = (sample_idx, person_idx)
                            if slider_key in idx_color_sliders:
                                color_idx = int(idx_color_sliders[slider_key].value)
                                base_color = SLAHMR_COLORS[color_idx]
                            else:
                                c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                                base_color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
                        elif color_mode == "Trajectory (rgb)":
                            trajectory_key = (sample_idx, person_idx)
                            if trajectory_key in trajectory_rgb_pickers:
                                base_color = trajectory_rgb_pickers[trajectory_key].value
                            else:
                                c_idx = min(list_pos, len(viz_data.colors_pred) - 1)
                                base_color = viz_data.colors_pred[c_idx][person_idx] if viz_data.colors_pred else (200, 200, 200)
                        else:
                            assert False

                        # Apply lightness gradient based on actual timestep value
                        if not apply_lightness:
                            body_handle.color = base_color
                        else:
                            mesh_start_bright = gui_mesh_start_brightness.value
                            mesh_end_bright = gui_mesh_end_brightness.value

                            if viz_data.timesteps > 1:
                                timestep_progress = t / (viz_data.timesteps - 1)
                            else:
                                timestep_progress = 0

                            lightness = (
                                mesh_start_bright
                                + (mesh_end_bright - mesh_start_bright)
                                * timestep_progress
                            )
                            gradient_colors = VizTheme.get_color_gradient(
                                base_color, 1, lightness, lightness
                            )
                            body_handle.color = gradient_colors[0]

            # Hide timestep frames
            for timestep_frame in timestep_handles:
                timestep_frame.visible = False

            # Draw motion trails between first and last active timesteps
            if motion_trails_enabled and len(active_timesteps) >= 2:
                start_t = min(active_timesteps)
                end_t = max(active_timesteps)
                draw_motion_trails(
                    start_t, end_t, gt_shift, sample_shift, active_timesteps
                )
            else:
                for handle in trail_handles.values():
                    try:
                        handle.remove()
                    except Exception:
                        pass
                trail_handles.clear()

    # ------------------------------------------------------------------
    # GUI callbacks
    # ------------------------------------------------------------------

    get_viser_file = server.gui.add_button("Get .viser file")
    remove_old_bodies = server.gui.add_button("Remove old bodies")
    prev_time = time.time()
    handle = None

    @gui_next_frame.on_click
    def _(_):
        max_timestep = gui_timestep.max + 1
        gui_timestep.value = (gui_timestep.value + 1) % max_timestep

    @gui_prev_frame.on_click
    def _(_):
        max_timestep = gui_timestep.max + 1
        gui_timestep.value = (gui_timestep.value - 1) % max_timestep

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    @gui_framerate_options.on_click
    def _(_):
        gui_framerate.value = int(gui_framerate_options.value)

    @refresh_file_list.on_click
    def _(_) -> None:
        subdir_dropdown.options = get_subdir_list()
        file_dropdown.options = get_file_list()

    @remove_old_bodies.on_click
    def _(_):
        remove_old_bodies.disabled = True
        gui_playing.value = False
        gui_timestep.disabled = False
        gui_next_frame.disabled = False
        gui_prev_frame.disabled = False

        nonlocal viz_data
        viz_data.remove_prev_data()
        remove_old_bodies.disabled = False

    @mode_dropdown.on_update
    def _(_):
        nonlocal viz_data
        nonlocal prev_multitimestep_state
        viz_data.mode = mode_dropdown.value

        # Reset state tracking when switching modes
        prev_multitimestep_state = None

        # Toggle folder visibility
        if viz_data.mode == "Playback":
            playback_folder.visible = True
            multitimestep_folder.visible = False

            # Hide all multitimestep handles
            viz_data.hide_multitimestep_handles()

            # Make sure regular handles are visible
            viz_data.visible_gt_data(show_gt_checkbox.value)
            viz_data.change_visible_pred_samples()

        else:  # Multitimestep
            playback_folder.visible = False
            multitimestep_folder.visible = True

            # Hide all regular body handles
            viz_data.invisible_all_data()

            # Create multitimestep handles on demand
            if viz_data.skin_weights is not None and len(viz_data.multi_timestep_handles) != len(multitimestep_sliders):
                viz_data.create_multitimestep_handles(len(multitimestep_sliders), server.scene)

    @gui_num_visible_timesteps.on_update
    def _(_):
        nonlocal prev_multitimestep_state
        nonlocal viz_data

        # Force update on next frame
        prev_multitimestep_state = None

        # Hide all existing multitimestep handles before recreating
        viz_data.hide_multitimestep_handles()

        create_timestep_sliders()

    @sample_num_dropdown.on_update
    def _(_):
        nonlocal viz_data
        nonlocal server
        nonlocal prev_multitimestep_state

        if len(viz_data.visible_sample_idx_list) == int(sample_num_dropdown.value):
            return

        sample_num = int(sample_num_dropdown.value)
        sample_num_ = min(sample_num, viz_data.sample_num)
        visible_sample_idx_list = list(range(sample_num_))

        for sample_check_button in sample_check_buttons[:sample_num_]:
            sample_check_button.value = True
        for sample_check_button in sample_check_buttons[sample_num_:]:
            sample_check_button.value = False

        prev_multitimestep_state = None
        viz_data.update_visible_sample_idx_list(visible_sample_idx_list)

    data_name = None
    subdir_name = None

    @subdir_dropdown.on_update
    def _(_):
        nonlocal subdir_name
        nonlocal data_name

        if subdir_name == subdir_dropdown.value:
            return
        subdir_name = subdir_dropdown.value

        if subdir_dropdown.value == "None":
            file_dropdown.options = get_file_list()
        else:
            file_dropdown.options = get_file_list(data_root_dir / subdir_dropdown.value)
            file_dropdown.value = "None"
            data_name = "None"

    @file_dropdown.on_update
    def _(_):
        nonlocal data_name
        nonlocal raw_data
        nonlocal prev_multitimestep_state

        if file_dropdown.value in [data_name, "None"]:
            return

        data_name = file_dropdown.value
        if subdir_dropdown.value == "None":
            raw_data = np.load(data_root_dir / file_dropdown.value)
        else:
            raw_data = np.load(
                data_root_dir / subdir_dropdown.value / file_dropdown.value
            )

        nonlocal is_update_ok
        is_update_ok = False
        gui_playing.value = False
        prev_multitimestep_state = None

        load_data()
        gui_timestep.value = 0
        gui_playing.value = True
        gui_timestep.disabled = True
        gui_next_frame.disabled = True
        gui_prev_frame.disabled = True

        is_update_ok = True

    @hand_pose_dropdown.on_update
    def _(_):
        nonlocal viz_data
        nonlocal raw_data
        nonlocal is_update_ok
        nonlocal prev_multitimestep_state

        if viz_data.hand_pose_type == hand_pose_dropdown.value:
            return

        viz_data.hand_pose_type = hand_pose_dropdown.value

        # Reload data if already loaded
        if raw_data is not None:
            is_update_ok = False
            gui_playing.value = False
            prev_multitimestep_state = None
            load_data()
            gui_timestep.value = 0
            gui_playing.value = True
            is_update_ok = True

    for sample_check_button in sample_check_buttons:

        @sample_check_button.on_update
        def _(_):
            visible_sample_idx_list = []
            for sample_idx, sample_check_button in enumerate(sample_check_buttons):
                if sample_check_button.value:
                    visible_sample_idx_list.append(sample_idx)

            nonlocal viz_data
            nonlocal server
            nonlocal prev_multitimestep_state

            if viz_data.visible_sample_idx_list == visible_sample_idx_list:
                return

            prev_multitimestep_state = None
            viz_data.update_visible_sample_idx_list(visible_sample_idx_list)
            sample_num_dropdown.value = str(len(visible_sample_idx_list))

            # Update color controls for newly visible samples
            update_color_controls()

    # ------------------------------------------------------------------
    # Main loop callback
    # ------------------------------------------------------------------

    prev_time = time.time()
    handle = None

    def loop_cb() -> int:
        start, end = gui_start_end.value
        duration = end - start

        if get_viser_file.value is False:
            nonlocal prev_time
            now = time.time()
            sleepdur = 1.0 / gui_framerate.value - (now - prev_time)
            inc = 1
            if sleepdur > 0.0:
                time.sleep(sleepdur)
            elif sleepdur < 0.0:
                inc = np.ceil((now - prev_time) * gui_framerate.value)
                sleepdur_ = inc / gui_framerate.value - (now - prev_time)
                time.sleep(sleepdur_)
            prev_time = now
            if gui_playing.value:
                gui_timestep.value = (
                    gui_timestep.value + inc - start
                ) % duration + start
            do_update()
            return gui_timestep.value
        else:
            # Save trajectory.
            nonlocal handle
            if handle is None:
                handle = server._start_scene_recording()
                handle.set_loop_start()
                gui_timestep.value = start

            assert handle is not None
            handle.insert_sleep(1.0 / gui_framerate.value)
            gui_timestep.value = (gui_timestep.value + 1 - start) % duration + start

            if gui_timestep.value == start:
                get_viser_file.value = False
                server.send_file_download(
                    "recording.viser", content=handle.end_and_serialize()
                )
                handle = None

            do_update()

            return gui_timestep.value

    return loop_cb


def visualize(
    server: viser.ViserServer,
    data_root_dir: Path | None,
) -> None:
    assert data_root_dir is not None

    loop_cb = load_visualizer(server, data_root_dir)

    while True:
        loop_cb()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--port", type=int, default=8084)
    args.add_argument("--data_dir", type=str, default="./dfot_outputs")
    args = args.parse_args()

    server = viser.ViserServer(port=args.port)
    visualize(server, data_root_dir=Path(args.data_dir))
