# MAGNet : Diffusion Forcing for Multi-Agent Interaction Sequence Modeling

<p align="center">
  <a href="https://von31.github.io/MAGNet/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2512.17900"><img src="https://img.shields.io/badge/arXiv-2512.17900-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"></a>



https://github.com/user-attachments/assets/d5bade34-5b74-45f1-8927-ef4f1cd4b821
 
 

<!-- 
**Diffusion Forcing for Multi-Agent Interaction Sequence Modeling** -- a diffusion-based framework for generating multi-person (2-4) coordinated motion using SMPL-X body models. -->

<table><tr><td>
  <strong>Diffusion Forcing for Multi-Agent Interaction Sequence Modeling -- a diffusion-based framework for generating multi-person (2-4) coordinated motion using SMPL-X body models.</strong><br />
  <small></small>Vongani H.&nbsp;Maluleke<sup>&sect;*</sup>, Kie&nbsp;Horiuchi<sup>&dagger;&sect;*</sup>, Lea Wilken<sup>&sect;</sup>, Evonne Ng<sup>&Dagger;</sup>, Jitendra&nbsp; Malik<sup>&sect;</sup> and&nbsp;Angjoo&nbsp;Kanazawa<sup>&sect;</sup><small>

</td></tr>
</table>
<sup>&dagger;</sup><em>Sony Group Corporation </em>, <sup>&Dagger;</sup><em>Meta</em>, <sup>&sect;</sup><em>UC Berkeley</em>, <sup>*</sup><em>Equal Contribution</em>

### Planned Timeline

- [x] (Dec 3, 2025) Paper release
- [x] (Mar 18, 2026) Code Release
- [ ] (Mar 20, 2026) Data Preparation




The system is built in two stages:

1. **Pose VQ-VAE** -- learns a discrete latent codebook.
2. **DFOT (Diffusion Forcing Transformer)** -- a diffusion transformer that operates in the
   VQ-VAE latent space to generate long, multi-person motion sequences with
   various conditioning modes (joint, partner prediction, motion control,
   inpainting, inbetweening, synchronous / asynchronous turn-taking).

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)

---

## Environment Setup

### Prerequisites

- Linux (tested on Ubuntu)
- CUDA 12.4 compatible GPU
- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Create the conda environment

```bash
conda env create -f environment.yml
conda activate mc
```

### SMPL-X body model

Place the neutral SMPL-X model at `data/smplx/SMPLX_NEUTRAL.npz`.
You can obtain it from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/).

---

## Data Preparation
The project supports several multi-person motion datasets:

| Dataset | Key |
|---|---|
| DD100 | `DUOLANDO` |


The preprocessed data and pretrained checkpoints can be downloaded automatically:

```bash
bash scripts/download_data_checkpoint.sh
```

Or manually from [Google Drive](https://drive.google.com/drive/folders/1Mu2M6kERuOz0-4oOsUSA4yzUstC0yCKc?usp=drive_link).

The preprocessing pipeline is coming soon.

## Training

Training proceeds in two stages. Both stages use YAML config files under
`configs/` and shell scripts under `scripts/`.

### Stage 1 -- Pose VQ-VAE

Train the pose VQ-VAE that learns a discrete motion codebook:

```bash
bash scripts/run_vqvae_train.sh [GPU] [CONFIG]
```

<details>
<summary>Arguments & available configs</summary>

| Argument | Default | Description |
|---|---|---|
| `GPU` | `0` | CUDA device index |
| `CONFIG` | `configs/train/vqvae/dd100.yaml` | Training config |

Example — train on GPU 1:

```bash
bash scripts/run_vqvae_train.sh 1
```

Available VQ-VAE training configs:

```
configs/train/vqvae/
  dd100.yaml
```

</details>

### Stage 2 -- DFOT Diffusion Model

Once the VQ-VAE is trained, train the DFOT diffusion transformer:

```bash
bash scripts/run_dfot_train.sh [GPU] [CONFIG]
```

**Important:** Update `vqvae_model_path` in the DFOT config to point to your
trained VQ-VAE checkpoint directory.

<details>
<summary>Arguments & available configs</summary>

| Argument | Default | Description |
|---|---|---|
| `GPU` | `0` | CUDA device index |
| `CONFIG` | `configs/train/dfot/dd100.yaml` | Training config |

Available DFOT training configs:

```
configs/train/dfot/
  dd100.yaml
```

</details>

<details>
<summary>Key training parameters</summary>

| Parameter | Description | Example |
|---|---|---|
| `batch_size` | Training batch size | `256` |
| `learning_rate` | Initial learning rate | `2e-4` |
| `total_steps` | Total training steps | `300000` |
| `warmup_steps` | LR warmup steps | `1000` |
| `person_num` | Number of persons in the dataset (1-4) | `2` |
| `vqvae_model_path` | Path to trained VQ-VAE | `experiments/vqvae/<name>/v0/checkpoints_*` |

Training logs are sent to **WandB** and **TensorBoard**. Checkpoints are saved
every 5,000 steps under `experiments/dfot/<experiment_name>/`.

</details>

---

## Inference

### VQ-VAE inference

```bash
bash scripts/run_vqvae_inference.sh [CONFIG]
```

Default config: `configs/inference/vqvae/dd100.yaml`

### DFOT inference

```bash
bash scripts/run_dfot_inference.sh [CONFIG]
```

Default config: `configs/inference/dfot/dd100.yaml`

Output `.npz` files are saved to the directory specified by `output_dir` in the config.

Available per-task inference configs:

```
configs/inference/dfot/
  dd100.yaml                    # joint (default)
  dd100_joint.yaml              # joint generation
  dd100_partner_prediction.yaml # partner prediction
  dd100_turn_taking.yaml        # agentic turn-taking
  dd100_motion_control.yaml     # motion control
  dd100_partner_inpainting.yaml # partner inpainting
  dd100_inbetweening.yaml       # inbetweening
```

<details>
<summary>Sampling tasks</summary>

The inference config's `sampling_cfg.sampling_task` field controls the
generation mode:

| Task | Description |
|---|---|
| `joint` | Generate all persons jointly from context |
| `partner_prediction` | Predict one person's motion given the other's full GT sequence |
| `agentic_turn_taking` | Leader-follower asynchronous generation |
| `agentic_sync` | Both agents denoise simultaneously |
| `motion_control_live` | Generate person B conditioned on person A's GT (past and current) |
| `partner_inpainting` | Inpaint one person given the other's full sequence |
| `inbetweening` | Fill in motion between sparse keyframes |

</details>

<details>
<summary>Per-task config settings</summary>

To switch between tasks, update the `sampling_cfg` block in your inference
YAML. The key fields that change per task are shown below.

#### Joint generation

```yaml
sampling_cfg:
  sampling_task: "joint"
  sampling_schedule: "causal_uncertainty"
  sampling_subseq_len: 16
  ar_seq_stride: 16
  sampling_seq_len: 200
  root_transform_mode: temporal_partner
```

#### Partner prediction (predict person B given person A's full GT)

```yaml
sampling_cfg:
  sampling_task: "partner_prediction"
  sampling_schedule: "causal_uncertainty"
  sampling_subseq_len: 1
  ar_seq_stride: 8
  sampling_seq_len: 200
  root_transform_mode: temporal_partner
```

#### Asynchronous turn-taking (leader / follower)

```yaml
sampling_cfg:
  sampling_task: "agentic_turn_taking"
  sampling_schedule: "causal_uncertainty"
  sampling_subseq_len: 4
  ar_seq_stride: 8
  sampling_seq_len: 200
  root_transform_mode: temporal
```

#### Motion control (generate B conditioned on A's past + current GT)

```yaml
sampling_cfg:
  sampling_task: "motion_control_live"
  sampling_schedule: "causal_uncertainty"
  sampling_subseq_len: 4
  offset_proportion: 0.75
  ar_seq_stride: 8
  sampling_seq_len: 200
  root_transform_mode: temporal
```

#### Partner inpainting (inpaint one person given the other's full sequence)

```yaml
sampling_cfg:
  sampling_task: "partner_inpainting"
  sampling_schedule: "full_sequence"
  sampling_subseq_len: 2
  offset_proportion: 0.0
  ar_seq_stride: 32
  sampling_seq_len: 150
  root_transform_mode: temporal_partner
```

#### Inbetweening (fill between sparse keyframes)

```yaml
sampling_cfg:
  sampling_task: "inbetweening"
  sampling_schedule: "causal_uncertainty"
  sampling_subseq_len: 2
  ar_seq_stride: 4
  sampling_seq_len: 64
  root_transform_mode: temporal
  inbetweening_key_frame_indices:
    - 0
    - 63
```

</details>

---

## Evaluation

Evaluate generated motions against ground truth:

```bash
python -m libs.utils.eval \
    --data_dir <path_to_inference_output> \
    --fps 30 \
    --sample_num 10
```

<details>
<summary>Metrics computed</summary>

| Metric | Description |
|---|---|
| **FD (Frechet Distance)** | Distribution-level quality for body/hand positions and velocities |
| **Diversity** | Sample diversity across generated sequences |
| **MPJPE** | Mean Per-Joint Position Error (mm) |
| **MPJVE** | Mean Per-Joint Velocity Error |
| **Person Correlation** | Synchrony between generated persons |
| **Foot Skating** | Foot-ground contact consistency |
| **Penetration** | Inter-person body penetration |

</details>

<details>
<summary>Additional flags</summary>

```bash
python -m libs.utils.eval \
    --data_dir ./outputs/dfot/<experiment> \
    --fps 30 \
    --sample_num 10 \
    --whole_seq            # evaluate full sequences (not windowed) \
    --seq_len 200          # specific sequence length to evaluate \
    --test_set data/data_split/test.yaml \
    --cuda                 # use GPU for evaluation
```

Results are saved as `eval_dict_c*.yaml` and `eval_dict_c*.csv` inside the
output directory.

</details>

---

## Visualization

Visualize generated motions in 3D using [Viser](https://viser.studio/):

```bash
bash scripts/run_visualizer.sh [DATA_DIR]
```

Default data directory: `outputs/dfot/dd100`

This launches an interactive 3D viewer in your browser where you can:

- Play / pause / scrub through the generated motion
- View multiple persons with distinct color coding
- Inspect individual joints and body parts

---

<details>
<summary><h2>Project Structure</h2></summary>

```
MAGNet/
├── configs/
│   ├── train/
│   │   ├── vqvae/                  # VQ-VAE training configs
│   │   └── dfot/                   # DFOT training configs
│   ├── inference/
│   │   ├── vqvae/                  # VQ-VAE inference configs
│   │   └── dfot/                   # DFOT inference configs (per-task)
│   └── wandb.yaml                  # Weights & Biases config
│
├── libs/
│   ├── train/
│   │   ├── vqvae_train.py          # VQ-VAE training entry point
│   │   └── dfot_train.py           # DFOT training entry point
│   ├── inference/
│   │   ├── vqvae_inference.py      # VQ-VAE inference entry point
│   │   └── dfot_inference.py       # DFOT inference entry point
│   ├── model/
│   │   ├── dfot/                   # DFOT diffusion model
│   │   │   ├── network.py          # Main network (DFOTBase + DFOTNetwork)
│   │   │   ├── diffusion_transformer.py
│   │   │   ├── diffusion.py        # Diffusion utilities
│   │   │   ├── dfot_guidance.py    # Classifier-free guidance
│   │   │   └── config.py
│   │   └── vqvae/                  # Pose VQ-VAE model
│   │       ├── network.py
│   │       ├── pose_vqvae.py
│   │       ├── encdec.py           # Encoder / decoder (Conv1d + ResNet blocks)
│   │       ├── quantizer.py
│   │       └── config.py
│   ├── dataloaders/                # Dataset loading, batching, and preprocessing
│   ├── preproc/                    # Data splits and dataset analysis
│   ├── utils/
│   │   ├── eval.py                 # Evaluation metrics
│   │   ├── fncsmpl.py              # SMPL-X body model
│   │   ├── transforms/             # SO(3) / SE(3) utilities
│   │   ├── root_transform_processor.py
│   │   └── training_utils.py
│   └── viz/                        # Visualization (Viser-based)
│       ├── visualizer.py
│       └── viz_manager.py
│
├── scripts/
│   ├── download_data_checkpoint.sh # Download data and pretrained models
│   ├── run_vqvae_train.sh          # Stage 1 training
│   ├── run_dfot_train.sh           # Stage 2 training
│   ├── run_vqvae_inference.sh      # VQ-VAE inference
│   ├── run_dfot_inference.sh       # DFOT inference
│   └── run_visualizer.sh           # Launch Viser 3D visualizer
│
├── checkpoints/                    # Pretrained model checkpoints
│   ├── vqvae/magnet_dd100/
│   └── dfot/magnet_dd100/
│
├── data/
│   └── smplx/                      # SMPL-X body model files
│
├── demo_inference.ipynb            # Interactive inference demo
└── environment.yml                 # Conda environment specification
```

</details>

---

<details>
<summary><h2>Configuration Reference</h2></summary>

All configs use [OmegaConf](https://omegaconf.readthedocs.io/) YAML format
with variable interpolation (`${variable}`).

### DFOT training config keys

| Key | Type | Description |
|---|---|---|
| `experiment_dir` | `str` | Root directory for experiment outputs |
| `experiment_name` | `str` | Name of this experiment run |
| `person_num` | `int` | Number of persons (1-4) |
| `dataset_list` | `list[str]` | Datasets to train on |
| `vqvae_model_path` | `str` | Path to pretrained VQ-VAE checkpoint |
| `batch_size` | `int` | Training batch size |
| `learning_rate` | `float` | Peak learning rate |
| `total_steps` | `int` | Total training iterations |
| `warmup_steps` | `int` | Linear warmup steps |
| `model_cfg` | `dict` | DFOT model architecture config |

### DFOT inference / sampling config keys

| Key | Type | Description |
|---|---|---|
| `sampling_schedule` | `str` | `full_sequence`, `autoregressive`, or `causal_uncertainty` |
| `sampling_task` | `str` | See [Sampling tasks](#sampling-tasks) |
| `sampling_steps` | `int` | Number of denoising steps |
| `sampling_num` | `int` | Number of samples to generate per sequence |
| `sampling_seq_len` | `int` | Length of generated sequence (frames) |
| `sampling_subseq_len` | `int` | Sub-sequence length per denoising window |
| `context_seq_len` | `int` | Number of context frames (must be a multiple of 4, minimum 4) |
| `ar_seq_stride` | `int` | Stride for auto-regressive sliding window |
| `offset_proportion` | `float` | Noise offset proportion (task-dependent) |
| `cfg_scale_dict` | `dict` | Classifier-free guidance scales |
| `root_transform_mode` | `str` | `temporal` or `temporal_partner` |
| `denoising_process` | `str` | Denoising process (`ddim`) |
| `ddim_eta` | `float` | DDIM stochasticity parameter |
| `is_update_history_transforms` | `bool` | Update root transforms from generated history |
| `use_temporal_smoothing` | `bool` | Enable temporal smoothing between windows |
| `temporal_smoothing_strength` | `float` | Blending weight for temporal smoothing |
| `use_smoothing_guidance` | `bool` | Enable smoothing guidance during denoising |
| `smoothing_guidance_weight` | `float` | Strength of smoothing guidance |
| `use_foot_skating_guidance` | `bool` | Enable foot skating reduction guidance |
| `foot_skating_guidance_weight` | `float` | Strength of foot skating guidance |
| `inbetweening_key_frame_indices` | `list[int]` | Keyframe indices (inbetweening task only) |

</details>
