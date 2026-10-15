# DAHCFormer

PyTorch implementation of **Experience-Driven Cross-Task Interaction for Higher-Quality Zero-Shot Image Fusion**.

DAHCFormer learns from infrared–visible image fusion (IVIF) and synthetic multi-focus image fusion (MFIF). The trained student processes one aligned input pair with frozen parameters, including for unseen multi-exposure, NIR–VIS, medical, and remote-sensing fusion tasks.

- **Hybrid Cross-Fusion Gating Mechanism (HCFGM):** windowed task interaction with channel gains conditioned on the updated main and auxiliary features. GateNet produces channel weights that are broadcast across spatial positions.
- **Ensemble Pseudo-Labeling (EP):** the fusion and reconstruction outputs of one exponential moving average (EMA) teacher form additional IVIF supervision through source-error confidence, pixel reliability weighting, and batch-quality screening.
- **Single-pair inference:** one shared encoding initializes both task streams. The digital photography (DP) stream is main and the multimodal (MM) stream is auxiliary. The teacher and reconstruction decoder are unused during inference.

## Installation

Python 3.10 or later is required. The implementation was checked locally with Python 3.13, PyTorch 2.8.0, torchvision 0.23.0, and CUDA 12.6 on an NVIDIA RTX 4060 GPU.

```bash
git clone https://github.com/zeizei0610/DAHCFormer.git
cd DAHCFormer
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the CUDA 12.6 PyTorch build, install the matching packages first:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Training uses a CUDA GPU. Inference supports CPU and CUDA. The pretrained DenseNet121 used by the training loss is downloaded by torchvision on the first training run.

## Repository layout

```text
DAHCFormer/
├── GIFNet_model.py                 # Shared backbone, HCFGM, and decoders
├── GIFNetDataset.py                # Same-scene IVIF / synthetic MFIF loader
├── args.py                        # Architecture and default training settings
├── train.py                       # Alternating training with EMA and EP
├── test.py                        # Grayscale and color pair inference
├── evaluate_fusion_metrics.py      # EI, VIF, SCD, and AG evaluation
├── utils.py                       # Image and loss utilities
├── testMat.py                     # Training-loss plotting
├── pytorch_msssim/                 # Local SSIM / MS-SSIM implementation
├── tools/
│   ├── checkpoint_health.py        # Checkpoint output diagnostics
│   ├── compute_efficiency.py       # Forward-path profiling comparison
│   └── efficiency_profile.py       # Attention-aware MAC, latency, and memory profiling
├── Matlab_SeparateChannelsIntoFused.m
├── requirements.txt
└── README.md
```

The backbone and dataset filenames retain their original names for compatibility. Training data, checkpoints, evaluation outputs, and virtual environments are local resources excluded by `.gitignore`.

## Training data

Use the paired source-task training data supplied with [GIFNet](https://github.com/AWCXV/GIFNet). Each indexed group contains an infrared image, a visible image, and complementary near-focused and far-focused images synthesized from that visible image. The original visible image supplies the all-in-focus target for the synthetic MFIF pair.

Arrange the data as follows:

```text
train_data/GIFNet_Train/
├── infrared/train/1.jpg ... 19999.jpg
├── visible/train/1.jpg ... 19999.jpg
├── visible_focus_near/train/1.jpg ... 19999.jpg
└── visible_focus_far/train/1.jpg ... 19999.jpg
```

Matching indices must represent the same scene. Training patches are grayscale, 128 × 128 pixels, scaled to [0, 1]. The loader uses indices `1` through `train-num - 1`; the default `--train-num 20000` therefore loads 19,999 groups. Target-task images are excluded from source training.

## Train DAHCFormer

```bash
python train.py --trainDataRoot train_data/GIFNet_Train --gate --teacher --seed 0 --output-dir model
```

The default configuration is:

| Setting | Value |
| --- | --- |
| Training passes | 2 |
| Batch size | 8; final partial batch retained |
| Adam learning rate | 0.001, constant |
| Adam betas | (0.9, 0.999) |
| Optimizer updates | 10,000: 5,000 IVIF + 5,000 MFIF |
| Gate regularization weight | 0.001 |
| EMA decay | 0.5 |
| Confidence inverse temperature | 2.0 |
| Fusion source-error mixing weight | 0.5 |
| Reconstruction coefficient cap | 0.6 before renormalization |
| Pixel / batch-quality thresholds | (0.97, 0.97) |
| EP warm-up | First 1,000 outer iterations |
| EP coefficient schedule | 0.06 to 1.0 over the following 1,000 outer iterations |

Each outer iteration performs an IVIF update followed by an MFIF update. EP applies to IVIF after warm-up when the batch-quality acceptance condition is met.

The final checkpoint is `model/MTFusion_net_epoch_2_twoBranches.model`. With EP enabled, it contains `student` and `teacher` state dictionaries. `test.py` loads the student automatically. Training also writes `run_config.json`, `run_summary.json`, and loss histories to the output directory. Checkpoints are not bundled in Git; use your trained checkpoint in the commands below.

### Component and sensitivity runs

```bash
# Backbone control: HCFGM and EP disabled
python train.py --no-gate --no-teacher --output-dir runs/backbone

# HCFGM without EP
python train.py --gate --no-teacher --output-dir runs/hcfgm

# EP components: fusion, dual, confidence, or full
python train.py --gate --teacher --ep-mode confidence --output-dir runs/ep-confidence

# EMA sensitivity
python train.py --ema-decay 0.9 --output-dir runs/ema-09

# Training configuration sensitivity
python train.py --batch-size 4 --learning-rate 0.0005 --epochs 2 --output-dir runs/training-config

# Within-batch auxiliary-pair permutation
python train.py --gate --no-teacher --permute-auxiliary --output-dir runs/permuted
```

Use `--seed 0`, `--seed 1`, and `--seed 2` for repeated training. `python train.py --help` lists all available options.

## Inference

Source folders must contain spatially aligned images with identical filenames and dimensions. The `ir` and `vis` argument names identify the first and second sources; the same interface serves all fusion tasks. For a single-image enhancement operation, pass the same folder as both sources.

### Grayscale pair

```bash
python test.py --checkpoint model/MTFusion_net_epoch_2_twoBranches.model --test_ir_root data/source1 --test_vis_root data/source2 --save_path outputs/fusion --device cpu
```

Use `--device cuda` for GPU inference.

### Infrared–visible or NIR–VIS

```bash
python test.py --checkpoint model/MTFusion_net_epoch_2_twoBranches.model --test_ir_root data/infrared --test_vis_root data/visible --VIS_IS_RGB --save_path outputs/ivif --device cuda
```

For NIR–VIS, supply the near-infrared folder as the first source and the visible folder as the second source.

### Multi-focus or multi-exposure

```bash
python test.py --checkpoint model/MTFusion_net_epoch_2_twoBranches.model --test_ir_root data/source1 --test_vis_root data/source2 --IR_IS_RGB --VIS_IS_RGB --save_path outputs/mfif --device cuda
```

### PET / MRI

```bash
python test.py --checkpoint model/MTFusion_net_epoch_2_twoBranches.model --test_ir_root data/pet --test_vis_root data/mri --IR_IS_RGB --save_path outputs/medical --device cuda
```

Color inputs are converted to luminance for fusion. With one color source, its chrominance is restored; with two color sources, their chrominance is combined by the implemented rule.

### Remote sensing

Fuse each multispectral band with the aligned panchromatic image separately:

```bash
python test.py --checkpoint model/MTFusion_net_epoch_2_twoBranches.model --test_ir_root images/Remote/MS_band1 --test_vis_root images/Remote/PAN --save_path outputsRemoteBand1 --device cuda
```

Repeat for bands 2–4, replacing both the input band folder and output folder. `Matlab_SeparateChannelsIntoFused.m` combines the four outputs using the original band folders; its base-layer histogram matching is an additional postprocessing operation.

## Fusion metrics

```bash
python evaluate_fusion_metrics.py --source1 data/source1 --source2 data/source2 --fused outputs/fusion --task IVIF --output-dir outputs/metrics
```

The evaluator writes per-image scores to `per_image.csv` and dataset means, definitions, and excluded-image information to `summary.json`.

- **EI:** edge intensity, based on Sobel gradient magnitude.
- **VIF:** summed visual information fidelity relative to the two sources.
- **SCD:** sum of correlations of differences.
- **AG:** average gradient.

EI and AG describe gradient strength; interpret them together with information-preservation metrics. A higher gradient score alone does not establish more faithful fusion.

## Efficiency profiling

```bash
python tools/efficiency_profile.py --gate --height 128 --width 128 --warmup 20 --repeat 100 --output outputs/efficiency.json
```

This profiles the deployed student forward path using seeded random weights. It reports parameters, MACs including attention products, forward latency, and peak CUDA tensor memory when CUDA is available. It excludes image preprocessing, transfer, postprocessing, and teacher training. MAC estimates are reported as MACs, rather than an exact count of every floating-point operation.

## Acknowledgements

The backbone, paired training data, and alternating task-learning framework build on [GIFNet](https://github.com/AWCXV/GIFNet), by Cheng et al., *One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion*, CVPR 2025. DAHCFormer adds HCFGM channel modulation and EP teacher supervision. Please acknowledge GIFNet when using the inherited components.
