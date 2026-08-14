# Vision-Based RL Benchmark & Evaluation Suite
**ICVGIP 2026 Submission Repository**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29.0-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/stable--baselines3-2.2.1-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A standardized benchmark, empirical validation framework, and baseline suite for **Vision-Based Reinforcement Learning (Visual RL)**. This repository addresses five major priority research gaps in visual RL—generalization under distribution shift, evaluation standardization, sample efficiency, reward design, and visual attention interpretability—with theoretical grounding in recent literature (*Lyu et al., 2024; Ma et al., 2025; Wu et al., 2025; Guo et al., 2025*).

---

## Key Features & Research Contributions

- **Procgen & Headless Visual Environments:** Standardized integration of Procgen (`procgen:procgen-coinrun-v0`) and fast 2D custom visual environments (`SingleObjectTracking-v0`, `MultiStageNavigation-v0`).
- **Canonical Procgen Wrapper:** Unified `ProcgenGymnasiumWrapper` ([envs/wrappers.py](file:///d:/gitfork/vision%20rl/envs/wrappers.py)) enforcing explicit episode termination vs truncation, RGB normalization (`[0, 1]`), and channel-first (`CHW`) observation formatting.
- **Predictive Data Augmentation Selection (TTA Proxy):** Verified Test-Time Augmentation (TTA) feature distance proxy achieving a **$16.1\times$ compute speedup** ($62.8\text{s}$ vs $1013.9\text{s}$) with statistically significant pooled Spearman rank correlation ($\rho = -0.549, p = 1.29 \times 10^{-11}$) across 13 candidate augmentations and 10 visual shift severity settings.
- **Photometric & Appearance-Based Shift Scoping:** Rigorous benchmark focus on continuous visual perturbations (additive Gaussian noise, moving distractors, Cutout occlusions, spatial blur).
- **Canonical Evaluation Pipeline:** Reconciled evaluation pipeline ([utils/eval_pipeline.py](file:///d:/gitfork/vision%20rl/utils/eval_pipeline.py)) reporting 95% Student-$t$ Confidence Intervals ($\pm \text{CI}_{95}$) across random seeds.
- **Visual Attention Heatmaps & Latency Audit:** Saliency diagnostics ([utils/saliency.py](file:///d:/gitfork/vision%20rl/utils/saliency.py)) and visualizers ([utils/visualization.py](file:///d:/gitfork/vision%20rl/utils/visualization.py)).

---

## Directory Structure

```
vision-rl/
├── envs/                         # Custom Gymnasium visual environments & wrappers
│   ├── __init__.py               # Environment registration
│   ├── tracking_envs.py          # Single-Object and Active Tracking envs
│   ├── navigation_envs.py        # MultiStageNavigation-v0 environment
│   └── wrappers.py               # Canonical ProcgenGymnasiumWrapper, Noise, Distractor, Cutout wrappers
├── baselines/                    # Research scripts & GPU deployment suite
│   ├── run_scale_experiment.py   # Multi-seed 2.5M-step Procgen scale-up runner
│   ├── predictive_augmentation_selection.py # TTA predictive selection proxy experiment
│   ├── smoke_procgen.py          # Procgen 300k-step smoke test & perturbation suite
│   ├── deploy_remote_gpu.py      # SSH/SFTP cluster deployment & nohup launcher
│   ├── check_predictive_remote.py# Remote GPU status & result verification utility
│   └── read_tb_logs.py           # TensorBoard event log parser
├── utils/                        # Metrics, statistics & visualization
│   ├── eval_pipeline.py          # Unified canonical evaluation & 95% CI pipeline
│   ├── dataset_partitions.py     # Train / Validation / Test benchmark splits
│   ├── metrics.py                # CLE, Success Rate metrics
│   ├── plotting.py               # Experiment result plotting utilities
│   ├── saliency.py               # Grad-CAM visual attention heatmap generator
│   └── visualization.py          # Publication figure visualizer
├── results/                      # Experiments, logs, tables & figures
├── CONTEXT.md                    # Project context log & session history
├── SCOPE.md                      # Project scope & core research claims
├── RESEARCH_PROGRESS.md          # Research hypotheses & experimental ledger
├── HOW_TO_RUN.md                 # Step-by-step execution guide
├── references.md                 # Standardized literature references
├── requirements.txt              # Environment dependencies
└── README.md                     # Main repository documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shounak-max/vision-rl.git
   cd vision-rl
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   # On Linux/macOS: source venv/bin/activate
   # On Windows PowerShell: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Procgen 300k-Step Smoke Test & Perturbation Suite
Run the 300,000-timestep Procgen CoinRun training and evaluate across 11 visual perturbation wrappers:
```bash
python baselines/smoke_procgen.py
```

### 2. Predictive Augmentation Selection via TTA Proxy
Evaluate feature centroid distance vs ground-truth policy return degradation across 13 candidate augmentations and 10 visual shift conditions ($16.1\times$ speedup):
```bash
python baselines/predictive_augmentation_selection.py
```

### 3. Multi-Seed Scale-Up Experiment (2.5M Timesteps, 3 Seeds)
Run the canonical scale-up experiment comparing PPO vs DataAugmented_PPO on `procgen:procgen-coinrun-v0`:
```bash
python baselines/run_scale_experiment.py
```

### 4. Remote GPU Cluster Deployment
Deploy and monitor experiments on a remote GPU cluster via SSH/SFTP:
```bash
python baselines/deploy_remote_gpu.py
```

---

## Summary of Key Empirical Results

### Predictive Augmentation Selection (TTA Proxy)
| Evaluation Mode | Execution Time | Pooled Spearman Correlation ($\rho$) | Statistical Significance ($p$-value) | Compute Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Ground-Truth Rollouts** | $1013.90\text{ s}$ | — | — | $1.0\times$ |
| **TTA Representation Distance Proxy** | **$62.81\text{ s}$** | **$-0.549$** | **$p = 1.29 \times 10^{-11}$** | **$16.1\times$** |

### Per-Condition Correlation Highlights (Severe Perturbations)
- **High Gaussian Noise (`Noise_High`)**: Spearman $\rho = -0.940$
- **Moving Distractors (`Distractor_3`)**: Spearman $\rho = -0.863$
- **Cutout Occlusion (`Cutout_0.3`)**: Spearman $\rho = -0.712$

---

## References & Citations

If you use this benchmark suite in your research, please cite the underlying theoretical papers:

- **Lyu et al. (2024):** *Understanding What Affects the Generalization Gap in Visual Reinforcement Learning: Theory and Empirical Evidence.* JAIR.
- **Ma et al. (2025):** *A Comprehensive Survey of Data Augmentation in Visual Reinforcement Learning.* IJCV.
- **Wu et al. (2025):** *Reinforcement Learning in Vision: A Survey.* arXiv:2508.08189.
- **Guo et al. (2025):** *Improving Vision-Language-Action Model with Online Reinforcement Learning (iRe-VLA).* ICRA.
- **Lu et al. (2022):** *Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations (V-D4RL).* TMLR.
