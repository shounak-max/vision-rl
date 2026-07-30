# Vision-Based RL Benchmark & Evaluation Suite
**ICVGIP 2026 Submission Repository**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29.0-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/stable--baselines3-2.2.1-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A standardized benchmark, empirical validation framework, and baseline suite for **Vision-Based Reinforcement Learning (Visual RL)**. This repository addresses five major priority research gaps in visual RL—generalization under distribution shift, evaluation standardization, sample efficiency, reward design, and visual attention interpretability—with theoretical grounding in recent literature (*Lyu et al., 2024; Ma et al., 2025; Wu et al., 2025; Guo et al., 2025*).

---

## Key Features & Research Contributions

- **Standardized Visual Environments:** Headless, fast 2D visual environments (`SingleObjectTracking-v0`, `MultiObjectTracking-v0`, `ActiveTracking-v0`, `MultiStageNavigation-v0`).
- **Hungarian-Matched Tracking Metric (MOTA):** Implements Multi-Object Tracking Accuracy using Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment`).
- **Representation Distance & Generalization Gap:** Empirical verification of Lyu et al. (2024), measuring CNN embedding distances under Gaussian noise, visual distractors, and viewpoint shifts.
- **Grad-CAM Visual Attention Heatmaps:** Computer-vision-native saliency diagnostics (`utils/saliency.py`) exposing visual attention hijack under moving distractors.
- **Multi-Seed Statistical Engine:** Evaluates performance across 5 seeds (`[0, 42, 100, 123, 999]`) reporting Mean $\pm$ Std and Welch's $t$-test $p$-values ($p = 0.00295$).
- **Offline Visual RL Dataset (V-D4RL Proxy):** Dataset generator producing compressed `.npz` visual transition tuples $(O_t, A_t, R_t, O_{t+1}, \text{done})$ per Lu et al. (2022).
- **Hybrid iRe-VLA Fine-Tuning:** Alternating online PPO updates with offline Behavior Cloning (BC) regularization to stabilize visual representations (Guo et al., 2025).

---

## Directory Structure

```
vision-rl/
├── envs/                         # Custom Gymnasium visual environments & wrappers
│   ├── __init__.py               # Environment registration
│   ├── tracking_envs.py          # Single-Object, Multi-Object, and Active Tracking envs
│   ├── navigation_envs.py        # MultiStageNavigation-v0 (long-horizon reward env)
│   └── wrappers.py               # DataAugmentation, Noise, Distractor, Viewpoint wrappers
├── baselines/                    # Training algorithms, diagnostic & ablation scripts
│   ├── train_baselines.py        # Baseline PPO / SAC runner
│   ├── train_multiseed.py        # 5-Seed statistical significance evaluator
│   ├── run_ablations.py          # Component ablation study runner
│   ├── evaluate_generalization.py # Evaluation under visual distribution shifts
│   ├── evaluate_ood.py           # Corruption severity evaluation runner
│   ├── pretrained_policy.py      # ResNet-18 / Residual vision backbone policies
│   ├── run_pretrained_experiment.py # Scratch CNN vs Pre-trained backbone benchmark
│   ├── reward_diagnostics_advanced.py # Reward hacking diagnostic
│   ├── generate_offline_data.py  # V-D4RL-lite offline dataset generator
│   └── train_ire_vla_lite.py     # iRe-VLA alternating online RL + offline BC runner
├── utils/                        # Metrics, statistics, latency audits & visualization
│   ├── metrics.py                # CLE, Success Rate, Hungarian-Matched MOTA
│   ├── stats.py                  # Welch's t-test, Mann-Whitney U, Confidence Intervals
│   ├── saliency.py               # Grad-CAM visual attention heatmap generator
│   ├── visualization.py          # Publication figure visualizer (t-SNE, grid, degradation)
│   └── compute_audit.py          # Model parameter, FPS, & inference latency auditor
├── results/                      # Experiments, logs, datasets & figures
│   ├── tables/                   # CSV tables & JSON summary metrics
│   ├── figures/                  # Publication-ready plots (PNG/PDF)
│   ├── datasets/                 # Offline visual datasets (.npz)
│   ├── models/                   # Saved policy checkpoints (.zip)
│   └── logs/                     # TensorBoard event logs
├── TECHNICAL_REPORT.md           # Submission-ready technical report & manuscript
├── requirements.txt              # Environment dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Version control ignore rules
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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Multi-Seed Benchmark Evaluation
Run PPO across 5 distinct seeds (`0, 42, 100, 123, 999`) and compute statistical significance:
```bash
python baselines/train_multiseed.py --steps 30000
```
*Outputs:* `results/tables/multiseed_results.csv` and `results/tables/multiseed_summary.json`.

### 2. Grad-CAM Visual Attention Diagnostics
Generate gradient-weighted class activation heatmaps to inspect policy visual focus:
```bash
python utils/saliency.py
```
*Outputs:* `results/figures/gradcam_attention.png`.

### 3. Out-of-Distribution (OOD) Corruption Severity Test
Evaluate performance degradation across noise, distractor count, and viewpoint rotation severities:
```bash
python baselines/evaluate_ood.py --model "results/models/PPO_SingleObjectTracking-v0_s42.zip"
```
*Outputs:* `results/tables/ood_corruption_results.csv`.

### 4. Component Ablation Study
Evaluate the impact of Data Augmentations (`DataAugmentationWrapper`):
```bash
python baselines/run_ablations.py --steps 25000
```
*Outputs:* `results/tables/ablation_results.csv`.

### 5. Generate Offline Visual Dataset (V-D4RL Proxy)
Generate a compressed visual dataset for offline RL:
```bash
python baselines/generate_offline_data.py --steps 5000
```
*Outputs:* `results/datasets/offline_dataset.npz`.

### 6. iRe-VLA Alternating Training (RL + Offline BC)
Run hybrid alternating online RL + offline Supervised Learning regularization:
```bash
python baselines/train_ire_vla_lite.py --iters 5
```

### 7. Generate Paper Figures & Compute Audit
```bash
python utils/visualization.py
python utils/compute_audit.py
```

---

## Summary of Results

### Multi-Seed Benchmark Performance (Table 1)
| Policy Algorithm | Evaluated Seeds | Success Rate (%) | Mean CLE (pixels) | Welch's $t$-test ($p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Policy** | 5 | $3.40 \pm 1.20\%$ | $44.12 \pm 3.10$ | — |
| **PPO (CNN Policy)** | 5 | $0.42 \pm 0.51\%$ | $20.57 \pm 2.14$ | **$p = 0.00295$ ($p < 0.01$)** |

### Compute & Latency Audit (Table 4)
| Algorithm | Policy Parameters | Inference Latency | Inference FPS | Model Size |
| :--- | :---: | :---: | :---: | :---: |
| **PPO (CNN Policy)** | 1,683,621 | $3.31 \pm 3.21$ ms | 302.2 | **6.42 MB** |
| **SAC (CNN Policy)** | 6,035,944 | $3.39 \pm 2.66$ ms | 295.3 | **23.03 MB** |

---

## References & Citations

If you use this benchmark suite in your research, please cite the underlying theoretical papers:

- **Lyu et al. (2024):** *Understanding What Affects the Generalization Gap in Visual Reinforcement Learning: Theory and Empirical Evidence.* JAIR.
- **Ma et al. (2025):** *A Comprehensive Survey of Data Augmentation in Visual Reinforcement Learning.* IJCV.
- **Wu et al. (2025):** *Reinforcement Learning in Vision: A Survey.* arXiv:2508.08189.
- **Guo et al. (2025):** *Improving Vision-Language-Action Model with Online Reinforcement Learning (iRe-VLA).* ICRA.
- **Lu et al. (2022):** *Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations (V-D4RL).* TMLR.
- **Barrientos Rojas et al. (2024):** *The use of reinforcement learning algorithms in object tracking: A systematic literature review.* Neurocomputing.
