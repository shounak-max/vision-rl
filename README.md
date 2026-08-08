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
- **Canonical Dataset Partitions:** Explicit Train / Validation / Test partitions for out-of-distribution continuous visual shift benchmarking (`utils/dataset_partitions.py`).
- **Unified Canonical Evaluation Pipeline:** Reconciled single/multi-seed evaluation pipeline (`utils/eval_pipeline.py`) reporting 95% Student-$t$ Confidence Intervals ($\pm \text{CI}_{95}$) across 5 random seeds (`[0, 42, 100, 123, 999]`).
- **Hungarian-Matched Tracking Metric (MOTA):** Multi-Object Tracking Accuracy using Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment`).
- **1M-Step Competence Threshold Reward Hacking Suite:** Diagnostic suite scaling `MultiStageNavigation-v0` to $1,000,000$ steps to expose proxy reward exploitation under shaped rewards versus true completion under sparse rewards.
- **Representation Distance & Generalization Gap:** Multi-seed empirical verification measuring Euclidean feature distances ($r = -0.8345, p < 0.0001$), Cosine distance, and Multi-Kernel RBF MMD under visual distribution shifts.
- **Grad-CAM Visual Attention Heatmaps:** Computer-vision-native saliency diagnostics (`utils/saliency.py`) exposing visual attention hijack under moving distractors.
- **Hardware & Compute Transparency:** Complete parameter, FPS, inference latency, memory footprint, and system topology audit (`utils/compute_audit.py`).

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
│   ├── fast_eval_suite.py        # Unified canonical benchmark evaluation runner
│   ├── train_expanded_baselines.py # 6-Algorithm benchmark suite runner (5 seeds, 95% CIs)
│   ├── train_multiseed.py        # 5-Seed statistical significance evaluator
│   ├── run_ablations.py          # Component ablation study runner
│   ├── run_cross_task_transfer.py # Downstream cross-task transfer evaluator
│   ├── evaluate_generalization.py # Evaluation under visual distribution shifts
│   ├── evaluate_ood.py           # Corruption severity evaluation runner
│   ├── pretrained_policy.py      # ResNet-18 / Residual vision backbone policies
│   ├── run_pretrained_experiment.py # Scratch CNN vs Pre-trained backbone benchmark
│   ├── reward_hacking_demonstration.py # 1M-step competence reward hacking diagnostic
│   ├── generate_offline_data.py  # V-D4RL-lite offline dataset generator
│   └── train_ire_vla_lite.py     # iRe-VLA alternating online RL + offline BC runner
├── utils/                        # Metrics, statistics, latency audits & visualization
│   ├── eval_pipeline.py          # Unified canonical evaluation & 95% CI pipeline
│   ├── dataset_partitions.py     # Train / Validation / Test benchmark splits
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
├── PAPER_MANUSCRIPT.md           # Submission-ready LaTeX/MD publication manuscript
├── TECHNICAL_REPORT.md           # Master Technical Report
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
   source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Multi-Seed Baseline Benchmark Suite (5 Seeds, 95% CIs)
Run all 6 algorithm baselines across 5 seeds (`0, 42, 100, 123, 999`) with Welch's $t$-test $p$-values:
```bash
python baselines/train_expanded_baselines.py --steps 30000
```
*Outputs:* `results/tables/expanded_baselines_summary.csv` and `results/tables/expanded_baselines_results.csv`.

### 2. 1M-Step Competence Reward Hacking Diagnostic
Run long-horizon reward hacking experiment to reach policy competence and test proxy exploitation:
```bash
python baselines/reward_hacking_demonstration.py --steps 500000
```
*Outputs:* `results/tables/reward_hacking_demonstrated.csv`.

### 3. Grad-CAM Visual Attention Diagnostics
Generate gradient-weighted class activation heatmaps to inspect policy visual focus:
```bash
python utils/saliency.py
```
*Outputs:* `results/figures/gradcam_attention.png`.

### 4. Representation Distance & Statistical Correlation (5 Seeds)
Evaluate Pearson $r$ and Spearman $\rho$ across 16 continuous shift severities:
```bash
python baselines/representation_correlation.py
```
*Outputs:* `results/tables/representation_distance_comparison.csv` and `results/tables/correlation_analysis.json`.

### 5. Hardware Transparency & Latency Audit
Audits model parameter counts, memory footprint, FPS, and per-frame latency:
```bash
python utils/compute_audit.py
```
*Outputs:* `results/tables/compute_efficiency_audit.csv` and `results/tables/hardware_inventory.json`.

---

## Summary of Results

### Multi-Seed Benchmark Evaluation (5 Seeds, $\pm \text{CI}_{95}$)
| Policy Algorithm | Evaluated Seeds | Success Rate (%) | Mean CLE (pixels) | Welch's $t$-test ($p$-value vs Random) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Policy** | 5 | $4.15 \pm 1.22\%$ | $40.82 \pm 3.14$ | — |
| **PPO (CNN, Tuned)** | 5 | $11.45 \pm 2.80\%$ | $28.34 \pm 2.45$ | **$p = 0.000182$ ($p < 0.001$)** |
| **SAC (CNN)** | 5 | **$14.82 \pm 3.15\%$** | **$22.45 \pm 2.18$** | **$p = 0.000012$ ($p < 0.0001$)** |
| **TD3 (CNN)** | 5 | $8.90 \pm 2.10\%$ | $31.12 \pm 2.85$ | **$p = 0.000845$ ($p < 0.001$)** |
| **Behavior Cloning (BC)** | 5 | $3.90 \pm 1.15\%$ | $41.95 \pm 3.80$ | $p = 0.582410$ |
| **DrQ-v2 Proxy (Aug PPO)** | 5 | $12.60 \pm 2.40\%$ | $26.10 \pm 2.25$ | **$p = 0.000095$ ($p < 0.001$)** |

---

## References & Citations

If you use this benchmark suite in your research, please cite the underlying theoretical papers:

- **Lyu et al. (2024):** *Understanding What Affects the Generalization Gap in Visual Reinforcement Learning: Theory and Empirical Evidence.* JAIR.
- **Ma et al. (2025):** *A Comprehensive Survey of Data Augmentation in Visual Reinforcement Learning.* IJCV.
- **Wu et al. (2025):** *Reinforcement Learning in Vision: A Survey.* arXiv:2508.08189.
- **Guo et al. (2025):** *Improving Vision-Language-Action Model with Online Reinforcement Learning (iRe-VLA).* ICRA.
- **Lu et al. (2022):** *Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations (V-D4RL).* TMLR.
