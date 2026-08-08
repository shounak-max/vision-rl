# How to Run the Vision RL Benchmark Suite
**Complete Step-by-Step Execution & Reproduction Guide**

This guide provides clear instructions to set up, run, evaluate, and reproduce all experiments, benchmarks, statistical significance tests, dataset partitions, and figure generation routines in this repository.

---

## 1. System Requirements & Dependencies

- **Operating System:** Linux, macOS, or Windows 10/11
- **Python Version:** Python 3.10 or higher
- **Hardware:** 
  - **Local Execution:** CPU (Multi-core recommended, runs at >300 FPS per core)
  - **GPU Execution (Optional):** NVIDIA GPU with CUDA 11+ support

---

## 2. Environment Setup

### Option A: Using Standard Python Virtual Environment (`venv`)
```bash
# 1. Clone or navigate to the repository directory
cd "d:/gitfork/vision rl"

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# 4. Install required dependencies
pip install -r requirements.txt
```

---

## 3. Quick Start: Execute All Benchmarks (One-Liner)

To run the complete canonical benchmark suite, statistical tests, and figure visualizers sequentially:

```bash
python baselines/fast_eval_suite.py && \
python baselines/run_pretrained_experiment.py --steps 30000 && \
python baselines/run_cross_task_transfer.py --steps 30000 && \
python baselines/run_ablations.py --steps 30000 && \
python baselines/representation_correlation.py && \
python baselines/reward_hacking_demonstration.py --steps 500000 && \
python utils/saliency.py && \
python utils/visualization.py && \
python utils/compute_audit.py
```

---

## 4. Detailed Module-by-Module Execution Guide

### Step 1: Multi-Seed Baseline Benchmark Evaluation (5 Seeds, 95% CIs)
Evaluates 6 algorithmic baselines (Random, PPO, SAC, TD3, BC, DrQ-v2) across 5 distinct random seeds (`[0, 42, 100, 123, 999]`), reporting 95% Student-$t$ CIs and Welch's $t$-test $p$-values.
```bash
python baselines/train_expanded_baselines.py --steps 30000
```
- **Outputs:** `results/tables/expanded_baselines_summary.csv` and `results/tables/expanded_baselines_results.csv`.

---

### Step 2: Visual Representation Backbone Comparison
Compares a 3-layer NatureCNN trained from scratch against a Deep Residual Vision Backbone (`PretrainedVisionFeatureExtractor`) across 5 seeds.
```bash
python baselines/run_pretrained_experiment.py --steps 30000
```
- **Outputs:** `results/tables/pretrained_vs_scratch_results.csv`.

---

### Step 3: 16-Level Representation Distance Correlation Analysis
Evaluates policy checkpoints across 16 continuous shift severities ($\sigma \in [0.0, 0.4]$, $N \in [0, 4]$, $\theta \in [0^\circ, 45^\circ]$) under the canonical Test partition. Computes **Pearson correlation ($r$)**, **Spearman rank correlation ($\rho$)**, Cosine distance, and Multi-Kernel RBF MMD distance across 5 seeds.
```bash
python baselines/representation_correlation.py
```
- **Outputs:** `results/tables/representation_distance_comparison.csv` and `results/tables/correlation_analysis.json`.

---

### Step 4: Active Reward Hacking Competence Diagnostic (1M Steps)
Pairs the pre-trained residual backbone with `MultiStageNavigation-v0` up to $1,000,000$ steps to reach a policy competence threshold, demonstrating hovering proxy exploitation under shaped rewards versus true completion under sparse rewards.
```bash
python baselines/reward_hacking_demonstration.py --steps 500000
```
- **Outputs:** `results/tables/reward_hacking_demonstrated.csv`.

---

### Step 5: Downstream Cross-Task Transfer Learning
Evaluates feature transfer from `SingleObjectTracking-v0` to `ActiveTracking-v0` across 5 seeds.
```bash
python baselines/run_cross_task_transfer.py --steps 30000
```
- **Outputs:** `results/tables/cross_task_transfer_results.csv`.

---

### Step 6: Component Ablation Study
Evaluates policy robustness with vs. without `DataAugmentationWrapper` under out-of-distribution Gaussian noise across 5 seeds.
```bash
python baselines/run_ablations.py --steps 30000
```
- **Outputs:** `results/tables/ablation_results.csv`.

---

### Step 7: Grad-CAM Visual Attention Heatmap Extraction
Extracts gradient-weighted class activation heatmaps from policy CNNs to inspect visual attention focus under clean vs. distractor conditions.
```bash
python utils/saliency.py
```
- **Outputs:** `results/figures/gradcam_attention.png`.

---

### Step 8: Hardware Transparency & Latency Audit
Audits model parameter count, memory footprint (MB), FPS during training/inference, and per-frame latency (ms).
```bash
python utils/compute_audit.py
```
- **Outputs:** `results/tables/compute_efficiency_audit.csv` and `results/tables/hardware_inventory.json`.

---

## 5. Output Files Inventory

```
results/
├── tables/
│   ├── expanded_baselines_summary.csv           # 6-baseline 5-seed summary & 95% CIs
│   ├── expanded_baselines_results.csv           # Per-seed baseline raw metrics
│   ├── multiseed_results.csv                    # PPO multi-seed results
│   ├── multiseed_summary.json                   # Welch's t-test p-values & CIs
│   ├── pretrained_vs_scratch_results.csv        # ResNet-18 vs Scratch CNN benchmark
│   ├── cross_task_transfer_results.csv          # Downstream transfer jumpstart/fine-tune
│   ├── representation_distance_comparison.csv   # 16-level d_Euc, d_Cos, MMD distances
│   ├── correlation_analysis.json                # Pearson r & Spearman rho values
│   ├── reward_hacking_demonstrated.csv          # 1M-step competence reward exploit proof
│   ├── ablation_results.csv                     # Data augmentation ablation table
│   ├── compute_efficiency_audit.csv             # Parameters, latency ms, FPS audit
│   └── hardware_inventory.json                  # System hardware topology inventory
└── figures/
    ├── env_grid.png                             # 4-panel visual environment grid
    ├── gradcam_attention.png                    # Grad-CAM visual saliency heatmaps
    ├── tsne_features.png                        # PCA feature space embeddings
    ├── degradation_curves.png                   # Performance degradation curves
    └── correlation_curves.png                   # Pearson/Spearman regression plots
```
