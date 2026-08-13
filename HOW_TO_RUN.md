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

## 3. Quick Start: Execute Core Research Experiments

To run the predictive augmentation proxy evaluation, Procgen smoke test, and scale-up experiments:

```bash
# 1. Test-Time Augmentation (TTA) Predictive Selection Proxy:
python baselines/predictive_augmentation_selection.py

# 2. Procgen Smoke Test (300k steps, 50 layout-randomized episodes/shift):
python baselines/smoke_procgen.py

# 3. Scale-Up Benchmark (Procgen + Navigation across 5 seeds x 10M timesteps):
python baselines/run_scale_experiment.py
```

---

## 4. Detailed Module-by-Module Execution Guide

### Step 1: Predictive Augmentation Selection via TTA Proxy
Evaluates representation distance ($d_{\text{Euc}}$) as a cheap proxy to rank 13 candidate data augmentation strategies across 10 visual distribution shifts.
```bash
python baselines/predictive_augmentation_selection.py
```
- **Outputs:** `results/tables/predictive_selection.csv` and `results/tables/predictive_selection_summary.json`.

---

### Step 2: Procgen Procedural Generalization Smoke Test
Trains PPO on `procgen:procgen-coinrun-v0` and evaluates policy performance across 11 perturbation wrappers (Noise, Distractor, Viewpoint, Occlusion, Blur, Compound) using 50 layout-randomized seeds.
```bash
python baselines/smoke_procgen.py
```
- **Outputs:** `results/tables/procgen_smoke_test_results.csv` and `results/tables/procgen_smoke_test_summary.json`.

---

### Step 3: Multi-Seed Scale-Up Experiment (Remote GPU / Cluster)
Trains `PPO_Standard` and `DataAugmented_PPO` across 5 random seeds (`[0, 42, 100, 123, 999]`) and 10M timesteps on Procgen. Computes Pearson $r$ and Spearman $\rho$ correlation between representation distance and return degradation.
```bash
python baselines/run_scale_experiment.py
# Or launch detached background job on remote GPU server:
python baselines/deploy_remote_gpu.py
```
- **Outputs:** `results/tables/scale_experiment_results.csv` and `results/tables/correlation_procgen_procgen-coinrun-v0.json`.

---

### Step 4: Hardware Transparency & Latency Audit
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
