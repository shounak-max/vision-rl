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

To run the exploratory test-time transform probe and legacy Procgen/scale-up scripts:

```bash
# 1. Exploratory test-time transform sensitivity (not training-time selection):
python baselines/predictive_augmentation_selection.py

# 2. Procgen Smoke Test (300k steps, 50 held-out-level episodes/shift):
python baselines/smoke_procgen.py

# 3. Legacy Scale-Up Benchmark (3 seeds x 2.5M Procgen steps):
python baselines/run_scale_experiment.py
```

---

## 4. Detailed Module-by-Module Execution Guide

### Step 1: Exploratory Test-Time Transform Sensitivity
Applies candidate image transforms to a fixed PPO policy under visual shifts. This is a legacy exploratory sensitivity probe; it does not train policies with different augmentations and therefore does not test training-time augmentation selection. Its feature proxy uses reset observations rather than rollout states. The pooled shift/candidate rows are dependent, so correlations and timing ratios are descriptive only and should not be read as inferential evidence or end-to-end speedups.
```bash
python baselines/predictive_augmentation_selection.py
```
- **Outputs:** `results/tables/predictive_selection.csv` and `results/tables/predictive_selection_summary.json`.
- **Procgen note:** training defaults to levels `[0, 500)` and evaluation uses `[10000, 10500)`; verify the installed Gym/Procgen runtime before using results as evidence.

---

### Step 2: Procgen Procedural Generalization Smoke Test
Trains PPO on `procgen:procgen-coinrun-v0` and evaluates policy performance across 11 perturbation wrappers (Noise, Distractor, Viewpoint, Occlusion, Blur, Compound) on 50 episodes from the held-out evaluation-level range.
```bash
python baselines/smoke_procgen.py
```
- **Outputs:** `results/tables/procgen_smoke_test_results.csv` and `results/tables/procgen_smoke_test_summary.json`.

---

### Step 3: Multi-Seed Scale-Up Experiment (Remote GPU / Cluster)
Trains `PPO_Standard` and `DataAugmented_PPO` across 3 seeds (`[0, 42, 100]`) and 2.5M timesteps per Procgen run. This legacy benchmark is not the preregistered failure-prediction study.
```bash
python baselines/run_scale_experiment.py
# Or launch detached background job on remote GPU server:
python baselines/deploy_remote_gpu.py
```
- **Outputs:** `results/tables/scale_experiment_results.csv` and `results/tables/correlation_procgen_procgen-coinrun-v0.json`.
- **Configuration:** currently 3 seeds (`[0, 42, 100]`) and 2.5M Procgen steps per run.

---

### Study protocol and analysis commands
The full confirmatory study remains gated. Validate the frozen protocol before preparing data:
```bash
pip install -e .
vision-rl-study validate-protocol
vision-rl-study validate-episodes path/to/episodes.csv
vision-rl-study analyze path/to/episodes.csv --output results/tables/study_analysis.json
```
Do not treat legacy benchmark outputs as study results; the confirmatory episode-level table has not been collected.

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
