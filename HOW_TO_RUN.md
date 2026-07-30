# How to Run the Vision RL Benchmark Suite
**Complete Step-by-Step Execution & Reproduction Guide**

This guide provides clear instructions to set up, run, evaluate, and reproduce all experiments, benchmarks, statistical significance tests, and figure generation routines in this repository.

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

### Option B: Using Conda
```bash
conda create -n vision_rl python=3.10 -y
conda activate vision_rl
pip install -r requirements.txt
```

---

## 3. Quick Start: Execute All Experiments (One-Liner)

To run the complete benchmark suite, statistical tests, and figure visualizers sequentially:

```bash
python baselines/train_multiseed.py --steps 30000 && \
python baselines/run_pretrained_experiment.py --steps 20000 && \
python baselines/run_ablations.py --steps 25000 && \
python baselines/representation_correlation.py && \
python baselines/reward_hacking_demonstration.py --steps 50000 && \
python utils/saliency.py && \
python utils/visualization.py && \
python utils/compute_audit.py
```

---

## 4. Detailed Module-by-Module Execution Guide

### Step 1: Multi-Seed Statistical Evaluation (5 Seeds)
Trains PPO across 5 distinct random seeds (`[0, 42, 100, 123, 999]`), evaluates against a Random baseline, and computes Welch's $t$-test $p$-values and 95% Confidence Intervals.
```bash
python baselines/train_multiseed.py --env SingleObjectTracking-v0 --steps 30000
```
- **Outputs:** `results/tables/multiseed_results.csv` and `results/tables/multiseed_summary.json`.

---

### Step 2: Visual Representation Backbone Comparison
Compares a 3-layer NatureCNN trained from scratch against a Deep Residual Vision Backbone (`PretrainedVisionFeatureExtractor`) to isolate visual representation bottlenecks.
```bash
python baselines/run_pretrained_experiment.py --steps 20000
```
- **Outputs:** `results/tables/pretrained_vs_scratch_results.csv`.

---

### Step 3: 16-Level Representation Distance Correlation Analysis
Evaluates the trained policy across 16 continuous shift severities ($\sigma \in [0.0, 0.4]$, $N \in [0, 4]$, $\theta \in [0^\circ, 45^\circ]$) and computes **Pearson correlation ($r$)**, **Spearman rank correlation ($\rho$)**, Cosine distance, and RBF MMD distance.
```bash
python baselines/representation_correlation.py --model "results/models/PPO_SingleObjectTracking-v0_s42.zip"
```
- **Outputs:** `results/tables/representation_distance_comparison.csv` and `results/tables/correlation_analysis.json`.

---

### Step 4: Active Reward Hacking Demonstration
Pairs the pre-trained residual backbone with `MultiStageNavigation-v0` to demonstrate active shaped reward proxy exploitation (hovering at $5.1$ px distance to harvest reward without terminating episodes).
```bash
python baselines/reward_hacking_demonstration.py --steps 50000
```
- **Outputs:** `results/tables/reward_hacking_demonstrated.csv`.

---

### Step 5: Component Ablation Study
Evaluates policy robustness with vs. without `DataAugmentationWrapper` (spatial shift and color jitter) under out-of-distribution Gaussian noise.
```bash
python baselines/run_ablations.py --steps 25000
```
- **Outputs:** `results/tables/ablation_results.csv`.

---

### Step 6: Out-of-Distribution Corruption Evaluation
Evaluates policy performance across noise, distractor, and viewpoint severity spectrums.
```bash
python baselines/evaluate_ood.py --model "results/models/PPO_SingleObjectTracking-v0_s42.zip"
```
- **Outputs:** `results/tables/ood_corruption_results.csv`.

---

### Step 7: Grad-CAM Visual Attention Heatmap Extraction
Extracts gradient-weighted class activation heatmaps from the policy CNN to inspect visual attention focus under clean vs. distractor conditions.
```bash
python utils/saliency.py
```
- **Outputs:** `results/figures/gradcam_attention.png`.

---

### Step 8: Generate All Publication Figures
Generates environment grids, t-SNE / PCA feature cluster plots, corruption degradation curves, and failure case matrices.
```bash
python utils/visualization.py
```
- **Outputs:** `results/figures/env_grid.png`, `results/figures/tsne_features.png`, `results/figures/degradation_curves.png`, `results/figures/failure_cases.png`.

---

### Step 9: Compute & Inference Latency Audit
Audits model parameter count, memory footprint (MB), FPS during training/inference, and per-frame latency (ms).
```bash
python utils/compute_audit.py
```
- **Outputs:** `results/tables/compute_efficiency_audit.csv`.

---

### Step 10: Generate Offline Dataset & Run iRe-VLA Hybrid Training
Exports a 5,000-transition V-D4RL proxy dataset (`.npz`) and runs alternating online PPO + offline Behavior Cloning (BC) regularization.
```bash
python baselines/generate_offline_data.py --steps 5000
python baselines/train_ire_vla_lite.py --iters 5
```
- **Outputs:** `results/datasets/offline_dataset.npz`.

---

## 5. Running on Remote Campus GPU Server

To deploy and execute training on a remote GPU server (e.g., 4$\times$ Tesla K80 GPUs via SSH):

```bash
python baselines/deploy_remote_gpu.py
```
This script automatically connects over Paramiko SSH, synchronizes the code, activates the remote Conda environment, installs dependencies, and launches PyTorch CUDA GPU training.

---

## 6. Output Files & Results Inventory

After running the complete suite, your output files will be structured as follows:

```
results/
├── tables/
│   ├── multiseed_results.csv                    # 5-seed metrics
│   ├── multiseed_summary.json                   # Welch's t-test p-values & CIs
│   ├── pretrained_vs_scratch_results.csv        # ResNet-18 vs Scratch CNN benchmark
│   ├── representation_distance_comparison.csv   # 16-level d_Euc, d_Cos, MMD distances
│   ├── correlation_analysis.json                # Pearson r & Spearman rho values
│   ├── reward_hacking_demonstrated.csv          # Active reward exploit proof
│   ├── ablation_results.csv                     # Data augmentation ablation table
│   ├── ood_corruption_results.csv               # Severity spectrum metrics
│   └── compute_efficiency_audit.csv             # Parameters, latency ms, FPS audit
├── figures/
│   ├── env_grid.png                             # 4-panel visual environment grid
│   ├── gradcam_attention.png                    # Grad-CAM visual saliency heatmaps
│   ├── tsne_features.png                        # PCA feature space embeddings
│   ├── degradation_curves.png                   # Performance degradation curves
│   ├── correlation_curves.png                   # Pearson/Spearman regression plots
│   ├── reward_hacking_exploit.png               # Proxy return vs task success plot
│   └── failure_cases.png                        # Qualitative failure matrix
└── datasets/
    └── offline_dataset.npz                      # V-D4RL proxy visual dataset
```

---

## 7. Troubleshooting

- **ModuleNotFoundError (`No module named 'envs'`):**  
  Set the Python path environment variable before running scripts:
  - **PowerShell:** `$env:PYTHONPATH="."`
  - **Linux/macOS:** `export PYTHONPATH="."`
- **MemoryError in SAC:**  
  Set `--buffer_size 10000` when instantiating SAC for visual inputs on CPU.
