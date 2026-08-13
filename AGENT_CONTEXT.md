# Vision RL Benchmark: AI Agent Context & Execution Ledger

> [!IMPORTANT]
> **CRITICAL MANDATE FOR ALL AI AGENTS**:
> 1. **ZERO SYNTHETIC DATA**: Never invent, fabricate, or pre-populate estimated or mock numbers in CSVs, JSONs, markdown files, or LaTeX manuscripts. Every single metric must trace directly to actual PyTorch/Gymnasium code execution.
> 2. **THINK BEFORE CODING**: State assumptions explicitly, surface tradeoffs, don't hide confusion. If anything is unclear, stop and ask.
> 3. **SURGICAL CHANGES**: Write the minimum necessary code. Touch only lines required to solve the task. Do not refactor unbroken adjacent code.
> 4. **GOAL-DRIVEN EXECUTION**: Define verifiable success criteria and loop until verified via actual test output.

---

## 1. Project Mission & Overview

This repository (`vision-rl`) is a high-throughput (**>300 FPS per CPU core**) continuous visual pursuit benchmark suite for visual reinforcement learning (Visual RL). It decouples visual observation shifts ($\xi$) from underlying control dynamics ($P, r$) to evaluate representation learning, target tracking accuracy (Center Location Error $\text{CLE} < 10.0$ px), Hungarian Multi-Object Tracking Accuracy (**MOTA**), visual attention interpretability (**Grad-CAM**), and reward-hacking competence diagnostics.

---

## 2. Theoretical Grounding & Literature Map

- **Lyu et al. (ICML 2024)**: Proves Theorem 4.2 & 4.4 showing that feature representation distance ($d_{\text{rep}} = \|\bar{f}_{\text{src}} - \bar{f}_{\text{tar}}\|_2$) upper-bounds the out-of-distribution generalization gap. Implemented in `baselines/representation_correlation.py` ($d_{\text{Euc}}$ Pearson $r = -0.8099, p < 0.001$).
- **Cherepanov et al. (ICML 2026, KAGE-Bench)**: Known-axis visual observation shifts (`NoiseWrapper`, `DistractorWrapper`, `ViewpointWrapper`, `DataAugmentationWrapper`) holding transition dynamics $P(s' \mid s, a)$ and reward $r(s, a)$ strictly invariant. Implemented in `envs/wrappers.py`.
- **Fujimoto et al. (NeurIPS 2024)**: Multi-seed statistical evaluation protocols using 5 random seeds (`[0, 42, 100, 123, 999]`), Student-$t$ 95% Confidence Intervals ($\pm \text{CI}_{95}$), and Welch's $t$-tests. Implemented in `utils/eval_pipeline.py` & `baselines/train_expanded_baselines.py`.
- **Batra & Sukhatme (May 2025)**: Pre-aligned vision backbones and canonical Train / Validation / Test dataset partitions. Implemented in `utils/dataset_partitions.py` and `baselines/run_pretrained_experiment.py`.
- **Salhab et al. (Neurocomputing 2026)**: Policy competence threshold ($\ge 1\text{M}$ steps / ResNet-18 backbone) for reward-hacking diagnostics. Implemented in `baselines/reward_hacking_demonstration.py`.

---

## 3. Directory & Code Architecture

```
vision-rl/
├── envs/                         # Gymnasium visual environments & wrappers
│   ├── __init__.py               # Gymnasium registration
│   ├── tracking_envs.py          # SingleObjectTracking-v0, MultiObjectTracking-v0, ActiveTracking-v0
│   ├── navigation_envs.py        # MultiStageNavigation-v0 (long-horizon key-door navigation)
│   └── wrappers.py               # DataAugmentation, Noise, Distractor, Viewpoint wrappers
├── baselines/                    # Benchmark execution scripts
│   ├── fast_eval_suite.py        # Entry point for baseline & transfer evaluations
│   ├── train_expanded_baselines.py # 6-Algorithm benchmark runner (5 seeds, 95% CIs)
│   ├── run_pretrained_experiment.py # Scratch NatureCNN vs Pretrained ResNet-18 backbone
│   ├── run_cross_task_transfer.py # Transfer evaluator (SingleObjectTracking -> ActiveTracking)
│   ├── representation_correlation.py # 16-shift representation distance probe evaluator
│   ├── reward_hacking_demonstration.py # 1M-step competence threshold reward exploit proof
│   └── deploy_remote_gpu.py      # Remote GPU deployment script (SSH keepalive + sync)
├── utils/                        # Core utilities & metrics engines
│   ├── eval_pipeline.py          # Canonical evaluation routine (evaluate_policy_canonical)
│   ├── dataset_partitions.py     # Train / Validation / Test dataset splits
│   ├── metrics.py                # CLE, Success Rate, Hungarian-Matched MOTA
│   ├── stats.py                  # Welch's t-test, Student-t 95% CIs
│   ├── saliency.py               # Action-norm Grad-CAM visual attention heatmap generator
│   ├── visualization.py          # Publication plotting engine (t-SNE, grid, degradation)
│   └── compute_audit.py          # Parameter count, FPS, latency ms, memory auditor
├── results/                      # Output artifacts
│   ├── tables/                   # Ground-truth CSV tables & JSON summary files
│   ├── figures/                  # Publication PNG plots
│   └── models/                   # PyTorch / SB3 policy checkpoints (.zip)
├── PAPER_MANUSCRIPT.md           # Submission-ready markdown manuscript
├── PAPER_MANUSCRIPT.tex          # Submission-ready LaTeX manuscript
├── TECHNICAL_REPORT.md           # Master Technical Report
├── RESEARCH_PROGRESS.md          # Research progress ledger & hypothesis log
├── PROJECT_ONBOARDING_CONTEXT.md # Project context & getting started guide
└── AGENT_CONTEXT.md              # AI agent mandate & context file
```

---

## 4. Ground-Truth Empirical Benchmark Data (Locked Reference)

> [!WARNING]
> This table is **STALE** and currently contradicts `CONTEXT.md`. It references files that no longer exist (e.g., `representation_correlation.py`) and reports numbers that have not been re-run since the recent repository pivot. **Do not treat any number in this table as real.**

All manuscript text, tables, and AI agent outputs must strictly match these ground-truth numbers:

| Experiment / Metric | Ground-Truth Empirical Value | Verification Source File |
| :--- | :--- | :--- |
| **Random Policy Baseline** | CLE: $40.43 \pm 4.28$ px, Success: $3.08 \pm 1.32\%$ | `results/tables/expanded_baselines_summary.csv` |
| **PPO (NatureCNN Scratch)** | CLE: $61.09 \pm 2.22$ px, Success: $0.22 \pm 0.24\%$ ($p=0.000736$) | `results/tables/expanded_baselines_summary.csv` |
| **SAC (NatureCNN Scratch)** | CLE: **$25.85 \pm 3.90$ px**, Success: **$9.38 \pm 2.95\%$** ($p=0.000053$) | `results/tables/expanded_baselines_summary.csv` |
| **TD3 (NatureCNN Scratch)** | CLE: $59.27 \pm 2.32$ px, Success: $0.46 \pm 0.24\%$ ($p=0.000006$) | `results/tables/expanded_baselines_summary.csv` |
| **Behavior Cloning (BC)** | CLE: $42.26 \pm 4.53$ px, Success: $3.90 \pm 1.26\%$ ($p=0.457729$) | `results/tables/expanded_baselines_summary.csv` |
| **DrQ-v2 Proxy (Aug PPO)** | CLE: $55.76 \pm 3.87$ px, Success: $1.11 \pm 0.40\%$ ($p=0.000044$) | `results/tables/expanded_baselines_summary.csv` |
| **Pretrained ResNet-18 Backbone** | CLE: **$47.04 \pm 5.53$ px**, Success: **$2.47 \pm 0.93\%$** ($11.2\times$ gain) | `results/tables/pretrained_vs_scratch_results.csv` |
| **Cross-Task Transfer (Scratch)** | CLE: $29.31 \pm 13.61$ px, Success: $37.25 \pm 14.18\%$ | `results/tables/cross_task_transfer_results.csv` |
| **Cross-Task Transfer (Fine-Tuned)**| CLE: $63.26 \pm 26.04$ px, Success: $16.40 \pm 12.87\%$ | `results/tables/cross_task_transfer_results.csv` |
| **Euclidean Distance Correlation**| Pearson **$r = -0.8099, p < 0.001$**; Spearman **$\rho = -0.7441$** | `results/tables/correlation_analysis.json` |
| **1M-Step Reward Hacking** | Shaped Reward: **$168.4 \pm 7.2$ hover steps** (Sparse: $88.0\%$ succ) | `results/tables/reward_hacking_demonstrated.csv` |

---

## 5. Execution Commands for AI Agents

When executing or verifying code changes in this workspace, use PowerShell and always activate the virtual environment first:

```powershell
# 1. Run codebase regression suite
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "cd 'd:\gitfork\vision rl'; .\venv\Scripts\Activate.ps1; python scratch\verify_all.py"

# 2. Run utilities & visualization suite
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "cd 'd:\gitfork\vision rl'; .\venv\Scripts\Activate.ps1; python scratch\verify_utils.py"

# 3. Execute scale experiment (new primary runner)
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "cd 'd:\gitfork\vision rl'; .\venv\Scripts\Activate.ps1; python baselines\run_scale_experiment.py"

# 4. Deploy to remote GPU cluster
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -Command "cd 'd:\gitfork\vision rl'; .\venv\Scripts\Activate.ps1; python baselines\deploy_remote_gpu.py"
```
