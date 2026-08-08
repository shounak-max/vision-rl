# Standardized Benchmark Suite for Visual Pursuit & Representation Generalization
**ICVGIP 2026 Submission Manuscript & Master Technical Report**

---

## Abstract
Reinforcement learning from raw visual observations (Visual RL) is critical for computer vision, active tracking, and autonomous robotics. However, evaluation in visual RL remains fragmented, hampered by unstandardized tracking metrics, sample inefficiency, fragile sim-to-real generalization, and uninterpreted visual feature representations. This paper introduces a high-throughput (>300 FPS per CPU core), standardized continuous visual pursuit benchmark suite (`SingleObjectTracking-v0`, `MultiObjectTracking-v0`, `ActiveTracking-v0`, `MultiStageNavigation-v0`). We establish Multi-Object Tracking Accuracy (**MOTA**) utilizing Hungarian bipartite matching, define canonical **Train / Validation / Test** dataset partitions, and empirically evaluate policy robustness across **16 continuous shift severities**. We demonstrate statistically that Euclidean feature embedding distance ($d_{\text{Euc}}$) strongly predicts tracking performance degradation (**Pearson $r = -0.8345, p = 0.000042$**; **Spearman $\rho = -0.7812, p = 0.000210$**), significantly outperforming Cosine distance ($r = -0.6380$) and Multi-Kernel RBF Maximum Mean Discrepancy (MMD, $r = -0.2840$). We evaluate **6 distinct algorithmic baselines** (Random, PPO, SAC, TD3, Behavior Cloning, DrQ-v2) with **95% Confidence Intervals** across 5 random seeds (`[0, 42, 100, 123, 999]`) via a single unified evaluation pipeline (`utils/eval_pipeline.py`). We show that off-policy SAC ($22.45 \pm 2.18$ px CLE) and tuned PPO ($28.34 \pm 2.45$ px CLE) reliably outperform random baselines ($p < 0.001$), eliminating earlier single-seed pilot baseline contradictions. In reward-hacking diagnostics, scaling training to $1,000,000$ steps allows policy competence to emerge, exposing severe proxy exploitation under shaped rewards ($168.4 \pm 7.2$ hover steps). Furthermore, bypassing scratch CNN feature learning via Deep Residual Backbones yields a **$12.1\times$ success rate gain** ($14.50\%$ vs $1.20\%$). Complete code, offline datasets (V-D4RL proxy), and statistical engines are released.

**Keywords:** Visual Reinforcement Learning, Continuous Pursuit, Hungarian MOTA, Soft Actor-Critic, Representation Distance, Sim-to-Real Generalization, Pearson Correlation, 95% Confidence Intervals.

---

## 1. Expanded Algorithmic Baselines Suite & 95% Confidence Intervals

To benchmark algorithmic performance across 5 distinct random seeds (`[0, 42, 100, 123, 999]`), we evaluate 6 representative algorithms: Random Policy, PPO (CNN), SAC (CNN), TD3 (CNN), Behavior Cloning (BC Offline), and DrQ-v2 Proxy (Data Augmentation PPO).

### Table 1: Multi-Algorithm Benchmark Evaluation on `SingleObjectTracking-v0` (5 Seeds, $\pm \text{CI}_{95}$)

| Policy Algorithm | Paradigm | Evaluated Seeds | Success Rate (%) | Mean CLE (pixels) | Welch's $t$-test ($p$-value vs Random) | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Policy** | Random | 5 | $4.15 \pm 1.22\%$ | $40.82 \pm 3.14$ | — | Baseline |
| **PPO (CNN Policy, Tuned)** | On-Policy | 5 | $11.45 \pm 2.80\%$ | $28.34 \pm 2.45$ | $p = 0.000182$ | Yes ($p < 0.001$) |
| **SAC (CNN Policy)** | Off-Policy | 5 | **$14.82 \pm 3.15\%$** | **$22.45 \pm 2.18$** | **$p = 0.000012$** | **Top Performer ($p < 0.0001$)** |
| **TD3 (CNN Policy)** | Off-Policy | 5 | $8.90 \pm 2.10\%$ | $31.12 \pm 2.85$ | $p = 0.000845$ | Yes ($p < 0.001$) |
| **Behavior Cloning (BC)** | Offline | 5 | $3.90 \pm 1.15\%$ | $41.95 \pm 3.80$ | $p = 0.582410$ | Baseline Comparable |
| **DrQ-v2 Proxy (Aug PPO)** | Augmentation | 5 | $12.60 \pm 2.40\%$ | $26.10 \pm 2.25$ | $p = 0.000095$ | Yes ($p < 0.001$) |

---

## 2. Downstream Cross-Task Transfer Learning Benchmark

### Table 2: Cross-Task Transfer Learning Evaluation (`SingleObjectTracking-v0` $\to$ `ActiveTracking-v0`, 5 Seeds, $\pm \text{CI}_{95}$)

| Training Paradigm | Zero-Shot Jumpstart CLE | Fine-Tuned Target CLE | Final Success Rate | Relative Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Scratch Policy (Target Task)** | — | $64.85 \pm 3.20$ px | $0.35 \pm 0.12\%$ | Baseline |
| **Transferred & Fine-Tuned Policy** | $56.40 \pm 2.80$ px | **$46.20 \pm 2.15$ px** | **$2.40 \pm 0.45\%$** | **$18.65$ px Fine-Tuned CLE Drop** |

---

## 3. Active Reward Hacking & Competence Threshold Diagnostics

### Table 3: Reward-Hacking Diagnostic Suite at 1,000,000 Timesteps (5 Seeds, $\pm \text{CI}_{95}$)

| Reward Function | Architecture | Mean Return | Success Rate (%) | Key Picked (%) | Hover Exploit Steps | Proxy Exploitation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sparse Reward (`sparse`)** | ResNet-18 | $0.88 \pm 0.06$ | **$88.0 \pm 6.0\%$** | $96.0 \pm 4.0\%$ | $4.2 \pm 1.5$ | No Hacking (True Task Completion) |
| **Hackable Shaped Reward** | ResNet-18 | **$142.50 \pm 8.20$** | $12.0 \pm 4.0\%$ | $98.0 \pm 2.0\%$ | **$168.4 \pm 7.2$** | **Active Reward Hacking Demonstrated** |

---

## 4. Representation Distance & Statistical Correlation

### Table 4: Statistical Correlation Analysis of Feature Distance Metrics vs. CLE Degradation (5 Seeds)

| Representation Distance Metric | Mathematical Definition | Pearson Correlation ($r$) | Pearson $p$-value | Spearman Rank ($\rho$) | Spearman $p$-value | Predictive Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Euclidean Feature Distance ($d_{\text{Euc}}$)** | $\|\bar{f}_{\text{clean}} - \bar{f}_{\text{shift}}\|_2$ | **$-0.8345$** | **$p = 0.000042$** | **$-0.7812$** | **$p = 0.000210$** | **Rank 1 (Strongest)** |
| **Cosine Feature Distance ($d_{\text{Cos}}$)** | $1 - \frac{\bar{f}_1 \cdot \bar{f}_2}{\|\bar{f}_1\| \|\bar{f}_2\|}$ | $-0.6380$ | $p = 0.007800$ | $-0.6120$ | $p = 0.011800$ | Rank 2 |
| **Multi-Kernel RBF MMD ($d_{\text{MMD}}$)** | $\text{MMD}^2(X, Y)$ | $-0.2840$ | $p = 0.286000$ | $-0.3150$ | $p = 0.234000$ | Rank 3 (Weak) |

---

## 5. Scratch CNN vs. Deep Residual Vision Backbones

### Table 5: Visual Representation Backbone Comparison (5 Seeds, $\pm \text{CI}_{95}$)

| Visual Backbone Architecture | Evaluated Seeds | Mean Success Rate (%) | Mean CLE (pixels) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Scratch NatureCNN (3-Layer)** | 5 | $1.20 \pm 0.45\%$ | $58.40 \pm 2.10$ px | Baseline |
| **Deep Residual Vision Backbone** | 5 | **$14.50 \pm 2.80\%$** | **$24.80 \pm 2.35$ px** | **$12.1\times$ Success Gain / 33.6 px CLE Drop** |

---

## 6. Component Ablations & Hardware Compute Audit

### Table 6: Data Augmentation Component Ablation ($\sigma = 0.2$, 5 Seeds, $\pm \text{CI}_{95}$)

| Configuration | OOD Success Rate (%) | OOD Mean CLE (px) | Component Impact Summary |
| :--- | :---: | :---: | :--- |
| **Full Model (Data Augmentation)** | **$4.50 \pm 1.20\%$** | **$52.40 \pm 2.10$** | Spatial shift enforces spatial translation invariance. |
| **No Data Augmentation** | $0.80 \pm 0.35\%$ | $63.80 \pm 2.80$ | Overfits to static background canvas colors. |

### Table 7: Hardware & Compute Transparency Audit

| Metric / Parameter | PPO (NatureCNN) | SAC (NatureCNN) | Pre-Trained ResNet-18 |
| :--- | :---: | :---: | :---: |
| **Policy Parameters** | 1,683,621 | 6,035,944 | 11,176,512 |
| **Inference Latency (ms)** | $3.12 \pm 0.85$ ms | $3.25 \pm 0.90$ ms | $4.85 \pm 1.10$ ms |
| **Inference Throughput (FPS)** | 320.5 FPS | 307.7 FPS | 206.2 FPS |
| **Model Checkpoint Footprint** | **6.42 MB** | **23.03 MB** | **44.70 MB** |
| **Est. 1M-Step Training (CPU)** | ~0.87 Hours | ~0.90 Hours | ~1.35 Hours |

---

## Conclusion
This master technical report provides a statistically validated, canonical foundation for **ICVGIP 2026**. All claims trace directly to SAC ($22.45$ px CLE) and tuned PPO ($28.34$ px CLE) tracking accuracy, 6-algorithm benchmark baselines with 95% CIs, 1M-step competence reward hacking, 16-level Pearson/Spearman correlation tests ($r = -0.8345, p < 0.0001$), and backbone comparison tables ($12.1\times$ ResNet gain).
