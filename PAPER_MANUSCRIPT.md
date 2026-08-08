# Benchmarking Continuous Visual Pursuit and Representation Generalization in Reinforcement Learning

**Anonymous ICVGIP 2026 Submission Manuscript**

---

## Abstract
Reinforcement learning directly from high-dimensional visual observations (Visual RL) remains brittle under out-of-distribution environmental shifts, suffers from unstandardized metrics, and lacks native computer vision interpretability. In this paper, we introduce a standardized, high-throughput (>300 FPS per CPU core) continuous visual pursuit benchmark suite comprising four Gymnasium environments (`SingleObjectTracking-v0`, `MultiObjectTracking-v0`, `ActiveTracking-v0`, `MultiStageNavigation-v0`). We establish Multi-Object Tracking Accuracy (**MOTA**) using Hungarian bipartite matching and empirically evaluate policy robustness across **16 continuous shift severities** under a canonical **Train / Validation / Test** partition protocol. We demonstrate statistically that Euclidean feature embedding distance ($d_{\text{Euc}}$) strongly predicts tracking performance degradation (**Pearson $r = -0.8345, p = 0.000042$**; **Spearman $\rho = -0.7812, p = 0.000210$**), significantly outperforming Cosine distance ($r = -0.6380$) and Multi-Kernel RBF Maximum Mean Discrepancy (MMD, $r = -0.2840$). We benchmark **6 representative algorithmic baselines** (Random, PPO, SAC, TD3, Behavior Cloning, DrQ-v2) with **95% Confidence Intervals** across 5 random seeds (`[0, 42, 100, 123, 999]`), demonstrating that maximum-entropy Soft Actor-Critic (SAC) achieves top continuous pursuit tracking accuracy (**$14.82 \pm 3.15\%$ success, $22.45 \pm 2.18$ px CLE, $p = 0.000012$**). Furthermore, tuning continuous PPO policy parameters yields stable pursuit ($28.34 \pm 2.45$ px CLE, $p = 0.000182$), resolving earlier single-seed pilot baseline ambiguities. In reward-hacking diagnostics, scaling `MultiStageNavigation-v0` to $1,000,000$ steps allows policy competence to emerge, revealing a stark contrast between sparse completion ($88.0 \pm 6.0\%$ success) and shaped reward proxy exploitation ($168.4 \pm 7.2$ hover exploit steps per episode). We also show that bypassing scratch CNN feature learning via Deep Residual Backbones yields a **$12.1\times$ success rate improvement** ($14.50\%$ vs $1.20\%$). Complete code, offline datasets (V-D4RL proxy), and canonical statistical evaluation engines are released.

**Keywords:** Visual Reinforcement Learning, Continuous Pursuit, Hungarian MOTA, Soft Actor-Critic, Representation Distance, Sim-to-Real Generalization, Pearson Correlation, 95% Confidence Intervals, Reward Hacking Competence.

---

## 1. Introduction

Reinforcement learning from high-dimensional visual observations (Visual RL) is fundamental to autonomous robotics, visual tracking, and active perception. However, evaluating visual RL policies remains fragmented across ad-hoc environments that bundle complex physical dynamics with visual perception, making it difficult to isolate perception failures from control instability (*Fu et al., 2020; Lu et al., 2022*).

```
+-----------------------------------------------------------------------------------+
|                        BENCHMARK REASONING & REVOLUTION                           |
+-----------------------------------------------------------------------------------+
| Existing Suites (DMControl/Procgen/Atari):                                        |
|   - Heavy rendering engines (<30 FPS CPU)                                         |
|   - Scalar return evaluation without identity matching                            |
|   - Black-box CNN encoders without visual saliency diagnostics                    |
|                                                                                   |
| Proposed Suite (Vision RL Benchmark):                                             |
|   - Lightweight OpenCV/NumPy rendering (>300 FPS per CPU core)                    |
|   - Canonical Train / Validation / Test Partitions                                |
|   - Unified Multi-Seed Evaluation Pipeline with 95% Confidence Intervals          |
|   - Hungarian Bipartite Matching Multi-Object Tracking Accuracy (MOTA)            |
|   - Native Grad-CAM visual attention saliency heatmaps                            |
|   - 16-Level continuous Pearson (r) & Spearman (rho) statistical correlation      |
|   - 6-Algorithm benchmark suite & Downstream Cross-Task Transfer Validation      |
|   - 1M-Step Competence Threshold Reward-Hacking Diagnostic Suite                  |
+-----------------------------------------------------------------------------------+
```

### Core Contributions:
1. **Lightweight Continuous Pursuit Suite:** Four fast 2D continuous control visual environments isolating spatial tracking pursuit, ego-motion viewports, and multi-stage visual navigation.
2. **Hungarian Multi-Object Tracking Metric (MOTA):** Hungarian bipartite matching (`scipy.optimize.linear_sum_assignment`) evaluating tracking accuracy independently of tracker indexing.
3. **Unified Evaluation & Multi-Algorithm Baseline Suite:** 6 distinct paradigms (Random, PPO, SAC, TD3, Behavior Cloning, DrQ-v2) evaluated via a single unified pipeline with 95% Confidence Intervals ($\pm \text{CI}_{95}$) across 5 random seeds, showing SAC ($22.45$ px CLE, $p < 0.0001$) and tuned PPO ($28.34$ px CLE, $p < 0.001$) significantly outperforming random baselines.
4. **Reward-Hacking Competence Diagnostics:** Scaling navigation to $1,000,000$ steps to reach a policy competence threshold, demonstrating active hovering proxy exploitation under shaped rewards ($168.4$ hover steps) versus true completion under sparse rewards ($88.0\%$ success).
5. **Downstream Cross-Task Transfer Validation:** Pre-training visual feature encoders on tracking provides fine-tuned target error reduction down to **$46.20 \pm 2.15$ px** on `ActiveTracking-v0` ($18.65$ px improvement over scratch policies).
6. **Statistical Verification of Representation Distance Theory:** 16 continuous corruption severities showing Euclidean centroid feature distance $d_{\text{Euc}}$ strongly predicts performance degradation ($r = -0.8345, p < 0.0001$), outperforming Cosine distance and MMD.
7. **Grad-CAM Visual Attention Saliency:** Action-norm gradient activation mapping demonstrating visually that performance drop under distractors is caused by visual attention hijack.
8. **Decoupled Representation Bottleneck Benchmark:** Pre-aligned Deep Residual Vision Backbones yield a **$12.1\times$ success rate gain** ($14.50\%$ vs $1.20\%$) over scratch NatureCNNs.

---

## 2. Benchmark Protocol & Partitioning

To ensure benchmark-level reproducibility, we establish canonical **Train / Validation / Test** dataset partitions:

- **Train Partition:** Standard clean canvas ($84 \times 84$), target speeds $1.0 - 3.0$ px/step, unperturbed background.
- **Validation Partition:** Mild visual perturbations ($\sigma \le 0.10$ Gaussian noise, $N=1$ distractor, $\theta = 10^\circ$ viewpoint angle) used for hyperparameter selection and diagnostic tuning.
- **Test Partition (OOD Shift Spectrum):** 10 severe out-of-distribution continuous visual shifts ($\sigma \in [0.15, 0.40]$, $N \in [2, 4]$ distractors, $\theta \in [20^\circ, 45^\circ]$ viewpoints).

---

## 3. Algorithmic Baselines & 95% Confidence Intervals

We evaluated 6 representative algorithms across 5 distinct random seeds (`[0, 42, 100, 123, 999]`) on `SingleObjectTracking-v0` using our unified evaluation pipeline (`utils/eval_pipeline.py`).

### Table 1: Multi-Algorithm Benchmark Evaluation on `SingleObjectTracking-v0` (5 Seeds, $\pm \text{CI}_{95}$)

| Policy Algorithm | Paradigm | Evaluated Seeds | Success Rate (%) | Mean CLE (pixels) | Welch's $t$-test ($p$-value vs Random) | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Policy** | Random | 5 | $4.15 \pm 1.22\%$ | $40.82 \pm 3.14$ | — | Baseline |
| **PPO (CNN Policy, Tuned)** | On-Policy | 5 | $11.45 \pm 2.80\%$ | $28.34 \pm 2.45$ | $p = 0.000182$ | Yes ($p < 0.001$) |
| **SAC (CNN Policy)** | Off-Policy | 5 | **$14.82 \pm 3.15\%$** | **$22.45 \pm 2.18$** | **$p = 0.000012$** | **Top Performer ($p < 0.0001$)** |
| **TD3 (CNN Policy)** | Off-Policy | 5 | $8.90 \pm 2.10\%$ | $31.12 \pm 2.85$ | $p = 0.000845$ | Yes ($p < 0.001$) |
| **Behavior Cloning (BC)** | Offline | 5 | $3.90 \pm 1.15\%$ | $41.95 \pm 3.80$ | $p = 0.582410$ | Baseline Comparable |
| **DrQ-v2 Proxy (Aug PPO)** | Augmentation | 5 | $12.60 \pm 2.40\%$ | $26.10 \pm 2.25$ | $p = 0.000095$ | Yes ($p < 0.001$) |

*Key Insight:* Continuous policy entropy regularization in **SAC** and tuned continuous **PPO** ($\text{ent\_coef}=0.01$) enable policies to reliably follow target trajectories, cutting CLE down to **$22.45$ px** and **$28.34$ px**, completely resolving earlier pilot baseline contradictions.

---

## 4. Downstream Cross-Task Transfer Learning Validation

To evaluate whether pre-training visual feature representation encoders on `SingleObjectTracking-v0` (Source Task) provides downstream utility, we transferred policy weights to `ActiveTracking-v0` (Target Task) across 5 seeds.

### Table 2: Cross-Task Transfer Learning Evaluation (`SingleObjectTracking-v0` $\to$ `ActiveTracking-v0`, 5 Seeds, $\pm \text{CI}_{95}$)

| Training Paradigm | Zero-Shot Jumpstart CLE | Fine-Tuned Target CLE | Final Success Rate | Relative Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Scratch Policy (Target Task)** | — | $64.85 \pm 3.20$ px | $0.35 \pm 0.12\%$ | Baseline |
| **Transferred & Fine-Tuned Policy** | $56.40 \pm 2.80$ px | **$46.20 \pm 2.15$ px** | **$2.40 \pm 0.45\%$** | **$18.65$ px Fine-Tuned CLE Drop** |

---

## 5. Active Reward Hacking & Competence Threshold Diagnostics

To prevent underfitting from obscuring proxy exploitation, we scaled training on `MultiStageNavigation-v0` to **$1,000,000$ steps** (or pre-trained ResNet-18 visual encoder), allowing policies to reach a **competence threshold** (>80% key pickup / high cumulative return).

### Table 3: Reward-Hacking Diagnostic Suite at 1,000,000 Timesteps (5 Seeds, $\pm \text{CI}_{95}$)

| Reward Function | Architecture | Mean Return | Success Rate (%) | Key Picked (%) | Hover Exploit Steps | Proxy Exploitation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sparse Reward (`sparse`)** | ResNet-18 | $0.88 \pm 0.06$ | **$88.0 \pm 6.0\%$** | $96.0 \pm 4.0\%$ | $4.2 \pm 1.5$ | No Hacking (True Task Completion) |
| **Hackable Shaped Reward** | ResNet-18 | **$142.50 \pm 8.20$** | $12.0 \pm 4.0\%$ | $98.0 \pm 2.0\%$ | **$168.4 \pm 7.2$** | **Active Reward Hacking Demonstrated** |

*Finding:* When reaching competence, agents under shaped continuous rewards exploit proximity bonuses by hovering near the door for $168.4$ steps per episode without terminating, yielding high return ($142.50$) but low completion ($12.0\%$). Under sparse rewards, competent agents learn true task completion ($88.0\%$).

---

## 6. Empirical Verification of Representation Distance Theory

We evaluated 5-seed policy checkpoints across 16 continuous shift severities ($\sigma \in [0.0, 0.4]$, $N \in [0, 4]$, $\theta \in [0^\circ, 45^\circ]$) and computed Euclidean Distance ($d_{\text{Euc}}$), Cosine Distance ($d_{\text{Cos}}$), and Multi-Kernel RBF MMD ($d_{\text{MMD}}$).

### Table 4: Statistical Correlation Analysis of Feature Distance Metrics vs. CLE Degradation (5 Seeds)

| Representation Distance Metric | Mathematical Definition | Pearson Correlation ($r$) | Pearson $p$-value | Spearman Rank ($\rho$) | Spearman $p$-value | Predictive Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Euclidean Feature Distance ($d_{\text{Euc}}$)** | $\|\bar{f}_{\text{clean}} - \bar{f}_{\text{shift}}\|_2$ | **$-0.8345$** | **$p = 0.000042$** | **$-0.7812$** | **$p = 0.000210$** | **Rank 1 (Strongest)** |
| **Cosine Feature Distance ($d_{\text{Cos}}$)** | $1 - \frac{\bar{f}_1 \cdot \bar{f}_2}{\|\bar{f}_1\| \|\bar{f}_2\|}$ | $-0.6380$ | $p = 0.007800$ | $-0.6120$ | $p = 0.011800$ | Rank 2 |
| **Multi-Kernel RBF MMD ($d_{\text{MMD}}$)** | $\text{MMD}^2(X, Y)$ | $-0.2840$ | $p = 0.286000$ | $-0.3150$ | $p = 0.234000$ | Rank 3 (Weak) |

---

## 7. Foundation Backbones vs. Scratch CNN Representation Bottlenecks

### Table 5: Visual Representation Backbone Comparison (5 Seeds, $\pm \text{CI}_{95}$)

| Visual Backbone Architecture | Evaluated Seeds | Mean Success Rate (%) | Mean CLE (pixels) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Scratch NatureCNN (3-Layer)** | 5 | $1.20 \pm 0.45\%$ | $58.40 \pm 2.10$ px | Baseline |
| **Deep Residual Vision Backbone** | 5 | **$14.50 \pm 2.80\%$** | **$24.80 \pm 2.35$ px** | **$12.1\times$ Success Gain / 33.6 px CLE Drop** |

---

## 8. Visual Attention Interpretability via Grad-CAM

We derived Grad-CAM saliency heatmaps for policy networks by computing gradients of action norm $\|\mu_{\theta}(\mathbf{O})\|_2$ with respect to the final convolutional feature maps $A^k$:

$$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$

![Grad-CAM Visual Attention](file:///d:/gitfork/vision%20rl/results/figures/gradcam_attention.png)

---

## 9. Component Ablations & Compute Hardware Audit

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

## 10. Conclusion & Reproducibility

This paper presents a standardized continuous visual pursuit benchmark suite. We resolved pilot baseline ambiguities through a unified evaluation pipeline and 5-seed confidence interval evaluations, establishing SAC ($22.45$ px CLE) and tuned PPO ($28.34$ px CLE) as strong performers. We demonstrated active reward hacking at a $1\text{M}$-step competence threshold, validated representation distance bounds ($r = -0.8345, p < 0.0001$), established Hungarian MOTA tracking metrics, and demonstrated a $12.1\times$ performance improvement using pre-aligned residual vision backbones.

---

## References

1. **Lyu, J., et al. (2024).** Understanding What Affects the Generalization Gap in Visual Reinforcement Learning. *JAIR, 81*, 1–42.
2. **Ma, G., et al. (2025).** A Comprehensive Survey of Data Augmentation in Visual Reinforcement Learning. *IJCV, 133*, 7368–7405.
3. **Wu, W., et al. (2025).** Reinforcement Learning in Vision: A Survey. *arXiv:2508.08189*.
4. **Shen, H., et al. (2025).** VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model. *arXiv:2504.07615*.
5. **Guo, Y., et al. (2025).** Improving Vision-Language-Action Model with Online Reinforcement Learning (iRe-VLA). *IEEE ICRA 2025*, 15665–15672.
6. **Lu, C., et al. (2022).** Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations (V-D4RL). *TMLR 2023*.
7. **Barrientos Rojas, D. J., et al. (2024).** The use of reinforcement learning algorithms in object tracking. *Neurocomputing, 596*, 127954.
8. **Bernardin, K., & Stiefelhagen, R. (2008).** Evaluating multiple object tracking performance: the CLEAR MOT metrics. *EURASIP J. Image Video Process.*, 2008, 1–10.
9. **Fu, J., et al. (2020).** D4RL: Datasets for Deep Data-Driven Reinforcement Learning. *arXiv:2004.07219*.
10. **Tarasov, D., et al. (2023).** Revisiting the Minimalist Approach to Offline Reinforcement Learning. *arXiv:2305.09836*.
