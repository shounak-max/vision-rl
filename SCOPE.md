# Project Scope

## Research Question
**Can a representation-shift statistic have weak contemporaneous correlation with task success yet predict held-out failure under cliff-edge degradation, and does that dissociation shrink under graded degradation?**

## Context
The preregistered study evaluates leakage-aware failure prediction across held-out visual-shift families. Correlation/AUROC discordance is interpreted descriptively until a prevalence-aware estimand is preregistered; the full-study run gate is currently closed.

Crucially, this project extends beyond correlation analysis by using this proxy *prescriptively*—to rank and select data augmentation strategies (e.g., Color Jitter, Random Shift, Cutout) for a given target distribution shift, significantly reducing compute costs.

## Primary Components
1. **Correlation Scale-Up**: Evaluating the predictive power of representation distance across multiple seeds (5+), standard benchmark environments (e.g., `procgen:procgen-coinrun-v0`, alongside the illustrative `MultiStageNavigation-v0`), and numerous (8-10) distinct perturbation conditions. 
   - *Environment Rationale*: Procgen was chosen as the required primary environment over DeepMind Control Suite due to its lightweight dependency profile (pip-installable) and built-in procedural backgrounds, which fit naturally with the visual perturbation scope of this project and mirror evaluations in recent literature (e.g. UCB-DrAC, PLR).
2. **Exploratory Transform Sensitivity (Legacy)**: Compare reset-state representation distances with the performance of one fixed policy under test-time image transforms. This does not evaluate training-time augmentation selection; the shared shift/candidate observations are dependent, so its correlations are descriptive only. A training-time selection claim requires separately trained policies and a preregistered independent evaluation design.
3. **Generalization Baseline**: Benchmarking a simple DrQ-style (Data-Regularized Q/PPO) approach to ensure that the correlations hold both for unaugmented and augmented base models.
