# Project Scope

## Research Question
**CNN embedding distance from the training distribution is a predictive, cheap-to-compute proxy for visual RL policy generalization gap, and can be used to select/design augmentation strategies without running full evaluation rollouts.**

## Context
This project aims to demonstrate that rather than evaluating an agent's generalization capability by running expensive full rollouts across unseen environments, one can simply compute the representation distance (e.g., Euclidean distance, Cosine distance, or MMD) in the CNN embedding space between clean training observations and perturbed observations. This distance is highly predictive of performance degradation.

Crucially, this project extends beyond correlation analysis by using this proxy *prescriptively*—to rank and select data augmentation strategies (e.g., Color Jitter, Random Shift, Cutout) for a given target distribution shift, significantly reducing compute costs.

## Primary Components
1. **Correlation Scale-Up**: Evaluating the predictive power of representation distance across multiple seeds (5+), standard benchmark environments (e.g., `procgen:procgen-coinrun-v0`, alongside the illustrative `MultiStageNavigation-v0`), and numerous (8-10) distinct perturbation conditions. 
   - *Environment Rationale*: Procgen was chosen as the required primary environment over DeepMind Control Suite due to its lightweight dependency profile (pip-installable) and built-in procedural backgrounds, which fit naturally with the visual perturbation scope of this project and mirror evaluations in recent literature (e.g. UCB-DrAC, PLR).
2. **Predictive Augmentation Selection**: Using the metric to rank candidate data augmentation strategies and comparing against ground-truth rollout evaluations.
3. **Generalization Baseline**: Benchmarking a simple DrQ-style (Data-Regularized Q/PPO) approach to ensure that the correlations hold both for unaugmented and augmented base models.
