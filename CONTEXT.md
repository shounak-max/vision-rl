# Project Context Log

## 2026-08-12
### What was done this session
- **Cleanup**: Transitioned the repository from a 5-phase exploratory codebase into a focused research project testing a single claim about representation distance. Deleted unused Phase 1, Phase 3, and Phase 5 code (e.g., `navigation_envs.py`, MOTA components, VLA baselines). Cleared outdated results and removed downloaded reference PDFs, replacing them with a standardized `references.md` and `SCOPE.md`.
- **Environment Selection**: Integrated `procgen:procgen-coinrun-v0` as the primary standard generalization benchmark. Kept the custom `MultiStageNavigation-v0` as a secondary, illustrative environment.
- **Experimental Script Generation**: 
  - Created `baselines/run_scale_experiment.py` to evaluate representation distance vs. generalization gap across 5 seeds, 10M timesteps for `procgen`, and 10 continuous visual perturbation conditions.
  - Created `baselines/predictive_augmentation_selection.py` to test the predictive power of representation distance for ranking candidate augmentations (e.g., Color Jitter, Shift, Cutout) computationally cheaply via test-time augmentation (TTA), compared against ground-truth evaluation rollouts.
  - Formulated a standard DrQ-style baseline using continuous data augmentation wrappers.

### Current state of the core claim
- **Not yet run**. The experimental scale-up code is implemented and verified syntactically, but the actual training (10M steps x 5 seeds) and evaluations have not been executed on the cluster yet.

### Decisions made and why
- **Chose Procgen over DeepMind Control Suite (DMC)**: Procgen pip-installed cleanly and provides robust built-in procedural distractors (backgrounds) which fit the project's visual perturbation scope natively. DMC requires heavier dependencies (MuJoCo) and isn't necessary given the illustrative nature of Procgen for generalization evaluation in recent literature (e.g., PLR).
- **Test-Time Augmentation (TTA) Proxy**: The predictive augmentation selection script uses TTA as the computationally cheap proxy to rank augmentation strategies, as it avoids any PPO rollout/training cost entirely.

### Open questions, blockers, and flags
- **Compute Budget Flag**: Training the scale-up on `procgen:procgen-coinrun-v0` across 5 seeds and 2 algorithms (PPO vs DrQ baseline) requires 100M total PPO interaction steps. This is estimated to consume roughly **~20-30 GPU-hours** on a standard GPU machine. Requires human approval before kicking off.

### Exact Next Steps
- Human reviewer to approve the 20-30 GPU-hour compute budget (or specify if seeds/steps should be trimmed).
- Execute `baselines/run_scale_experiment.py` (preferably overnight or on a compute cluster).
- Execute `baselines/predictive_augmentation_selection.py` using the trained baseline checkpoint.
- Analyze the output JSON correlation files and update this log with the exact empirical results (Spearman rho and p-values).
