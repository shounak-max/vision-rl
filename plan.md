---
title: "Does weak correlation hide strong threshold-based prediction of held-out failure for trained visual RL agents under cliff-edge shifts?"
type: design
created: 2026-10-31
updated: 2026-10-31
tags: [visual-rl, distribution-shift, failure-prediction, representation-distance, evaluation-methodology]
status: draft
author: mixed
---

# Correlation blindness in representation-shift diagnostics for visual RL

## Research Question
For trained pixel-based RL agents under controlled visual shifts, does a weak contemporaneous correlation between a representation-shift statistic and task performance coexist with strong threshold-based prediction of held-out failure when failure is cliff-edge, and does that dissociation disappear when failure is graded?

Hypotheses:
- **H1 (dissociation, cliff-edge cells):** correlation between the shift statistic and success is low to moderate, while held-out-family AUROC for predicting failure is high.
- **H2 (graded control):** in graded-failure cells, correlation and AUROC agree, so the dissociation is smaller. H1 and H2 together test a failure-sharpness-by-metric interaction.
- **H3 (secondary, exploratory):** predictor families (centroid, full-covariance, Grassmann at several k, semantic-weighted, critic uncertainty) differ in held-out AUROC. Grassmann distance is treated as descriptive unless it beats the centroid statistic on held-out families.
- **Null result:** AUROC tracks correlation in both regimes, or no predictor transfers across shift families. The numeric thresholds for "low", "high" and "agree" are fixed in the research plan before any run.

## Relevance / Forschungsstand (brief)
Lyu et al. (2024) link representation distance to the generalization gap in visual RL; their bound rests on a uniform deviation bound and Lipschitz conditions, so weak results for a centroid statistic do not refute it. Runtime failure prediction is active for VLA and generative robot policies (SAFE, FIPER, SAFECAST, early-warning studies), and a 2026 study stresses that fitting and evaluating on shared shifts shows only retrospective separability. Detecting shift and adapting to it are separate problems (2026 visual model-based RL study). The gap addressed here: trained pixel-based RL agents, controlled severity sweeps, held-out shift families, and a graded-failure control. This reflects the literature search (input/bibliography/literaturguide.md plus 2026 additions), not proof of absence.

## Method
Step 0 (validity gate): before any run, each agent's actions must come from its learned actor on its learned encoder; an encoder-ablation test (zeroed or shuffled encoder) must collapse performance; reward and success learning curves are logged for every agent.

Design:
- Agents: DrQ-v2, SVEA, TACO, one robustness-oriented agent (such as MaDi); 10 training seeds per agent per environment (120 training runs).
- Environments: Visual Pursuit, Procgen CoinRun, one Distracting Control Suite task.
- Shifts: 3 to 4 families (for example viewpoint, distractors, noise, color); 10 severity levels per family; 30 episodes per level, evenly allocated across 10 training seeds (14,400 episodes with four families).
- Success: use one task-native binary success definition per environment, fixed across agents and shifts. Freeze the Visual Pursuit CLE threshold/window and the native CoinRun/Distracting Control Suite completion criteria before training; run the predeclared threshold sensitivity analysis.
- Regime classification: for each agent x environment x shift-family cell, the severity interval over which success falls from 90% to 10%, as a fraction of the sweep; pre-specified cutoffs (fixed in the plan) label cells cliff-edge or graded. The contrast is observational and is reported as such.
- Predictors: centroid distance (Eq. 1, reconciled with the code), full-covariance and Mahalanobis, Grassmann at k = 5, 10, 20, semantic-weighted distance, critic or ensemble uncertainty, RND-style embedding score, and baselines (pixel statistics, action entropy, shift severity alone). Classical and semantic distances share one backbone.
- Targets: episode-level failure, return drop, failure within the next H steps given a feature window.
- Evaluation: leave-one-family-out with thresholds fit on other families and training seeds 1 to 5, tested on the held-out family and seeds 6 to 10; correlation (Pearson, Spearman) reported next to AUROC, AUPRC, calibration and threshold transfer; regime-by-metric interaction tested with a permutation test; seed-blocked bootstrap CIs, IQM and paired predictor comparisons.

## Data Sources / Material Basis
- Trained agents produced in this project from the official DrQ-v2, SVEA, TACO and MaDi implementations.
- Environments: Visual Pursuit, Procgen CoinRun, Distracting Control Suite (Stone et al., 2021).
- The project's existing feature-extraction and distance code, reconciled with Eq. 1.
- Released artifacts: configs, checkpoints, and a per-episode table (shift family, severity, seed, predictor values, outcome) from which every paper number is generated.

## Output Target
☐ Book chapter   ☒ Journal article   ☐ Grant proposal   ☐ Talk   ☐ Other

Plus the code and per-episode table, and a short recommendation on which predictive metrics to report.

## Expected Results
- Positive: dissociation in cliff-edge cells, shrinking in graded cells; at least one predictor transfers across held-out families; takeaway: report threshold-based prediction, not correlation alone.
- Mixed: dissociation for some agents or environments only; the paper characterizes when it holds.
- Null: AUROC tracks correlation in both regimes, or no predictor transfers; reported as a bounded negative result.

## Ethical / Methodological Risks
- Shift recognition masquerading as failure prediction: include a shift-severity-only baseline; report AUROC beyond what severity alone predicts.
- Leakage: windows exclude failure-event and adjacent frames; thresholds fit on training families and seeds only; held-out set touched once.
- Regime confound (observational contrast): report within agents and environments where both regimes occur; state the limit.
- Success-definition sensitivity: rerun over a range of CLE thresholds.
- Multiple comparisons: one pre-specified primary test (the interaction); everything else exploratory.
- Semantic-weighting validity: use simulator-provided segmentation where available, otherwise label the mask a proxy; drop interventions that cannot be implemented as described.
- Reward-hacking claims are dropped unless a proxy-reward versus task-objective mismatch is measured.
- No human-subject or sensitive-data concerns.

## Feasibility
Shared cluster or several GPUs, weeks. Training is the main cost; evaluation and analysis are comparatively cheap. The plan begins with a one-environment pilot. No near deadline (journal target).

## Scope Boundaries (what is NOT investigated)
Not investigated: the test-time adaptation safeguard (separate design doc later), VLA and generative-policy failure, real-robot shift, new agent algorithms, and any proof or refutation of Lyu et al.'s bound.

## Next Step
Hand off to writing-research-plan to produce `input/ideas/failure-prediction-visual-rl-plan.md`.d