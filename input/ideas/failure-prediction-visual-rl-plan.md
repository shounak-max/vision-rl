# Research Plan: Correlation Blindness in Representation-Shift Diagnostics for Visual RL

**Status:** Draft for preregistration  
**Source design:** [`plan.md`](../../plan.md)  
**Output:** Journal article, reproducible code, released checkpoints/configurations, and a per-episode results table

## 1. Research question

For trained pixel-based reinforcement-learning agents under controlled visual shifts, can a representation-shift statistic have weak contemporaneous correlation with task success yet strongly predict held-out failure when degradation is cliff-edge? Does this dissociation shrink when degradation is gradual?

The study tests predictive transfer across unseen shift families. It does not test test-time adaptation, real-robot transfer, VLA or generative policies, new agent algorithms, or the validity of theoretical bounds.

## 2. Hypotheses and primary claim

- **H1 — cliff-edge dissociation:** In cliff-edge agent/environment/family cells, the absolute Spearman correlation between the primary shift statistic and success is weak (at most 0.30), while held-out-family failure AUROC is high (at least 0.80).
- **H2 — graded control:** Graded cells have a lower rate of disagreement between correlation and AUROC categories than cliff-edge cells.
- **H3 — exploratory predictor comparison:** Predictor families differ in held-out AUROC. Grassmann distance is descriptive unless it improves on the centroid baseline on held-out families.
- **Null outcomes:** Correlation and AUROC have similar utility in both regimes, or no predictor transfers across held-out families. Either is a reportable result.

**Primary test:** Compare the cell-level rate of correlation/AUROC category disagreement between cliff-edge and graded regimes, using the centroid-distance predictor. Define the effect as the discordance rate in cliff-edge cells minus the rate in graded cells, among cells with both metrics outside the intermediate band. Test this contrast with a two-sided permutation test that permutes regime labels within agent-by-environment strata. Report the difference and a seed-blocked bootstrap 95% confidence interval; each bootstrap draw resamples test training-seed IDs within agent-by-environment and keeps all rows from each sampled seed together across shift families. Estimate regime labels from fit seeds 1–5 as specified above. Treat the contrast as observational, not causal. Use a two-sided 0.05 significance level; this is the single confirmatory test. Record the analysis random seed and replicate counts.

Classify absolute Spearman correlation as **weak** (≤0.30), **intermediate** (>0.30 and <0.50), or **strong** (≥0.50). Classify held-out AUROC as **low** (≤0.60), **intermediate** (>0.60 and <0.80), or **high** (≥0.80). A cell is category-discordant when both measures are outside the intermediate band but fall in opposite categories (weak/low versus strong/high). Cells with either measure intermediate are indeterminate for this primary classification and remain in continuous-metric analyses. Report the number and characteristics of indeterminate cells.

**Interpretation blocker — binary-outcome prevalence:** Spearman correlation with a binary outcome is not independent of AUROC and failure prevalence. Under common continuous-score assumptions, a useful approximation is `|rho| ≈ sqrt(12 p (1-p)) (AUROC - 0.5)`, where `p` is failure prevalence. Thus weak correlation plus high AUROC can arise from rare failures alone. The planned thresholds and H1 are descriptive until a prevalence-aware confirmatory estimand is preregistered. The analysis reports failure prevalence and the approximate prevalence reference per cell; do not interpret an H1 count as evidence of a distinct correlation-blindness effect without the protocol amendment.

## 3. Study design

### Agents and environments

- Agents: DrQ-v2, SVEA, TACO, and one robustness-oriented method (MaDi, subject to implementation and compute feasibility).
- Environments: Visual Pursuit, Procgen CoinRun, and one Distracting Control Suite task.
- Train 10 independent seeds per agent and environment: 120 training runs total.
- For Procgen, freeze disjoint training and evaluation level ranges and verify that reset seeds actually control the legacy environment before collecting confirmatory outcomes.
- Use official implementations and record code revisions, configuration, hardware, and dependencies.

The one-environment pilot must confirm that each selected implementation, environment, shift, and outcome can be run and logged before the full training campaign.

### Visual shifts and evaluation

- Target four shift families: viewpoint, distractors, noise, and color. The pilot may establish that a family cannot be implemented as specified; document any exclusion before full training.
- Use 10 ordered severity levels per family, including the unshifted reference where appropriate. Store the exact transformation parameters and randomization seeds.
- All stochastic shift transforms must be seeded from episode reset seeds; a viewpoint transform is sampled once per episode, not independently per frame.
- Evaluate 30 episodes per agent/environment/family/severity cell, distributed evenly across the 10 training seeds (three episodes per seed). This gives 14,400 episodes at four families: 4 agents × 3 environments × 4 families × 10 levels × 30 episodes. If only three families pass the pilot, the corresponding total is 10,800.
- Do not inspect held-out-family test outcomes while choosing features, thresholds, or analysis settings.

### Success and failure definitions — preregistration gate

CLE is not a generally meaningful outcome for CoinRun or a control-suite task. Before any full training run, resolve and freeze the task-native outcome definitions in a signed configuration:

1. Use one task-native binary success definition per environment, fixed across agents and shifts. For Visual Pursuit, specify the CLE threshold and aggregation window; for CoinRun and the selected control-suite task, specify the environment-native completion/success criterion.
2. Keep the primary outcome as episode success/failure. Analyze return drop and failure-within-horizon as secondary outcomes.
3. Freeze the exact definitions and any success-threshold sensitivity range before collecting study outcomes.

Do not silently apply a CLE rule to tasks for which CLE is undefined. If a valid, comparable task-native success definition cannot be fixed before training, pause the cross-environment study and revise the protocol.

### Regime classification

For each agent × environment × shift-family cell, estimate the normalized severity width over which success falls from 90% to 10%. Use linear interpolation between adjacent measured severity levels; do not choose a smoothing method after seeing results.

- **Cliff-edge:** 90%-to-10% width ≤0.20 of the normalized severity sweep.
- **Graded:** width ≥0.50.
- **Intermediate:** width >0.20 and <0.50; retain for descriptive analyses but exclude from the primary cliff-edge-versus-graded contrast.

Apply the same predeclared rule to every cell. Report cells that do not cross both success levels within the tested sweep as unclassified, not as either regime.

For leakage control, estimate each cell's regime from training seeds 1–5 only; evaluate correlation and AUROC for that cell on its held-out-family test episodes from seeds 6–10.

## 4. Validity and leakage controls

Before full evaluation, verify for every trained agent that actions are produced by its learned actor using its learned encoder. Log reward and success learning curves for every run. Compare the intact policy with zeroed-encoder and shuffled-encoder ablations under the same evaluation protocol. Predefine the quantitative pass criterion and random-policy reference in the pilot configuration; an agent/environment failing the gate is not included in confirmatory claims.

For prediction:

- Keep training seeds 1–5 for fitting and threshold selection; reserve seeds 6–10 for testing.
- Use leave-one-shift-family-out evaluation. For each held-out family, fit the predictor and any decision threshold using only the other families and seeds 1–5; evaluate once on the held-out family and seeds 6–10.
- Keep all episodes, frames, and windows from one seed together in the same split.
- Exclude the failure event and adjacent frames from feature windows. Freeze the window length and forecast horizon **H** in the pilot configuration before examining outcome associations.
- Include shift severity alone as a baseline to test whether a predictor adds value beyond identifying the intervention.
- Keep the held-out set untouched until the analysis code, thresholds, and exclusions are frozen.

## 5. Predictors and outcomes

### Predictors

Primary predictor: the existing centroid-distance statistic, reconciled with and documented against Equation 1 before the pilot. Use one shared backbone for classical and semantic-weighted distances.

Secondary/exploratory predictors:

- Full-covariance distance and Mahalanobis distance.
- Grassmann distance for k = 5, 10, and 20.
- Semantic-weighted distance. Use simulator-provided segmentation where available; otherwise label the mask as a proxy and report that limitation.
- Critic or ensemble uncertainty.
- RND-style embedding score.
- Baselines: pixel statistics, action entropy, and shift severity alone.

### Outcomes and reporting

- Primary label: episode failure under the frozen task-native success definition.
- Secondary labels: return drop and failure within the predeclared horizon H.
- Report Pearson and Spearman correlations, AUROC, AUPRC, calibration, and threshold transfer, all on the held-out family/seeds where applicable.
- Report seed-blocked bootstrap confidence intervals, IQM, and paired predictor comparisons. Identify exploratory comparisons as exploratory and account for multiplicity in their interpretation.

Recommend reporting correlation and held-out threshold-based prediction together, not correlation alone. Use the centroid statistic as the reference predictor; describe Grassmann results without superiority claims unless the preregistered held-out comparison supports them.

## 6. Analysis sequence

1. Freeze environment versions, agent revisions, success criteria, shift parameters, severity grid, episode allocation, feature windows, forecast horizon, exclusions, and analysis code.
2. Run the one-environment pilot and validity gate. Resolve implementation failures and the success-definition gate before scaling up.
3. Train all eligible agent/environment/seed combinations and preserve learning curves and checkpoints.
4. Evaluate the predeclared shifts and write one immutable row per episode, including agent, environment, training seed, shift family, severity, episode seed, predictor values, return, success/failure, and relevant window/horizon metadata.
5. Derive regime labels using the fixed 90%-to-10% rule.
6. Run leave-one-family-out prediction with the fixed seed split. Produce held-out metrics and baseline comparisons.
7. Run the single primary regime-by-metric interaction test. Run all remaining analyses as secondary or exploratory.
8. Generate every paper table and figure from versioned analysis code and the released per-episode table.

No reward-hacking claim is in scope unless proxy reward and task objective are both measured and their mismatch is demonstrated.

## 7. Feasibility and release

Training is the main compute cost; evaluation and analysis are expected to be less expensive. Begin with one environment and scale only after the validity and logging gates pass. Track GPU hours, wall time, failures, and completed episodes during the pilot; revise the full-run feasibility estimate before committing to 120 runs.

Release, subject to applicable upstream licenses:

- Experiment configurations, environment/agent revision identifiers, and training/evaluation scripts.
- Checkpoints and learning curves needed to reproduce the study.
- The per-episode table and schema/data dictionary.
- Analysis scripts that regenerate all paper numbers and figures.
- A short recommendation on predictive metrics to report.

Do not release incomplete or invalidated results as confirmatory evidence. Preserve exclusions and failed runs in an audit log.

## 8. Decisions to lock before the first full run

The following are preregistration blockers, not decisions to make after inspecting study outcomes:

1. Task-native success definitions, including the CLE threshold and window for Visual Pursuit.
2. The exact pilot environment/task and one DMC task, plus disjoint and reproducible Procgen training/evaluation levels.
3. Whether MaDi and all four shift families can be implemented with the available code and compute.
4. The quantitative encoder-ablation pass criterion and random-policy reference.
5. Feature-window length and forecast horizon H.
6. The precise severity parameterization and repeatable episode-seed scheme.
7. Amend the H1/primary estimand to account for the relationship among binary-outcome Spearman correlation, AUROC, and failure prevalence; then freeze predictor orientation and calibration procedures, including semantic masks and covariance regularization.
8. The analysis script for blocked permutations and seed-blocked bootstrap intervals.

The numeric correlation, AUROC, and regime-width cutoffs in this document are fixed for the initial protocol. Any change after full-run outcomes are inspected must be labeled a protocol amendment and cannot replace the original confirmatory analysis.
