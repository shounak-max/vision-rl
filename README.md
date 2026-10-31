# Visual-RL Failure Prediction Study

This repository contains the study protocol and analysis foundation for testing
whether representation-shift statistics predict held-out visual-RL failure,
especially when performance degradation is cliff-edge.

## Current status

The analysis scaffold and legacy environment/PPO experiment code are present;
the preregistered study is **not runnable yet**. The legacy experiments do not
provide the selected DrQ-v2/SVEA/TACO/MaDi adapters or the current
per-episode study table. The run gate in
[`config/study.json`](config/study.json) remains closed until the official
training/evaluation adapters and protocol blockers are resolved.

In particular, the Distracting Control Suite task and its success definition,
the encoder-ablation criterion, and feature-window length are not yet frozen.
Do not treat pilot or future incomplete runs as confirmatory results.

## Setup

Python 3.9 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Commands

Validate the protocol and inspect the currently closed full-study run gate:

```powershell
vision-rl-study validate-protocol
```

Validate a per-episode CSV:

```powershell
vision-rl-study validate-episodes path\to\episodes.csv
```

The CSV requires `agent`, `environment`, `training_seed`, `shift_family`,
normalized `severity` in `[0, 1]`, `episode_seed`, `episode_return`, `success`
(0 or 1), and every predictor column listed in the protocol.

Run leave-one-family-out evaluation and severity-regime classification:

```powershell
vision-rl-study analyze path\to\episodes.csv --output results\analysis.json
```

For each held-out family, threshold fitting uses only non-held-out families and
training seeds 1–5. Evaluation uses the held-out family and seeds 6–10.
Regime labels use fit seeds only. The analysis reports episode-level
correlation, AUROC, AUPRC, threshold transfer, and the preregistered blocked
regime interaction with a seed-blocked bootstrap interval when both regimes are
estimable. Missing primary-test conditions are explicitly reported as
not-estimable; predictor metrics still require both success classes in each
fit and test split.

The current four-family allocation is 14,400 episodes (30 episodes per
severity cell across 10 training seeds), not 144,000.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Protocol and data

- Research design: [`plan.md`](plan.md)
- Preregistration draft: [`input/ideas/failure-prediction-visual-rl-plan.md`](input/ideas/failure-prediction-visual-rl-plan.md)
- Machine-readable protocol: [`config/study.json`](config/study.json)

The episode table should be generated only from frozen evaluation runs. Keep
checkpoints, run configurations, and logs with the data release so all
reported results can be regenerated.
