"""Leakage-aware prediction and regime analysis."""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def family_seed_split(
    rows: Sequence[Dict[str, str]],
    held_out_family: str,
    fit_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    fit_seed_set = set(fit_seeds)
    test_seed_set = set(test_seeds)
    if fit_seed_set & test_seed_set:
        raise ValueError("Fit and test seed sets overlap.")

    fit_rows = []
    test_rows = []
    for row in rows:
        seed = int(row["training_seed"])
        family = row["shift_family"]
        if seed in fit_seed_set and family != held_out_family:
            fit_rows.append(row)
        elif seed in test_seed_set and family == held_out_family:
            test_rows.append(row)

    if not fit_rows:
        raise ValueError("No fit rows remain for held-out family {!r}.".format(
            held_out_family
        ))
    if not test_rows:
        raise ValueError("No test rows for held-out family {!r}.".format(
            held_out_family
        ))
    return fit_rows, test_rows


def _column(rows: Sequence[Dict[str, str]], name: str) -> np.ndarray:
    values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("{} contains a non-finite value.".format(name))
    return values


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Failure labels must be binary 0/1.")
    positive = labels == 1
    negatives = labels == 0
    if not np.any(positive) or not np.any(negatives):
        raise ValueError("AUROC/AUPRC require both failure classes.")

    auc = float(stats.mannwhitneyu(
        scores[positive], scores[negatives], alternative="two-sided"
    ).statistic / (positive.sum() * negatives.sum()))
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    precision = true_positives[group_ends] / (
        true_positives[group_ends] + false_positives[group_ends]
    )
    recall = true_positives[group_ends] / positive.sum()
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    auprc = float(np.sum((recall - previous_recall) * precision))
    prevalence = float(np.mean(positive))
    return {
        "auroc": auc,
        "auprc": auprc,
        "prevalence": prevalence,
        "average_precision_baseline": prevalence,
    }


def fit_failure_score_direction(
    fit_rows: Sequence[Dict[str, str]], predictor: str
) -> Tuple[int, float]:
    """Orient scores using fit rows only so larger values indicate failure."""
    failures = 1.0 - _column(fit_rows, "success")
    if not np.all(np.isin(failures, (0.0, 1.0))):
        raise ValueError("success must contain only 0 and 1.")
    raw_auc = _binary_metrics(failures, _column(fit_rows, predictor))["auroc"]
    direction = -1 if raw_auc < 0.5 else 1
    return direction, raw_auc


def fit_failure_threshold(
    fit_rows: Sequence[Dict[str, str]], predictor: str, score_direction: int = 1
) -> float:
    """Choose the score cutoff maximizing Youden's J on fit data only."""
    if score_direction not in (-1, 1):
        raise ValueError("score_direction must be -1 or 1.")
    scores = score_direction * _column(fit_rows, predictor)
    failures = 1.0 - _column(fit_rows, "success")
    if not np.all(np.isin(failures, (0.0, 1.0))):
        raise ValueError("success must contain only 0 and 1.")
    if len(np.unique(failures)) != 2:
        raise ValueError("Threshold fitting requires both outcome classes.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_failures = failures[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]
    true_positives = np.cumsum(sorted_failures == 1.0)[group_ends]
    false_positives = np.cumsum(sorted_failures == 0.0)[group_ends]
    youden_j = (
        true_positives / np.sum(failures == 1.0)
        - false_positives / np.sum(failures == 0.0)
    )
    best_index = int(np.argmax(youden_j))
    return float(sorted_scores[group_ends[best_index]])


def evaluate_predictor(
    fit_rows: Sequence[Dict[str, str]],
    test_rows: Sequence[Dict[str, str]],
    predictor: str,
) -> Dict[str, Any]:
    score_direction, raw_fit_auroc = fit_failure_score_direction(
        fit_rows, predictor
    )
    threshold = fit_failure_threshold(fit_rows, predictor, score_direction)
    scores = score_direction * _column(test_rows, predictor)
    labels = 1.0 - _column(test_rows, "success")
    metrics = _binary_metrics(labels, scores)
    predicted = scores >= threshold
    actual = labels == 1.0
    sensitivity = float(np.mean(predicted[actual])) if np.any(actual) else float("nan")
    specificity = (
        float(np.mean(~predicted[~actual])) if np.any(~actual) else float("nan")
    )
    accuracy = float(np.mean(predicted == actual))
    metrics.update({
        "score_direction": score_direction,
        "raw_fit_auroc": raw_fit_auroc,
        "threshold_fit_on_training_families": threshold,
        "threshold_test_accuracy": accuracy,
        "threshold_test_sensitivity": sensitivity,
        "threshold_test_specificity": specificity,
        "test_episodes": int(len(test_rows)),
        "test_failures": int(np.sum(actual)),
    })
    return metrics


def correlation_metrics(
    rows: Sequence[Dict[str, str]], predictor: str
) -> Dict[str, float]:
    x = _column(rows, predictor)
    success = _column(rows, "success")
    if len(x) < 2:
        raise ValueError("At least two episodes are required for correlations.")
    pearson = stats.pearsonr(x, success)
    spearman = stats.spearmanr(x, success)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "episodes": int(len(rows)),
    }


def classify_correlation_auroc(
    spearman_rho: float, auroc: float, cutoffs: Dict[str, float]
) -> Dict[str, Any]:
    if not math.isfinite(spearman_rho) or not math.isfinite(auroc):
        return {
            "correlation_class": "unavailable",
            "auroc_class": "unavailable",
            "discordant": None,
        }

    absolute_rho = abs(spearman_rho)
    if absolute_rho <= cutoffs["correlation_weak_max"]:
        correlation_class = "weak"
    elif absolute_rho >= cutoffs["correlation_strong_min"]:
        correlation_class = "strong"
    else:
        correlation_class = "intermediate"

    if auroc <= cutoffs["auroc_low_max"]:
        auroc_class = "low"
    elif auroc >= cutoffs["auroc_high_min"]:
        auroc_class = "high"
    else:
        auroc_class = "intermediate"

    if correlation_class == "intermediate" or auroc_class == "intermediate":
        discordant = None
    else:
        discordant = (
            (correlation_class == "weak" and auroc_class == "high")
            or (correlation_class == "strong" and auroc_class == "low")
        )
    return {
        "correlation_class": correlation_class,
        "auroc_class": auroc_class,
        "discordant": discordant,
    }


def _discordance_rate_difference(cells: Sequence[Dict[str, Any]]) -> Optional[float]:
    eligible = [
        cell for cell in cells
        if cell.get("regime") in ("cliff_edge", "graded")
        and cell.get("discordant") in (True, False)
    ]
    cliff_edge = [cell for cell in eligible if cell["regime"] == "cliff_edge"]
    graded = [cell for cell in eligible if cell["regime"] == "graded"]
    if not cliff_edge or not graded:
        return None
    cliff_rate = float(np.mean([cell["discordant"] for cell in cliff_edge]))
    graded_rate = float(np.mean([cell["discordant"] for cell in graded]))
    return cliff_rate - graded_rate


def blocked_regime_permutation_test(
    cells: Sequence[Dict[str, Any]],
    permutations: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Test cliff-edge minus graded discordance, permuting labels within agent/env."""
    eligible = [
        cell for cell in cells
        if cell.get("regime") in ("cliff_edge", "graded")
        and cell.get("discordant") in (True, False)
    ]
    observed = _discordance_rate_difference(eligible)
    if observed is None:
        raise ValueError(
            "The primary test requires evaluable cliff-edge and graded cells."
        )
    if permutations < 1:
        raise ValueError("permutations must be positive.")

    strata: Dict[Tuple[str, str], List[int]] = {}
    for index, cell in enumerate(eligible):
        stratum = (str(cell["agent"]), str(cell["environment"]))
        strata.setdefault(stratum, []).append(index)

    regimes = [cell["regime"] for cell in eligible]
    discordance = [bool(cell["discordant"]) for cell in eligible]
    rng = np.random.default_rng(random_seed)
    at_least_as_extreme = 0
    for _ in range(permutations):
        permuted_regimes = list(regimes)
        for indices in strata.values():
            permuted = rng.permutation([regimes[index] for index in indices])
            for index, regime in zip(indices, permuted):
                permuted_regimes[index] = str(regime)
        permuted_cells = [
            {"regime": regime, "discordant": discordant}
            for regime, discordant in zip(permuted_regimes, discordance)
        ]
        permuted_difference = _discordance_rate_difference(permuted_cells)
        if (
            permuted_difference is not None
            and abs(permuted_difference) >= abs(observed)
        ):
            at_least_as_extreme += 1

    p_value = (at_least_as_extreme + 1) / (permutations + 1)
    cliff_count = sum(cell["regime"] == "cliff_edge" for cell in eligible)
    graded_count = sum(cell["regime"] == "graded" for cell in eligible)
    return {
        "difference_cliff_minus_graded": observed,
        "two_sided_permutation_p": float(p_value),
        "permutations": permutations,
        "random_seed": random_seed,
        "cliff_edge_cells": cliff_count,
        "graded_cells": graded_count,
        "indeterminate_cells": len(cells) - len(eligible),
    }


def seed_blocked_bootstrap_ci(
    rows: Sequence[Dict[str, str]],
    regimes: Dict[Tuple[str, str, str], Dict[str, Any]],
    predictor: str,
    cutoffs: Dict[str, float],
    test_seeds: Sequence[int],
    replicates: int,
    random_seed: int,
    score_directions: Optional[Dict[Tuple[str, str, str], int]] = None,
) -> Dict[str, Any]:
    """Bootstrap test seeds within each agent/environment, keeping their rows together."""
    if replicates < 1:
        raise ValueError("replicates must be positive.")

    cell_seed_rows: Dict[
        Tuple[str, str, str], Dict[int, List[Dict[str, str]]]
    ] = {}
    seeds_by_stratum: Dict[Tuple[str, str], set] = {}
    test_seed_set = set(test_seeds)
    for row in rows:
        seed = int(row["training_seed"])
        if seed not in test_seed_set:
            continue
        cell = (row["agent"], row["environment"], row["shift_family"])
        stratum = (row["agent"], row["environment"])
        cell_seed_rows.setdefault(cell, {}).setdefault(seed, []).append(row)
        seeds_by_stratum.setdefault(stratum, set()).add(seed)

    if not seeds_by_stratum:
        raise ValueError("No test-seed rows are available for bootstrap.")

    rng = np.random.default_rng(random_seed)
    differences = []
    for _ in range(replicates):
        sampled_seeds = {
            stratum: rng.choice(sorted(seeds), size=len(seeds), replace=True)
            for stratum, seeds in seeds_by_stratum.items()
        }
        bootstrap_cells = []
        for cell, regime_result in regimes.items():
            regime = regime_result["regime"]
            if regime not in ("cliff_edge", "graded"):
                continue
            stratum = (cell[0], cell[1])
            rows_by_seed = cell_seed_rows.get(cell, {})
            sampled_rows = []
            for seed in sampled_seeds.get(stratum, []):
                sampled_rows.extend(rows_by_seed.get(int(seed), []))
            if len(sampled_rows) < 2:
                continue

            labels = 1.0 - _column(sampled_rows, "success")
            if len(np.unique(labels)) != 2:
                continue
            direction = (
                score_directions.get(cell, 1)
                if score_directions is not None else 1
            )
            rho = float(stats.spearmanr(
                _column(sampled_rows, predictor),
                _column(sampled_rows, "success"),
            ).statistic)
            auroc = _binary_metrics(
                labels, direction * _column(sampled_rows, predictor)
            )["auroc"]
            classification = classify_correlation_auroc(rho, auroc, cutoffs)
            bootstrap_cells.append({
                "regime": regime,
                "discordant": classification["discordant"],
            })

        difference = _discordance_rate_difference(bootstrap_cells)
        if difference is not None:
            differences.append(difference)

    if not differences:
        raise ValueError(
            "No bootstrap replicate contained evaluable cliff-edge and graded cells."
        )
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "confidence_level": 0.95,
        "difference_ci_lower": float(lower),
        "difference_ci_upper": float(upper),
        "valid_replicates": len(differences),
        "requested_replicates": replicates,
        "random_seed": random_seed,
    }


def _crossing_index(successes: np.ndarray, target: float, start: int = 0) -> Optional[int]:
    for index in range(start, len(successes) - 1):
        left = successes[index]
        right = successes[index + 1]
        if left >= target and right <= target and left != right:
            return index
    return None


def _crossing_severity(
    severities: np.ndarray, successes: np.ndarray, target: float, start: int = 0
) -> Optional[float]:
    index = _crossing_index(successes, target, start)
    if index is None:
        return None
    left = successes[index]
    right = successes[index + 1]
    fraction = (left - target) / (left - right)
    return float(
        severities[index] + fraction * (severities[index + 1] - severities[index])
    )


def classify_regime(
    severity_success: Iterable[Tuple[float, float]], cutoffs: Dict[str, float]
) -> Dict[str, Any]:
    points = sorted(severity_success)
    severities = np.asarray([point[0] for point in points], dtype=np.float64)
    successes = np.asarray([point[1] for point in points], dtype=np.float64)
    if len(points) < 2 or len(np.unique(severities)) != len(severities):
        raise ValueError("Regime classification needs unique severity levels.")
    if np.any(np.diff(severities) <= 0):
        raise ValueError("Severity levels must increase.")
    if np.any((successes < 0.0) | (successes > 1.0)):
        raise ValueError("Success rates must be within [0, 1].")
    sweep_width = float(severities[-1] - severities[0])
    if sweep_width <= 0.0:
        raise ValueError("Severity sweep must have positive width.")

    crossing_90_index = _crossing_index(successes, 0.9)
    if crossing_90_index is None:
        return {"regime": "unclassified", "normalized_width": None}
    crossing_90 = _crossing_severity(severities, successes, 0.9, start=crossing_90_index)
    crossing_10 = _crossing_severity(
        severities, successes, 0.1, start=crossing_90_index
    )
    if crossing_10 is None or crossing_10 < crossing_90:
        return {"regime": "unclassified", "normalized_width": None}

    normalized_width = (crossing_10 - crossing_90) / sweep_width
    if normalized_width <= cutoffs["cliff_edge_width_max"]:
        regime = "cliff_edge"
    elif normalized_width >= cutoffs["graded_width_min"]:
        regime = "graded"
    else:
        regime = "intermediate"
    return {"regime": regime, "normalized_width": float(normalized_width)}
