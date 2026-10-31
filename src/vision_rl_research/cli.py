"""Command-line entry points for protocol and episode-table validation."""

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from vision_rl_research.analysis import (
    blocked_regime_permutation_test,
    classify_correlation_auroc,
    classify_regime,
    correlation_metrics,
    evaluate_predictor,
    family_seed_split,
    seed_blocked_bootstrap_ci,
)
from vision_rl_research.data import read_episode_table, validate_predictor_columns
from vision_rl_research.protocol import (
    ProtocolError,
    default_protocol_path,
    load_protocol,
    run_gate_blockers,
)


def _aggregate_success(rows: List[Dict[str, str]]) -> List[Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["severity"])].append(float(row["success"]))
    return [
        (severity, float(np.mean(successes)))
        for severity, successes in sorted(grouped.items())
    ]


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


def analyze_table(input_path: str, output_path: str, protocol_path: str) -> None:
    protocol = load_protocol(protocol_path)
    rows, columns = read_episode_table(input_path)
    predictors = protocol["predictors"]
    validate_predictor_columns(columns, predictors)
    families = sorted({row["shift_family"] for row in rows})
    environments = sorted({row["environment"] for row in rows})
    agents = sorted({row["agent"] for row in rows})
    fit_seeds = set(protocol["fit_seeds"])
    regimes_by_cell = {}

    results = {
        "protocol_version": protocol["protocol_version"],
        "input": input_path,
        "episode_count": len(rows),
        "held_out_family_results": [],
        "regime_classifications": [],
        "primary_test": None,
        "run_gate_blockers": run_gate_blockers(protocol),
    }

    for agent in agents:
        for environment in environments:
            for family in families:
                cell_rows = [
                    row for row in rows
                    if row["agent"] == agent
                    and row["environment"] == environment
                    and row["shift_family"] == family
                    and int(row["training_seed"]) in fit_seeds
                ]
                if not cell_rows:
                    continue
                regime = classify_regime(
                    _aggregate_success(cell_rows),
                    protocol["classification"],
                )
                cell = (agent, environment, family)
                regimes_by_cell[cell] = regime
                results["regime_classifications"].append({
                    "agent": agent,
                    "environment": environment,
                    "shift_family": family,
                    "regime_fit_seeds": sorted(fit_seeds),
                    "regime_fit_episodes": len(cell_rows),
                    **regime,
                })

    for family in families:
        for agent in agents:
            for environment in environments:
                cell_rows = [
                    row for row in rows
                    if row["agent"] == agent
                    and row["environment"] == environment
                ]
                if not cell_rows:
                    continue
                fit_rows, test_rows = family_seed_split(
                    cell_rows,
                    family,
                    protocol["fit_seeds"],
                    protocol["test_seeds"],
                )
                cell = (agent, environment, family)
                regime = regimes_by_cell[cell]
                for predictor in predictors:
                    prediction = evaluate_predictor(fit_rows, test_rows, predictor)
                    correlation = correlation_metrics(test_rows, predictor)
                    classification = classify_correlation_auroc(
                        correlation["spearman_rho"],
                        prediction["auroc"],
                        protocol["classification"],
                    )
                    results["held_out_family_results"].append({
                        "agent": agent,
                        "environment": environment,
                        "held_out_family": family,
                        "predictor": predictor,
                        "fit_episodes": len(fit_rows),
                        "test_episodes": len(test_rows),
                        "regime": regime["regime"],
                        "correlation": correlation,
                        "prediction": prediction,
                        "correlation_auroc_classification": classification,
                    })

    primary_predictor = protocol["primary_predictor"]
    primary_cells = [
        {
            "agent": result["agent"],
            "environment": result["environment"],
            "regime": result["regime"],
            "discordant": result["correlation_auroc_classification"]["discordant"],
            "correlation_class": result[
                "correlation_auroc_classification"
            ]["correlation_class"],
            "auroc_class": result[
                "correlation_auroc_classification"
            ]["auroc_class"],
        }
        for result in results["held_out_family_results"]
        if result["predictor"] == primary_predictor
    ]
    cliff_cells = [
        cell for cell in primary_cells if cell["regime"] == "cliff_edge"
    ]
    h1_evaluable = [
        cell for cell in cliff_cells
        if cell["correlation_class"] in ("weak", "strong")
        and cell["auroc_class"] in ("low", "high")
    ]
    h1_supporting = [
        cell for cell in h1_evaluable
        if cell["correlation_class"] == "weak" and cell["auroc_class"] == "high"
    ]
    results["hypothesis_summaries"] = {
        "H1": {
            "evaluable_cliff_edge_cells": len(h1_evaluable),
            "weak_correlation_high_auroc_cells": len(h1_supporting),
            "supporting_cell_rate": (
                float(len(h1_supporting) / len(h1_evaluable))
                if h1_evaluable else None
            ),
            "indeterminate_cliff_edge_cells": len(cliff_cells) - len(h1_evaluable),
        },
        "H3": [],
    }

    paired_aurocs: Dict[str, List[float]] = {
        predictor: [] for predictor in predictors if predictor != primary_predictor
    }
    auroc_by_cell_predictor = {
        (
            result["agent"],
            result["environment"],
            result["held_out_family"],
            result["predictor"],
        ): result["prediction"]["auroc"]
        for result in results["held_out_family_results"]
    }
    for (agent, environment, family, predictor), auroc in auroc_by_cell_predictor.items():
        if predictor == primary_predictor:
            continue
        baseline = auroc_by_cell_predictor.get(
            (agent, environment, family, primary_predictor)
        )
        if baseline is not None:
            paired_aurocs[predictor].append(auroc - baseline)
    for predictor, differences in paired_aurocs.items():
        results["hypothesis_summaries"]["H3"].append({
            "predictor": predictor,
            "baseline_predictor": primary_predictor,
            "paired_cells": len(differences),
            "mean_held_out_auroc_difference": (
                float(np.mean(differences)) if differences else None
            ),
            "median_held_out_auroc_difference": (
                float(np.median(differences)) if differences else None
            ),
            "fraction_cells_better_than_baseline": (
                float(np.mean(np.asarray(differences) > 0.0))
                if differences else None
            ),
            "interpretation": "exploratory_descriptive",
        })

    observed_regimes = {
        cell["regime"]
        for cell in primary_cells
        if cell["regime"] in ("cliff_edge", "graded")
        and cell["discordant"] in (True, False)
    }
    if observed_regimes == {"cliff_edge", "graded"}:
        primary_test = blocked_regime_permutation_test(
            primary_cells,
            protocol["permutation_replicates"],
            protocol["analysis_random_seed"],
        )
        try:
            bootstrap = seed_blocked_bootstrap_ci(
                rows,
                regimes_by_cell,
                primary_predictor,
                protocol["classification"],
                protocol["test_seeds"],
                protocol["bootstrap_replicates"],
                protocol["analysis_random_seed"],
            )
            primary_test["seed_blocked_bootstrap_95_ci"] = bootstrap
        except ValueError as error:
            primary_test["seed_blocked_bootstrap_95_ci"] = {
                "status": "not_estimable",
                "reason": str(error),
            }
        primary_test["status"] = "estimated"
        results["primary_test"] = primary_test
    else:
        results["primary_test"] = {
            "status": "not_estimable",
            "reason": (
                "The primary contrast requires evaluable cliff-edge and graded "
                "cells; the available episode table does not contain both."
            ),
            "evaluable_regimes": sorted(observed_regimes),
        }

    _write_json(output_path, results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-rl-study",
        description="Validate and analyze the visual-RL failure-prediction study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_protocol_parser = subparsers.add_parser(
        "validate-protocol", help="Validate the frozen study configuration."
    )
    validate_protocol_parser.add_argument(
        "--protocol", default=default_protocol_path()
    )

    validate_episodes_parser = subparsers.add_parser(
        "validate-episodes", help="Validate the per-episode CSV schema and values."
    )
    validate_episodes_parser.add_argument("input")
    validate_episodes_parser.add_argument("--protocol", default=default_protocol_path())

    analyze_parser = subparsers.add_parser(
        "analyze", help="Run leave-one-family-out prediction and regime analysis."
    )
    analyze_parser.add_argument("input")
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.add_argument("--protocol", default=default_protocol_path())
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-protocol":
            protocol = load_protocol(args.protocol)
            blockers = run_gate_blockers(protocol)
            print("Protocol valid.")
            if blockers:
                print("Full-study run gate is closed:")
                for blocker in blockers:
                    print("- " + blocker)
            else:
                print("Full-study run gate is open.")
            return 0

        if args.command == "validate-episodes":
            protocol = load_protocol(args.protocol)
            rows, columns = read_episode_table(args.input)
            validate_predictor_columns(columns, protocol["predictors"])
            print("Valid episode table: {} episodes.".format(len(rows)))
            return 0

        analyze_table(args.input, args.output, args.protocol)
        print("Analysis written to {}.".format(args.output))
        return 0
    except (OSError, ValueError, ProtocolError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
