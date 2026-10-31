"""Load and validate the preregistered study configuration."""

import json
from pathlib import Path
from typing import Any, Dict, List


class ProtocolError(ValueError):
    """Raised when a study configuration violates its declared protocol."""


def load_protocol(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as protocol_file:
        protocol = json.load(protocol_file)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Dict[str, Any]) -> None:
    required = (
        "agents",
        "environments",
        "shift_families",
        "severity_levels",
        "episodes_per_level",
        "training_seeds",
        "fit_seeds",
        "test_seeds",
        "expected_episodes",
        "primary_predictor",
        "predictors",
        "classification",
        "bootstrap_replicates",
        "permutation_replicates",
        "analysis_random_seed",
        "orientation_auc_margin",
        "run_gate",
    )
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ProtocolError("Missing protocol fields: {}".format(", ".join(missing)))

    fit_seeds = set(protocol["fit_seeds"])
    test_seeds = set(protocol["test_seeds"])
    all_seeds = set(protocol["training_seeds"])
    if not fit_seeds or not test_seeds:
        raise ProtocolError("Fit and test seed sets must both be non-empty.")
    if fit_seeds & test_seeds:
        raise ProtocolError("Fit and test seeds overlap.")
    if fit_seeds | test_seeds != all_seeds:
        raise ProtocolError("Fit and test seeds must partition training_seeds.")
    if protocol["primary_predictor"] not in protocol["predictors"]:
        raise ProtocolError("primary_predictor must be in predictors.")
    if not 0.0 <= protocol["orientation_auc_margin"] < 0.5:
        raise ProtocolError("orientation_auc_margin must be in [0, 0.5).")

    classification = protocol["classification"]
    expected_orderings = (
        ("correlation_weak_max", "correlation_strong_min"),
        ("auroc_low_max", "auroc_high_min"),
        ("cliff_edge_width_max", "graded_width_min"),
    )
    for lower_key, upper_key in expected_orderings:
        if classification[lower_key] >= classification[upper_key]:
            raise ProtocolError(
                "{} must be less than {}.".format(lower_key, upper_key)
            )

    expected_episodes = (
        len(protocol["agents"])
        * len(protocol["environments"])
        * len(protocol["shift_families"])
        * protocol["severity_levels"]
        * protocol["episodes_per_level"]
    )
    if protocol["expected_episodes"] != expected_episodes:
        raise ProtocolError(
            "expected_episodes is {}, but design implies {}.".format(
                protocol["expected_episodes"], expected_episodes
            )
        )

    for key in ("bootstrap_replicates", "permutation_replicates"):
        if protocol[key] < 1:
            raise ProtocolError("{} must be positive.".format(key))

    gate = protocol["run_gate"]
    analysis_blockers = gate.get("analysis_design_blockers", [])
    if gate["full_study_enabled"] and (
        gate["remaining_blockers"] or analysis_blockers
    ):
        raise ProtocolError("Full study cannot be enabled while blockers remain.")


def run_gate_blockers(protocol: Dict[str, Any]) -> List[str]:
    """Return blockers that prevent starting the confirmatory full study."""
    gate = protocol["run_gate"]
    blockers = list(gate["remaining_blockers"])
    blockers.extend(gate.get("analysis_design_blockers", []))
    for environment, definition in gate["success_definitions"].items():
        if definition["status"] == "unresolved":
            blockers.append(
                "Success definition for {} is unresolved.".format(environment)
            )
    for key in (
        "encoder_ablation_pass_criterion",
        "random_policy_reference",
        "feature_window_length",
    ):
        if gate[key] is None:
            blockers.append("{} is not frozen.".format(key))
    if not gate["full_study_enabled"]:
        blockers.append("full_study_enabled is false.")
    return blockers


def default_protocol_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "config" / "study.json")
