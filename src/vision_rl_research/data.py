"""Episode-table schema and validation."""

import csv
import math
from typing import Dict, Iterable, List, Sequence, Tuple


CORE_COLUMNS = (
    "agent",
    "environment",
    "training_seed",
    "shift_family",
    "severity",
    "episode_seed",
    "episode_return",
    "success",
)


class EpisodeDataError(ValueError):
    """Raised when an episode table is invalid or incomplete."""


def read_episode_table(
    path: str, required_predictors: Sequence[str] = ()
) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as episode_file:
        reader = csv.DictReader(episode_file)
        if reader.fieldnames is None:
            raise EpisodeDataError("Episode table has no header.")
        rows = list(reader)
        columns = list(reader.fieldnames)
    validate_episode_rows(rows, columns, required_predictors)
    return rows, columns


def validate_episode_rows(
    rows: Sequence[Dict[str, str]],
    columns: Sequence[str],
    required_predictors: Sequence[str] = (),
) -> None:
    required_columns = tuple(CORE_COLUMNS) + tuple(required_predictors)
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise EpisodeDataError(
            "Missing required columns: {}".format(", ".join(missing))
        )
    if not rows:
        raise EpisodeDataError("Episode table contains no rows.")

    unique_keys = set()
    for row_number, row in enumerate(rows, start=2):
        prefix = "Row {}: ".format(row_number)
        for column in CORE_COLUMNS:
            if row.get(column) is None or row[column] == "":
                raise EpisodeDataError(prefix + "missing value for {}.".format(column))

        try:
            training_seed = int(row["training_seed"])
            episode_seed = int(row["episode_seed"])
            severity = float(row["severity"])
            episode_return = float(row["episode_return"])
            success = int(row["success"])
        except (TypeError, ValueError) as error:
            raise EpisodeDataError(prefix + "invalid numeric value: {}".format(error))

        if not math.isfinite(severity) or not math.isfinite(episode_return):
            raise EpisodeDataError(prefix + "severity and return must be finite.")
        if not 0.0 <= severity <= 1.0:
            raise EpisodeDataError(prefix + "severity must be normalized to [0, 1].")
        if success not in (0, 1) or str(success) != row["success"]:
            raise EpisodeDataError(prefix + "success must be encoded as 0 or 1.")

        key = (
            row["agent"],
            row["environment"],
            training_seed,
            row["shift_family"],
            severity,
            episode_seed,
        )
        if key in unique_keys:
            raise EpisodeDataError(prefix + "duplicate episode identity.")
        unique_keys.add(key)

        for column in columns:
            if column in CORE_COLUMNS:
                continue
            if row.get(column, "") == "":
                if column in required_predictors:
                    raise EpisodeDataError(
                        prefix + "missing predictor value for {}.".format(column)
                    )
                continue
            try:
                value = float(row[column])
            except ValueError as error:
                raise EpisodeDataError(
                    prefix + "{} must be numeric: {}".format(column, error)
                )
            if not math.isfinite(value):
                raise EpisodeDataError(prefix + "{} must be finite.".format(column))


def validate_episode_completeness(
    rows: Sequence[Dict[str, str]], protocol: Dict[str, object]
) -> None:
    expected_total = int(protocol["expected_episodes"])
    if len(rows) != expected_total:
        raise EpisodeDataError(
            "Expected {} episodes, found {}.".format(expected_total, len(rows))
        )

    agents = protocol["agents"]
    environments = protocol["environments"]
    families = protocol["shift_families"]
    seeds = protocol["training_seeds"]
    severity_levels = int(protocol["severity_levels"])
    episodes_per_level = int(protocol["episodes_per_level"])
    if episodes_per_level % len(seeds) != 0:
        raise EpisodeDataError(
            "episodes_per_level must divide evenly across training_seeds."
        )
    episodes_per_seed = episodes_per_level // len(seeds)

    grouped: Dict[Tuple[str, str, str, float], Dict[int, int]] = {}
    for row in rows:
        key = (
            row["agent"],
            row["environment"],
            row["shift_family"],
            float(row["severity"]),
        )
        seed_counts = grouped.setdefault(key, {})
        seed = int(row["training_seed"])
        seed_counts[seed] = seed_counts.get(seed, 0) + 1

    for agent in agents:
        for environment in environments:
            for family in families:
                cell_groups = [
                    (severity, counts)
                    for (row_agent, row_environment, row_family, severity), counts
                    in grouped.items()
                    if (row_agent, row_environment, row_family)
                    == (agent, environment, family)
                ]
                if len(cell_groups) != severity_levels:
                    raise EpisodeDataError(
                        "{} / {} / {} has {} severity levels; expected {}.".format(
                            agent, environment, family, len(cell_groups), severity_levels
                        )
                    )
                for severity, seed_counts in cell_groups:
                    if set(seed_counts) != set(seeds):
                        raise EpisodeDataError(
                            "{} / {} / {} severity {} does not include every training seed."
                            .format(agent, environment, family, severity)
                        )
                    if any(count != episodes_per_seed for count in seed_counts.values()):
                        raise EpisodeDataError(
                            "{} / {} / {} severity {} must have {} episodes per seed."
                            .format(
                                agent,
                                environment,
                                family,
                                severity,
                                episodes_per_seed,
                            )
                        )


def validate_predictor_columns(
    columns: Iterable[str], predictors: Sequence[str]
) -> None:
    available = set(columns)
    missing = [predictor for predictor in predictors if predictor not in available]
    if missing:
        raise EpisodeDataError(
            "Missing predictor columns: {}".format(", ".join(missing))
        )
