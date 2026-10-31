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


def read_episode_table(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as episode_file:
        reader = csv.DictReader(episode_file)
        if reader.fieldnames is None:
            raise EpisodeDataError("Episode table has no header.")
        rows = list(reader)
        columns = list(reader.fieldnames)
    validate_episode_rows(rows, columns)
    return rows, columns


def validate_episode_rows(
    rows: Sequence[Dict[str, str]], columns: Sequence[str]
) -> None:
    missing = [column for column in CORE_COLUMNS if column not in columns]
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
            if column in CORE_COLUMNS or row.get(column, "") == "":
                continue
            try:
                value = float(row[column])
            except ValueError as error:
                raise EpisodeDataError(
                    prefix + "{} must be numeric: {}".format(column, error)
                )
            if not math.isfinite(value):
                raise EpisodeDataError(prefix + "{} must be finite.".format(column))


def validate_predictor_columns(
    columns: Iterable[str], predictors: Sequence[str]
) -> None:
    available = set(columns)
    missing = [predictor for predictor in predictors if predictor not in available]
    if missing:
        raise EpisodeDataError(
            "Missing predictor columns: {}".format(", ".join(missing))
        )
