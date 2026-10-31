import csv
import json
import os
import tempfile
import unittest

from vision_rl_research.analysis import (
    blocked_regime_permutation_test,
    classify_correlation_auroc,
    classify_regime,
    evaluate_predictor,
    family_seed_split,
    seed_blocked_bootstrap_ci,
)
from vision_rl_research.data import EpisodeDataError, validate_episode_rows
from vision_rl_research.cli import analyze_table
from vision_rl_research.protocol import (
    ProtocolError,
    default_protocol_path,
    load_protocol,
    validate_protocol,
)


class ProtocolTests(unittest.TestCase):
    def test_checked_in_protocol_is_valid_and_full_study_is_blocked(self):
        protocol = load_protocol(default_protocol_path())
        self.assertEqual(protocol["expected_episodes"], 14400)
        self.assertTrue(protocol["run_gate"]["remaining_blockers"])
        self.assertFalse(protocol["run_gate"]["full_study_enabled"])

    def test_rejects_seed_leakage(self):
        protocol = {
            "agents": ["a"],
            "environments": ["e"],
            "shift_families": ["f"],
            "severity_levels": 1,
            "episodes_per_level": 1,
            "training_seeds": [1],
            "fit_seeds": [1],
            "test_seeds": [1],
            "expected_episodes": 1,
            "primary_predictor": "p",
            "predictors": ["p"],
            "classification": {
                "correlation_weak_max": 0.3,
                "correlation_strong_min": 0.5,
                "auroc_low_max": 0.6,
                "auroc_high_min": 0.8,
                "cliff_edge_width_max": 0.2,
                "graded_width_min": 0.5,
            },
            "run_gate": {"full_study_enabled": False, "remaining_blockers": []},
        }
        with self.assertRaises(ProtocolError):
            validate_protocol(protocol)


class EpisodeDataTests(unittest.TestCase):
    def test_requires_numeric_success(self):
        row = {
            "agent": "agent",
            "environment": "env",
            "training_seed": "1",
            "shift_family": "noise",
            "severity": "0.5",
            "episode_seed": "10",
            "episode_return": "2.0",
            "success": "2",
        }
        with self.assertRaises(EpisodeDataError):
            validate_episode_rows([row], list(row))

    def test_accepts_valid_episode_row(self):
        row = {
            "agent": "agent",
            "environment": "env",
            "training_seed": "1",
            "shift_family": "noise",
            "severity": "0.5",
            "episode_seed": "10",
            "episode_return": "2.0",
            "success": "1",
            "centroid_distance": "0.25",
        }
        validate_episode_rows([row], list(row))


class AnalysisTests(unittest.TestCase):
    def test_family_and_seed_split_excludes_heldout_data_from_fit(self):
        rows = [
            {"training_seed": "1", "shift_family": "noise"},
            {"training_seed": "1", "shift_family": "color"},
            {"training_seed": "6", "shift_family": "noise"},
            {"training_seed": "6", "shift_family": "color"},
        ]
        fit, test = family_seed_split(rows, "noise", [1], [6])
        self.assertEqual(fit, [rows[1]])
        self.assertEqual(test, [rows[2]])

    def test_threshold_is_fit_only_from_training_rows(self):
        fit_rows = [
            {"centroid_distance": "0.0", "success": "1"},
            {"centroid_distance": "0.1", "success": "1"},
            {"centroid_distance": "0.8", "success": "0"},
            {"centroid_distance": "1.0", "success": "0"},
        ]
        test_rows = [
            {"centroid_distance": "0.2", "success": "1"},
            {"centroid_distance": "0.9", "success": "0"},
            {"centroid_distance": "0.7", "success": "0"},
            {"centroid_distance": "0.1", "success": "1"},
        ]
        result = evaluate_predictor(fit_rows, test_rows, "centroid_distance")
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["auprc"], 1.0)
        self.assertGreater(result["threshold_fit_on_training_families"], 0.1)

    def test_regime_interpolates_crossings_and_classifies_width(self):
        points = [(0.0, 1.0), (0.1, 0.95), (0.2, 0.05), (1.0, 0.0)]
        cutoffs = {
            "cliff_edge_width_max": 0.2,
            "graded_width_min": 0.5,
        }
        result = classify_regime(points, cutoffs)
        self.assertEqual(result["regime"], "cliff_edge")
        self.assertLessEqual(result["normalized_width"], 0.2)

    def test_regime_is_unclassified_if_sweep_never_reaches_ten_percent(self):
        result = classify_regime(
            [(0.0, 1.0), (0.5, 0.8), (1.0, 0.4)],
            {"cliff_edge_width_max": 0.2, "graded_width_min": 0.5},
        )
        self.assertEqual(result["regime"], "unclassified")

    def test_correlation_auroc_categories_mark_dissociation(self):
        result = classify_correlation_auroc(
            -0.2,
            0.85,
            {
                "correlation_weak_max": 0.3,
                "correlation_strong_min": 0.5,
                "auroc_low_max": 0.6,
                "auroc_high_min": 0.8,
            },
        )
        self.assertTrue(result["discordant"])

    def test_primary_permutation_is_reproducible_and_blocked(self):
        cells = [
            {"agent": "a", "environment": "e", "regime": "cliff_edge", "discordant": True},
            {"agent": "a", "environment": "e", "regime": "cliff_edge", "discordant": True},
            {"agent": "a", "environment": "e", "regime": "graded", "discordant": False},
            {"agent": "a", "environment": "e", "regime": "graded", "discordant": False},
        ]
        result = blocked_regime_permutation_test(cells, 1000, 7)
        self.assertEqual(result["difference_cliff_minus_graded"], 1.0)
        self.assertEqual(
            result,
            blocked_regime_permutation_test(cells, 1000, 7),
        )
        self.assertAlmostEqual(result["two_sided_permutation_p"], 1 / 3, delta=0.06)

    def test_bootstrap_resamples_test_seeds_and_reports_ci(self):
        rows = []
        for family in ("cliff", "graded"):
            for seed in (6, 7):
                for predictor, success in (
                    (0.0, 1),
                    (0.1, 1),
                    (0.9, 0),
                    (1.0, 0),
                ):
                    rows.append({
                        "agent": "a",
                        "environment": "e",
                        "shift_family": family,
                        "training_seed": str(seed),
                        "centroid_distance": str(predictor),
                        "success": str(success),
                    })
        regimes = {
            ("a", "e", "cliff"): {"regime": "cliff_edge"},
            ("a", "e", "graded"): {"regime": "graded"},
        }
        result = seed_blocked_bootstrap_ci(
            rows,
            regimes,
            "centroid_distance",
            {
                "correlation_weak_max": 0.3,
                "correlation_strong_min": 0.5,
                "auroc_low_max": 0.6,
                "auroc_high_min": 0.8,
            },
            [6, 7],
            20,
            7,
        )
        self.assertEqual(result["valid_replicates"], 20)
        self.assertEqual(result["difference_ci_lower"], 0.0)
        self.assertEqual(result["difference_ci_upper"], 0.0)


class AnalysisCliTests(unittest.TestCase):
    def test_analysis_emits_preregistered_primary_test_for_both_regimes(self):
        protocol = load_protocol(default_protocol_path())
        protocol["bootstrap_replicates"] = 20
        protocol["permutation_replicates"] = 100
        families = ("viewpoint", "distractors", "noise", "color")
        severity_levels = (0.0, 0.3, 0.5, 1.0)
        success_rates = {
            "viewpoint": (1.0, 1.0, 0.0, 0.0),
            "distractors": (1.0, 1.0, 0.0, 0.0),
            "noise": (1.0, 0.8, 0.2, 0.0),
            "color": (1.0, 0.8, 0.2, 0.0),
        }
        columns = [
            "agent",
            "environment",
            "training_seed",
            "shift_family",
            "severity",
            "episode_seed",
            "episode_return",
            "success",
            *protocol["predictors"],
        ]
        rows = []
        for family in families:
            for seed in range(1, 11):
                for severity_index, severity in enumerate(severity_levels):
                    success_count = round(
                        success_rates[family][severity_index] * 30
                    )
                    for episode_index in range(3):
                        trial = (seed - 1) * 3 + episode_index
                        success = int(trial < success_count)
                        row = {
                            "agent": "agent",
                            "environment": "environment",
                            "training_seed": seed,
                            "shift_family": family,
                            "severity": severity,
                            "episode_seed": episode_index,
                            "episode_return": success,
                            "success": success,
                        }
                        for predictor_index, predictor in enumerate(
                            protocol["predictors"]
                        ):
                            if predictor == "severity":
                                continue
                            row[predictor] = (
                                severity + trial * 0.001 + predictor_index * 0.01
                            )
                        rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "episodes.csv")
            output_path = os.path.join(directory, "analysis.json")
            protocol_path = os.path.join(directory, "study.json")
            with open(input_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            with open(protocol_path, "w", encoding="utf-8") as file:
                json.dump(protocol, file)

            analyze_table(input_path, output_path, protocol_path)
            with open(output_path, "r", encoding="utf-8") as file:
                result = json.load(file)

        self.assertEqual(result["episode_count"], 480)
        self.assertEqual(result["primary_test"]["status"], "estimated")
        self.assertIn("two_sided_permutation_p", result["primary_test"])
        self.assertIn("seed_blocked_bootstrap_95_ci", result["primary_test"])
        self.assertIn("supporting_cell_rate", result["hypothesis_summaries"]["H1"])
        self.assertEqual(len(result["hypothesis_summaries"]["H3"]), 11)
        self.assertTrue(all(
            item["interpretation"] == "exploratory_descriptive"
            for item in result["hypothesis_summaries"]["H3"]
        ))
        self.assertEqual(
            result["primary_test"]["seed_blocked_bootstrap_95_ci"][
                "valid_replicates"
            ],
            protocol["bootstrap_replicates"],
        )
        self.assertEqual(
            {item["regime"] for item in result["regime_classifications"]},
            {"cliff_edge", "graded"},
        )


if __name__ == "__main__":
    unittest.main()
