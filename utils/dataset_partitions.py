"""
Canonical Train / Validation / Test partitions for benchmark-level reproducibility.
"""
from envs.wrappers import NoiseWrapper, DistractorWrapper, ViewpointWrapper

BENCHMARK_PARTITIONS = {
    "train": {
        "description": "Standard clean background canvas (84x84) with unperturbed dynamics.",
        "environments": ["SingleObjectTracking-v0", "MultiObjectTracking-v0", "ActiveTracking-v0", "MultiStageNavigation-v0"],
        "wrappers": []
    },
    "validation": {
        "description": "Mild visual shifts used for hyperparameter selection and early stopping diagnostics.",
        "shift_spectrum": [
            ("Noise (std=0.05)", NoiseWrapper, {"noise_std": 0.05}),
            ("Noise (std=0.10)", NoiseWrapper, {"noise_std": 0.10}),
            ("Distractors (N=1)", DistractorWrapper, {"num_distractors": 1}),
            ("Viewpoint (angle=10deg)", ViewpointWrapper, {"max_angle": 10})
        ]
    },
    "test": {
        "description": "Severe out-of-distribution continuous visual shifts for generalizability benchmarking.",
        "shift_spectrum": [
            ("Noise (std=0.15)", NoiseWrapper, {"noise_std": 0.15}),
            ("Noise (std=0.20)", NoiseWrapper, {"noise_std": 0.20}),
            ("Noise (std=0.30)", NoiseWrapper, {"noise_std": 0.30}),
            ("Noise (std=0.40)", NoiseWrapper, {"noise_std": 0.40}),
            ("Distractors (N=2)", DistractorWrapper, {"num_distractors": 2}),
            ("Distractors (N=3)", DistractorWrapper, {"num_distractors": 3}),
            ("Distractors (N=4)", DistractorWrapper, {"num_distractors": 4}),
            ("Viewpoint (angle=20deg)", ViewpointWrapper, {"max_angle": 20}),
            ("Viewpoint (angle=30deg)", ViewpointWrapper, {"max_angle": 30}),
            ("Viewpoint (angle=45deg)", ViewpointWrapper, {"max_angle": 45})
        ]
    }
}
