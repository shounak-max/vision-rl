import sys
import os
import gym
import procgen
import gymnasium
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from baselines.smoke_procgen import ProcgenGymnasiumWrapper
from envs.wrappers import (
    NoiseWrapper, DistractorWrapper, ViewpointWrapper, 
    OcclusionWrapper, BlurWrapper, CompoundShiftWrapper
)

def test_all():
    print("Testing base procgen...")
    base = ProcgenGymnasiumWrapper('procgen:procgen-coinrun-v0')
    obs, info = base.reset(seed=42)
    print(f"Base reset shape: {obs.shape}")

    wrappers = [
        ("Noise", NoiseWrapper, {"noise_std": 0.05}),
        ("Distractor", DistractorWrapper, {"num_distractors": 3}),
        ("Viewpoint", ViewpointWrapper, {"max_angle": 15}),
        ("Occlusion", OcclusionWrapper, {"occlusion_ratio": 0.15}),
        ("Blur", BlurWrapper, {"kernel_size": 7}),
        ("Compound", CompoundShiftWrapper, {"severity_level": 2})
    ]

    for name, wcls, kwargs in wrappers:
        print(f"Testing {name}...")
        w_env = wcls(ProcgenGymnasiumWrapper('procgen:procgen-coinrun-v0'), **kwargs)
        obs, _ = w_env.reset(seed=42)
        for _ in range(50):
            obs, r, term, trunc, _ = w_env.step(0)
            if term or trunc:
                obs, _ = w_env.reset()
        w_env.close()
        print(f"  {name} OK!")

if __name__ == "__main__":
    test_all()
