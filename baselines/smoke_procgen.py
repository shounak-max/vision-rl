import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import gym
import procgen
import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)
os.chdir(WORKSPACE_DIR)

from envs.wrappers import (
    NoiseWrapper, DistractorWrapper, ViewpointWrapper, 
    OcclusionWrapper, BlurWrapper, CompoundShiftWrapper, ProcgenGymnasiumWrapper
)

SHIFT_CONDITIONS = [
    ("Clean", None, {}),
    ("Noise_Low", NoiseWrapper, {"noise_std": 0.05}),
    ("Noise_High", NoiseWrapper, {"noise_std": 0.20}),
    ("Distractor_1", DistractorWrapper, {"num_distractors": 1}),
    ("Distractor_3", DistractorWrapper, {"num_distractors": 3}),
    ("Viewpoint_15deg", ViewpointWrapper, {"max_angle": 15}),
    ("Viewpoint_30deg", ViewpointWrapper, {"max_angle": 30}),
    ("Occlusion_15pct", OcclusionWrapper, {"occlusion_ratio": 0.15}),
    ("Blur_K7", BlurWrapper, {"kernel_size": 7}),
    ("Compound_Level2", CompoundShiftWrapper, {"severity_level": 2}),
    ("Compound_Level4", CompoundShiftWrapper, {"severity_level": 4})
]

def run_procgen_smoke_test(total_timesteps=50000, seed=42):
    print(f"=== PROCGEN SMOKE TEST ({total_timesteps} steps, Seed {seed}) ===")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    def make_env():
        return ProcgenGymnasiumWrapper("procgen:procgen-coinrun-v0")

    vec_env = DummyVecEnv([make_env for _ in range(4)])
    
    model_path = f"results/models/smoke_ppo_procgen_s{seed}.zip"
    if os.path.exists(model_path):
        print(f"Loading existing Procgen model checkpoint: {model_path}", flush=True)
        model = PPO.load(model_path)
    else:
        print("Training PPO on Procgen CoinRun (4 vectorized envs)...", flush=True)
        t0 = time.time()
        model = PPO("CnnPolicy", vec_env, verbose=1, seed=seed, n_steps=512, learning_rate=3e-4)
        model.learn(total_timesteps=total_timesteps)
        train_time = time.time() - t0
        print(f"Training Complete in {train_time:.2f}s!", flush=True)
        model.save(model_path)
    vec_env.close()

    print("\n--- Evaluating Trained Policy Across Shift Conditions ---", flush=True)
    eval_results = []

    for shift_name, wrapper_cls, wrapper_kwargs in SHIFT_CONDITIONS:
        base_env = ProcgenGymnasiumWrapper("procgen:procgen-coinrun-v0")
        if wrapper_cls is not None:
            eval_env = wrapper_cls(base_env, **wrapper_kwargs)
        else:
            eval_env = base_env

        ep_returns = []
        ep_lengths = []
        successes = []

        for ep in range(50):
            obs, info = eval_env.reset(seed=seed * 1000 + ep * 17 + 3)
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = eval_env.step(action)
                ep_ret += reward
                ep_len += 1
                done = term or trunc
            
            ep_returns.append(ep_ret)
            ep_lengths.append(ep_len)
            successes.append(1.0 if ep_ret > 0 else 0.0)

        eval_env.close()

        mean_ret = float(np.mean(ep_returns))
        std_ret = float(np.std(ep_returns))
        min_ret = float(np.min(ep_returns))
        max_ret = float(np.max(ep_returns))
        unique_returns = list(np.unique(np.round(ep_returns, 4)))
        succ_rate = float(np.mean(successes))

        print(f"[{shift_name:15s}] Mean Ret: {mean_ret:6.2f} +/- {std_ret:5.2f} | Succ: {succ_rate*100:5.1f}% | Range: [{min_ret:.2f}, {max_ret:.2f}] | Unique Returns count: {len(unique_returns)}", flush=True)

        eval_results.append({
            "Shift_Condition": shift_name,
            "Mean_Return": mean_ret,
            "Std_Return": std_ret,
            "Min_Return": min_ret,
            "Max_Return": max_ret,
            "Success_Rate": succ_rate,
            "Unique_Return_Values": unique_returns,
            "Ep_Returns": [float(r) for r in ep_returns]
        })

    out_df = pd.DataFrame(eval_results)
    out_df.to_csv("results/tables/procgen_smoke_test_results.csv", index=False)
    with open("results/tables/procgen_smoke_test_summary.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\nProcgen Smoke Test Complete! Results saved to results/tables/procgen_smoke_test_results.csv")

if __name__ == "__main__":
    run_procgen_smoke_test(total_timesteps=300000, seed=42)
