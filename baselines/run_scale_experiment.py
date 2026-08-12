import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from scipy import stats

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)
os.chdir(WORKSPACE_DIR)

import envs.tracking_envs
from envs.wrappers import (
    NoiseWrapper, DistractorWrapper, ViewpointWrapper, 
    OcclusionWrapper, BlurWrapper, CompoundShiftWrapper, DataAugmentationWrapper,
    RandomShiftWrapper, ColorJitterWrapper, CutoutWrapper
)
from utils.eval_pipeline import evaluate_policy_canonical

SEEDS = [0, 42, 100, 123, 999]
ENVS = ["procgen:procgen-coinrun-v0", "MultiStageNavigation-v0"]

def get_total_steps(env_id):
    if "procgen" in env_id:
        return 10000000  # 10M steps for procgen
    return 500000  # 500k for custom env

# 10 Perturbation Conditions
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

def compute_centroid_distance(clean_feats, shifted_feats):
    clean_centroid = np.mean(clean_feats, axis=0)
    shifted_centroid = np.mean(shifted_feats, axis=0)
    return float(np.linalg.norm(clean_centroid - shifted_centroid))

def run_experiment():
    print("=== STARTING SCALE EXPERIMENT ===")
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    
    all_eval_rows = []

    for env_id in ENVS:
        for algo in ["PPO_Standard", "DrQ_PPO"]:
            for seed in SEEDS:
                total_steps = get_total_steps(env_id)
                run_id = f"{algo}_{env_id}_s{seed}"
                print(f"\n=======================================================")
                print(f"   STARTING RUN: {run_id} ({total_steps} steps)")
                print(f"=======================================================")
                
                model_save_path = f"results/models/{run_id}.zip"
                
                if not os.path.exists(model_save_path):
                    # Setup vector environment
                    if algo == "PPO_Standard":
                        vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                    else: # DrQ_PPO baseline
                        aug_env_fn = lambda: DataAugmentationWrapper(gym.make(env_id))
                        vec_env = make_vec_env(aug_env_fn, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                        
                    model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, learning_rate=3e-4)
                    model.learn(total_timesteps=total_steps)
                    model.save(model_save_path)
                    vec_env.close()
                else:
                    print(f"Loaded existing model {model_save_path}")
                    model = PPO.load(model_save_path)
                
                feature_extractor = model.policy.features_extractor
                feature_extractor.eval()
                
                # Clean Features
                env_clean = gym.make(env_id)
                clean_feats = []
                obs_c, _ = env_clean.reset(seed=seed * 100)
                for _ in range(100):
                    action, _ = model.predict(obs_c, deterministic=True)
                    obs_t = torch.as_tensor(obs_c).unsqueeze(0).float() / 255.0
                    with torch.no_grad():
                        clean_feats.append(feature_extractor(obs_t).numpy()[0])
                    obs_c, _, term, trunc, _ = env_clean.step(action)
                    if term or trunc:
                        obs_c, _ = env_clean.reset()
                env_clean.close()
                clean_feats = np.array(clean_feats)
                
                for shift_name, wrapper_cls, wrapper_kwargs in SHIFT_CONDITIONS:
                    if wrapper_cls is None:
                        # Clean eval
                        metrics = evaluate_policy_canonical(model, env_id, n_episodes=15, seed=seed)
                        d_euc = 0.0
                    else:
                        metrics = evaluate_policy_canonical(model, env_id, n_episodes=15, seed=seed, wrapper_cls=wrapper_cls, wrapper_kwargs=wrapper_kwargs)
                        env_s = wrapper_cls(gym.make(env_id), **wrapper_kwargs)
                        shifted_feats = []
                        obs_s, _ = env_s.reset(seed=seed * 100)
                        for _ in range(100):
                            act_s, _ = model.predict(obs_s, deterministic=True)
                            obs_st = torch.as_tensor(obs_s).unsqueeze(0).float() / 255.0
                            with torch.no_grad():
                                shifted_feats.append(feature_extractor(obs_st).numpy()[0])
                            obs_s, _, term, trunc, _ = env_s.step(act_s)
                            if term or trunc:
                                obs_s, _ = env_s.reset()
                        env_s.close()
                        d_euc = compute_centroid_distance(clean_feats, np.array(shifted_feats))
                        
                    all_eval_rows.append({
                        "Env": env_id,
                        "Algorithm": algo,
                        "Seed": seed,
                        "Shift_Condition": shift_name,
                        "Success_Rate": metrics.get('success_rate', 0.0),
                        "Mean_Return": metrics.get('mean_return', 0.0),
                        "Representation_Distance": d_euc
                    })
                    print(f"  [{shift_name}] RepDist: {d_euc:.3f} | Perf: {metrics.get('success_rate', metrics.get('mean_return', 0.0)):.2f}")

    df_results = pd.DataFrame(all_eval_rows)
    df_results.to_csv("results/tables/scale_experiment_results.csv", index=False)
    print("\n=== SAVED ALL RESULTS ===")

    # Compute Correlations
    for env_id in ENVS:
        df_env = df_results[df_results["Env"] == env_id]
        if len(df_env) == 0: continue
        
        # We compute correlation between distance and performance degradation
        # First get clean performance per seed/algo
        clean_perf = df_env[df_env["Shift_Condition"] == "Clean"].set_index(["Algorithm", "Seed"])
        
        # Calculate degradation
        df_shift = df_env[df_env["Shift_Condition"] != "Clean"].copy()
        
        # Use success rate if it's there and > 0 for clean, else mean return
        perfs, dists = [], []
        for idx, row in df_shift.iterrows():
            c_perf = clean_perf.loc[(row["Algorithm"], row["Seed"])]
            perf_metric = "Success_Rate" if c_perf["Success_Rate"] > 0 else "Mean_Return"
            perf_deg = c_perf[perf_metric] - row[perf_metric]
            perfs.append(perf_deg)
            dists.append(row["Representation_Distance"])
            
        pearson_r, p_p = stats.pearsonr(dists, perfs)
        spearman_rho, s_p = stats.spearmanr(dists, perfs)
        
        print(f"\n[Correlation] {env_id} (Distance vs Performance Degradation):")
        print(f"  Pearson r = {pearson_r:.4f} (p={p_p:.4e})")
        print(f"  Spearman rho = {spearman_rho:.4f} (p={s_p:.4e})")

        with open(f"results/tables/correlation_{env_id}.json", "w") as f:
            json.dump({"Pearson_r": pearson_r, "Pearson_p": p_p, "Spearman_rho": spearman_rho, "Spearman_p": s_p}, f, indent=2)

if __name__ == "__main__":
    run_experiment()
