import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import gymnasium as gym

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)
os.chdir(WORKSPACE_DIR)

import envs.tracking_envs
from envs.wrappers import (
    NoiseWrapper, DistractorWrapper, ViewpointWrapper, 
    OcclusionWrapper, BlurWrapper, CompoundShiftWrapper, DataAugmentationWrapper
)
from baselines.rmr_ppo import RMRPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.eval_pipeline import evaluate_policy_canonical, compute_stat_ci95
from utils.stats import welch_ttest

SEEDS = [0, 42, 100, 123, 999, 1001, 2024, 3407]
TOTAL_STEPS = 500000
MANIFEST_PATH = "results/tables/rep_distance_manifest.json"

SHIFT_CONDITIONS = [
    ("Clean", None, {}),
    ("Noise_Std_0.10", NoiseWrapper, {"noise_std": 0.10}),
    ("Noise_Std_0.20", NoiseWrapper, {"noise_std": 0.20}),
    ("Distractor_N2", DistractorWrapper, {"num_distractors": 2}),
    ("Viewpoint_15deg", ViewpointWrapper, {"max_angle": 15}),
    ("Occlusion_15pct", OcclusionWrapper, {"occlusion_ratio": 0.15}),
    ("Blur_K7", BlurWrapper, {"kernel_size": 7}),
    ("Compound_Level3", CompoundShiftWrapper, {"severity_level": 3})
]

def init_manifest():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    manifest_entries = []
    for algo in ["PPO_Baseline", "RMR_PPO"]:
        for seed in SEEDS:
            run_id = f"{algo}_s{seed}"
            manifest_entries.append({
                "Run_ID": run_id,
                "Algorithm": algo,
                "Seed": seed,
                "Total_Steps": TOTAL_STEPS,
                "Status": "PENDING",
                "Log_File_Path": f"results/logs/{run_id}.log",
                "Checkpoint_Path": f"results/models/{run_id}.zip",
                "Convergence_Status": "PENDING",
                "Mean_CLE_Noise020": None,
                "Success_Rate_Noise020": None,
                "Euclidean_Distance_Noise020": None,
                "Relative_Variation_Last10": None,
                "Rolling_Std_Last10": None
            })
            
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_entries, f, indent=2)
    print(f"Initialized manifest skeleton with 16 PENDING runs at {MANIFEST_PATH}")

def update_manifest_entry(run_id, updates):
    if not os.path.exists(MANIFEST_PATH):
        init_manifest()
    with open(MANIFEST_PATH, "r") as f:
        entries = json.load(f)
    for entry in entries:
        if entry["Run_ID"] == run_id:
            entry.update(updates)
            break
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)

def check_convergence(eval_history):
    """
    Checks if training curve has plateaued over final 10% window.
    Criterion: relative variation <= 0.05 and rolling std <= 0.03.
    """
    if len(eval_history) < 4:
        return False, 1.0, 1.0
        
    n_evals = len(eval_history)
    window_size = max(1, int(n_evals * 0.10))
    
    last_window = eval_history[-window_size:]
    prev_window = eval_history[-2*window_size:-window_size] if len(eval_history) >= 2*window_size else eval_history[:-window_size]
    
    r_last = np.mean([m['success_rate'] for m in last_window])
    r_prev = np.mean([m['success_rate'] for m in prev_window])
    
    rel_var = float(abs(r_last - r_prev) / max(r_prev, 0.01))
    rolling_std = float(np.std([m['success_rate'] for m in last_window]))
    
    is_converged = bool(rel_var <= 0.05 and rolling_std <= 0.03)
    return is_converged, rel_var, rolling_std

def run_experiment_pipeline(env_id="SingleObjectTracking-v0"):
    print("=== STARTING 8.0M-STEP REPRESENTATION DISTANCE EXPERIMENT (16 RUNS) ===")
    init_manifest()
    
    os.makedirs("results/logs", exist_ok=True)
    all_eval_rows = []
    
    for algo in ["PPO_Baseline", "RMR_PPO"]:
        for seed in SEEDS:
            run_id = f"{algo}_s{seed}"
            print(f"\n=======================================================")
            print(f"   STARTING RUN: {run_id} ({TOTAL_STEPS} steps)")
            print(f"=======================================================")
            
            start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            update_manifest_entry(run_id, {
                "Status": "RUNNING",
                "Start_Timestamp": start_iso
            })
            
            try:
                # Setup vector environment
                if algo == "PPO_Baseline":
                    vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                    model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, learning_rate=3e-4)
                else: # RMR_PPO
                    aug_env_fn = lambda: DataAugmentationWrapper(gym.make(env_id))
                    vec_env = make_vec_env(aug_env_fn, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                    model = RMRPPO("CnnPolicy", vec_env, verbose=0, seed=seed, lambda_rep=0.05, n_steps=512, learning_rate=3e-4)
                    
                # Interleaved training and periodic evaluation checkpoints
                eval_history = []
                chunk_steps = 50000 # 10 evaluation checkpoints across 500k steps
                
                for step_idx in range(0, TOTAL_STEPS, chunk_steps):
                    model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
                    current_steps = step_idx + chunk_steps
                    
                    # Checkpoint eval on clean env
                    eval_metrics = evaluate_policy_canonical(model, env_id, n_episodes=10, seed=seed)
                    eval_history.append(eval_metrics)
                    print(f"[{run_id}] Step {current_steps}/{TOTAL_STEPS}: Success = {eval_metrics['success_rate']*100:.2f}%, CLE = {eval_metrics['mean_cle']:.2f} px")
                    
                vec_env.close()
                model_save_path = f"results/models/{run_id}.zip"
                model.save(model_save_path)
                
                # Check convergence
                is_converged, rel_var, rolling_std = check_convergence(eval_history)
                conv_status = "CONVERGED" if is_converged else "UNCONVERGED — Excluded"
                print(f"[{run_id}] Convergence Check -> {conv_status} (RelVar: {rel_var:.4f}, RollingStd: {rolling_std:.4f})")
            except Exception as e:
                print(f"ERROR: Run {run_id} encountered exception/divergence: {e}")
                update_manifest_entry(run_id, {
                    "Status": "FAILED",
                    "Convergence_Status": "UNCONVERGED — Excluded",
                    "Error_Message": str(e)
                })
                continue

            
            # Collect shift evaluations & feature embeddings
            feature_extractor = model.policy.features_extractor
            feature_extractor.eval()
            
            env_clean = gym.make(env_id)
            clean_feats = []
            obs_c, _ = env_clean.reset(seed=seed * 100)
            for _ in range(50):
                action, _ = model.predict(obs_c, deterministic=True)
                obs_t = torch.as_tensor(obs_c).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    clean_feats.append(feature_extractor(obs_t).numpy()[0])
                obs_c, _, term, trunc, _ = env_clean.step(action)
                if term or trunc:
                    obs_c, _ = env_clean.reset()
            env_clean.close()
            clean_centroid = np.mean(clean_feats, axis=0)
            
            noise020_cle, noise020_succ, noise020_deuc = None, None, None
            
            for shift_name, wrapper_cls, wrapper_kwargs in SHIFT_CONDITIONS:
                if wrapper_cls is None:
                    metrics = evaluate_policy_canonical(model, env_id, n_episodes=15, seed=seed)
                    d_euc = 0.0
                else:
                    metrics = evaluate_policy_canonical(model, env_id, n_episodes=15, seed=seed, wrapper_cls=wrapper_cls, wrapper_kwargs=wrapper_kwargs)
                    
                    env_s = wrapper_cls(gym.make(env_id), **wrapper_kwargs)
                    shifted_feats = []
                    obs_s, _ = env_s.reset(seed=seed * 100)
                    for _ in range(50):
                        act_s, _ = model.predict(obs_s, deterministic=True)
                        obs_st = torch.as_tensor(obs_s).unsqueeze(0).float() / 255.0
                        with torch.no_grad():
                            shifted_feats.append(feature_extractor(obs_st).numpy()[0])
                        obs_s, _, term, trunc, _ = env_s.step(act_s)
                        if term or trunc:
                            obs_s, _ = env_s.reset()
                    env_s.close()
                    shifted_centroid = np.mean(shifted_feats, axis=0)
                    d_euc = float(np.linalg.norm(clean_centroid - shifted_centroid))
                    
                if shift_name == "Noise_Std_0.20":
                    noise020_cle = metrics['mean_cle']
                    noise020_succ = metrics['success_rate']
                    noise020_deuc = d_euc
                    
                all_eval_rows.append({
                    "Run_ID": run_id,
                    "Algorithm": algo,
                    "Seed": seed,
                    "Shift_Condition": shift_name,
                    "Mean_CLE": metrics['mean_cle'],
                    "Success_Rate": metrics['success_rate'],
                    "Euclidean_Distance": d_euc,
                    "Convergence_Status": conv_status
                })
                
            update_manifest_entry(run_id, {
                "Status": "COMPLETED",
                "Convergence_Status": conv_status,
                "Mean_CLE_Noise020": noise020_cle,
                "Success_Rate_Noise020": noise020_succ,
                "Euclidean_Distance_Noise020": noise020_deuc,
                "Relative_Variation_Last10": rel_var,
                "Rolling_Std_Last10": rolling_std
            })
            
    df_results = pd.DataFrame(all_eval_rows)
    df_results.to_csv("results/tables/rep_distance_full_eval_results.csv", index=False)
    print("\n=== EXPERIMENT COMPLETE! FULL EVALUATION RESULTS SAVED ===")

if __name__ == "__main__":
    run_experiment_pipeline()
