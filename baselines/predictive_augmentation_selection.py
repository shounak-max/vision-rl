import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from scipy import stats

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)
os.chdir(WORKSPACE_DIR)

import envs.tracking_envs
from envs.wrappers import (
    CompoundShiftWrapper, 
    RandomShiftWrapper, ColorJitterWrapper, CutoutWrapper, BlurWrapper
)
from utils.eval_pipeline import evaluate_policy_canonical

def compute_centroid_distance(clean_feats, shifted_feats):
    clean_centroid = np.mean(clean_feats, axis=0)
    shifted_centroid = np.mean(shifted_feats, axis=0)
    return float(np.linalg.norm(clean_centroid - shifted_centroid))

def run_predictive_selection():
    print("=== PREDICTIVE AUGMENTATION SELECTION EXPERIMENT ===")
    os.makedirs("results/tables", exist_ok=True)
    
    env_id = "SingleObjectTracking-v0"
    seed = 42
    
    # 1. Ensure we have a trained baseline model
    model_path = f"results/models/PPO_Standard_{env_id}_s{seed}.zip"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Training a quick baseline...")
        os.makedirs("results/models", exist_ok=True)
        vec_env = gym.make(env_id)
        model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512)
        model.learn(total_timesteps=100000)
        model.save(model_path)
    else:
        model = PPO.load(model_path)
        
    feature_extractor = model.policy.features_extractor
    feature_extractor.eval()
    
    # Target Held-Out Perturbation
    target_shift_cls = CompoundShiftWrapper
    target_shift_kwargs = {"severity_level": 3}
    
    # Candidate Test-Time Augmentations (TTA)
    candidates = [
        ("No_Augmentation", None, {}),
        ("Color_Jitter", ColorJitterWrapper, {"brightness": 0.3, "contrast": 0.3}),
        ("Random_Shift", RandomShiftWrapper, {"max_shift": 6}),
        ("Cutout", CutoutWrapper, {"cutout_ratio": 0.2}),
        ("Blur", BlurWrapper, {"kernel_size": 5})
    ]
    
    # 2. Collect Clean Centroid
    env_clean = gym.make(env_id)
    clean_feats = []
    obs_c, _ = env_clean.reset(seed=seed)
    for _ in range(200):
        action, _ = model.predict(obs_c, deterministic=True)
        obs_t = torch.as_tensor(obs_c).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            clean_feats.append(feature_extractor(obs_t).numpy()[0])
        obs_c, _, term, trunc, _ = env_clean.step(action)
        if term or trunc:
            obs_c, _ = env_clean.reset()
    env_clean.close()
    
    # 3. Proxy Ranking (Computationally Cheap)
    print("\n--- Phase 1: Proxy Ranking (Representation Distance) ---")
    proxy_results = []
    start_time_proxy = time.time()
    
    for aug_name, aug_cls, aug_kwargs in candidates:
        # Wrap target shift
        env_shifted = target_shift_cls(gym.make(env_id), **target_shift_kwargs)
        # Apply augmentation on top
        if aug_cls is not None:
            env_aug = aug_cls(env_shifted, **aug_kwargs)
        else:
            env_aug = env_shifted
            
        shifted_feats = []
        obs_s, _ = env_aug.reset(seed=seed)
        # Only collect a few observations WITHOUT running full evaluation episodes
        for _ in range(200):
            action, _ = model.predict(obs_s, deterministic=True)
            obs_st = torch.as_tensor(obs_s).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                shifted_feats.append(feature_extractor(obs_st).numpy()[0])
            obs_s, _, term, trunc, _ = env_aug.step(action)
            if term or trunc:
                obs_s, _ = env_aug.reset()
        env_aug.close()
        
        dist = compute_centroid_distance(clean_feats, shifted_feats)
        proxy_results.append({"Augmentation": aug_name, "Rep_Distance": dist})
        print(f"  {aug_name} -> Distance: {dist:.3f}")
        
    proxy_time = time.time() - start_time_proxy
    
    # Rank by lowest distance
    proxy_df = pd.DataFrame(proxy_results).sort_values("Rep_Distance")
    proxy_df["Proxy_Rank"] = range(1, len(candidates) + 1)
    
    # 4. Ground-Truth Ranking (Computationally Expensive Rollouts)
    print("\n--- Phase 2: Ground-Truth Ranking (Full RL Rollouts) ---")
    gt_results = []
    start_time_gt = time.time()
    
    for aug_name, aug_cls, aug_kwargs in candidates:
        def make_eval_env():
            env_shifted = target_shift_cls(gym.make(env_id), **target_shift_kwargs)
            if aug_cls is not None:
                return aug_cls(env_shifted, **aug_kwargs)
            return env_shifted
            
        # We need to adapt evaluate_policy_canonical to use an env factory or pre-wrapped env
        # For simplicity, we just run the rollouts here
        env_eval = make_eval_env()
        successes = []
        returns = []
        for ep in range(30): # 30 full episodes for accurate GT
            obs, info = env_eval.reset(seed=seed*100 + ep)
            ep_ret = 0
            ep_success = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env_eval.step(action)
                ep_ret += reward
                if info.get('sparse_reward', 0) > 0 or info.get('success', 0) > 0:
                    ep_success = 1
                done = term or trunc
            successes.append(ep_success)
            returns.append(ep_ret)
        env_eval.close()
        
        mean_succ = np.mean(successes)
        mean_ret = np.mean(returns)
        
        gt_results.append({"Augmentation": aug_name, "Success_Rate": mean_succ, "Mean_Return": mean_ret})
        print(f"  {aug_name} -> Success: {mean_succ:.2f}, Return: {mean_ret:.2f}")
        
    gt_time = time.time() - start_time_gt
    
    # Rank by highest success/return
    gt_df = pd.DataFrame(gt_results).sort_values("Mean_Return", ascending=False)
    gt_df["GT_Rank"] = range(1, len(candidates) + 1)
    
    # 5. Compare & Save
    final_df = pd.merge(proxy_df, gt_df, on="Augmentation")
    print("\n--- Final Comparison ---")
    print(final_df[["Augmentation", "Proxy_Rank", "GT_Rank", "Rep_Distance", "Mean_Return"]])
    
    spearman_rho, s_p = stats.spearmanr(final_df["Proxy_Rank"], final_df["GT_Rank"])
    print(f"\nSpearman Rank Correlation: {spearman_rho:.3f} (p={s_p:.3f})")
    print(f"Compute Time Saved: Proxy took {proxy_time:.2f}s vs Ground-Truth took {gt_time:.2f}s (Speedup: {gt_time/proxy_time:.1f}x)")
    
    final_df.to_csv("results/tables/predictive_selection.csv", index=False)
    with open("results/tables/predictive_selection_summary.json", "w") as f:
        json.dump({
            "Spearman_rho": spearman_rho,
            "Spearman_p": s_p,
            "Proxy_Time_Seconds": proxy_time,
            "Ground_Truth_Time_Seconds": gt_time,
            "Speedup_Factor": gt_time / proxy_time if proxy_time > 0 else 0
        }, f, indent=2)
        
if __name__ == "__main__":
    run_predictive_selection()
