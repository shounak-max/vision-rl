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

from envs.wrappers import (
    RandomShiftWrapper, ColorJitterWrapper, CutoutWrapper, BlurWrapper
)
from baselines.run_scale_experiment import SHIFT_CONDITIONS

def compute_centroid_distance(clean_feats, shifted_feats):
    clean_centroid = np.mean(clean_feats, axis=0)
    shifted_centroid = np.mean(shifted_feats, axis=0)
    return float(np.linalg.norm(clean_centroid - shifted_centroid))

def run_predictive_selection():
    print("=== PREDICTIVE AUGMENTATION SELECTION EXPERIMENT ===")
    os.makedirs("results/tables", exist_ok=True)
    
    env_id = "SingleObjectTracking-v0"
    seed = 42
    
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
    
    # Exclude Clean from target shifts
    target_shifts = [s for s in SHIFT_CONDITIONS if s[0] != "Clean"]
    
    candidates = [
        ("No_Augmentation", None, {}),
        ("Color_Jitter_Low", ColorJitterWrapper, {"brightness": 0.1, "contrast": 0.1}),
        ("Color_Jitter_Med", ColorJitterWrapper, {"brightness": 0.3, "contrast": 0.3}),
        ("Color_Jitter_High", ColorJitterWrapper, {"brightness": 0.5, "contrast": 0.5}),
        ("Random_Shift_Low", RandomShiftWrapper, {"max_shift": 2}),
        ("Random_Shift_Med", RandomShiftWrapper, {"max_shift": 4}),
        ("Random_Shift_High", RandomShiftWrapper, {"max_shift": 6}),
        ("Cutout_Low", CutoutWrapper, {"cutout_ratio": 0.05}),
        ("Cutout_Med", CutoutWrapper, {"cutout_ratio": 0.15}),
        ("Cutout_High", CutoutWrapper, {"cutout_ratio": 0.3}),
        ("Blur_K3", BlurWrapper, {"kernel_size": 3}),
        ("Blur_K5", BlurWrapper, {"kernel_size": 5}),
        ("Blur_K7", BlurWrapper, {"kernel_size": 7})
    ]
    
    def collect_features_random_reset(env, n_samples=200):
        """Fix Bug 1: Collect features from a fixed, policy-independent observation distribution."""
        feats = []
        device = next(feature_extractor.parameters()).device
        for i in range(n_samples):
            obs, _ = env.reset(seed=seed * 1000 + i)
            obs_t = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
            obs_t = obs_t.to(device)
            with torch.no_grad():
                feats.append(feature_extractor(obs_t).cpu().numpy()[0])
        return np.array(feats)
    
    # 2. Collect Clean Centroid
    env_clean = gym.make(env_id)
    clean_feats = collect_features_random_reset(env_clean, n_samples=200)
    env_clean.close()
    
    all_proxy_results = []
    all_gt_results = []
    
    start_time_total = time.time()
    proxy_total_time = 0.0
    gt_total_time = 0.0
    
    print("\n--- Running Evaluations across Target Shifts ---")
    for shift_name, shift_cls, shift_kwargs in target_shifts:
        print(f"Shift: {shift_name}")
        shift_proxy = []
        shift_gt = []
        
        # Proxy Evaluation
        t0 = time.time()
        for aug_name, aug_cls, aug_kwargs in candidates:
            env_shifted = shift_cls(gym.make(env_id), **shift_kwargs)
            env_aug = aug_cls(env_shifted, **aug_kwargs) if aug_cls else env_shifted
            
            shifted_feats = collect_features_random_reset(env_aug, n_samples=200)
            env_aug.close()
            
            dist = compute_centroid_distance(clean_feats, shifted_feats)
            shift_proxy.append({
                "Shift_Condition": shift_name, 
                "Augmentation": aug_name, 
                "Rep_Distance": dist
            })
        proxy_total_time += (time.time() - t0)
        
        proxy_df = pd.DataFrame(shift_proxy).sort_values("Rep_Distance")
        proxy_df["Proxy_Rank"] = range(1, len(candidates) + 1)
        all_proxy_results.append(proxy_df)
        
        # Ground Truth Evaluation
        t1 = time.time()
        for aug_name, aug_cls, aug_kwargs in candidates:
            env_shifted = shift_cls(gym.make(env_id), **shift_kwargs)
            env_eval = aug_cls(env_shifted, **aug_kwargs) if aug_cls else env_shifted
            
            returns = []
            for ep in range(15): # using 15 episodes to speed up GT for 10x13 conditions
                obs, info = env_eval.reset(seed=seed*100 + ep)
                ep_ret = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, term, trunc, _ = env_eval.step(action)
                    ep_ret += reward
                    done = term or trunc
                returns.append(ep_ret)
            env_eval.close()
            
            shift_gt.append({
                "Shift_Condition": shift_name, 
                "Augmentation": aug_name, 
                "Mean_Return": np.mean(returns)
            })
        gt_total_time += (time.time() - t1)
        
        gt_df = pd.DataFrame(shift_gt).sort_values("Mean_Return", ascending=False)
        gt_df["GT_Rank"] = range(1, len(candidates) + 1)
        all_gt_results.append(gt_df)
        
    df_p = pd.concat(all_proxy_results)
    df_g = pd.concat(all_gt_results)
    final_df = pd.merge(df_p, df_g, on=["Shift_Condition", "Augmentation"])
    
    print("\n--- Final Comparison ---")
    print(final_df.head(20)) # Print a sample
    
    # Pooled Spearman correlation on ranks
    spearman_rho, s_p = stats.spearmanr(final_df["Proxy_Rank"], final_df["GT_Rank"])
    n_samples = len(final_df)
    print(f"\nPOOLED Spearman Rank Correlation (Proxy Rank vs GT Rank, n={n_samples}): {spearman_rho:.3f} (p={s_p:.3e})")
    print(f"Compute Time Saved: Proxy {proxy_total_time:.2f}s vs GT {gt_total_time:.2f}s (Speedup: {gt_total_time/proxy_total_time:.1f}x)")
    
    # Compute per-shift correlation for interpretability
    per_shift_rhos = {}
    for shift_name in final_df["Shift_Condition"].unique():
        shift_data = final_df[final_df["Shift_Condition"] == shift_name]
        rho, p = stats.spearmanr(shift_data["Proxy_Rank"], shift_data["GT_Rank"])
        per_shift_rhos[shift_name] = {"rho": rho, "p_value": p}
        print(f"  {shift_name}: rho={rho:.3f}")

    final_df.to_csv("results/tables/predictive_selection.csv", index=False)
    with open("results/tables/predictive_selection_summary.json", "w") as f:
        json.dump({
            "Pooled_Spearman_rho": spearman_rho,
            "Pooled_Spearman_p": s_p,
            "N_Pairs": n_samples,
            "Per_Shift_Correlations": per_shift_rhos,
            "Proxy_Time_Seconds": proxy_total_time,
            "Ground_Truth_Time_Seconds": gt_total_time,
            "Speedup_Factor": gt_total_time / proxy_total_time if proxy_total_time > 0 else 0
        }, f, indent=2)

if __name__ == "__main__":
    run_predictive_selection()
