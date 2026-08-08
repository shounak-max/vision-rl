import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from stable_baselines3 import PPO
import gymnasium as gym
import envs.tracking_envs
from envs.wrappers import NoiseWrapper, DistractorWrapper, ViewpointWrapper
from utils.eval_pipeline import evaluate_policy_canonical, compute_stat_ci95
from utils.dataset_partitions import BENCHMARK_PARTITIONS

def compute_multi_kernel_rbf_mmd(X, Y, gammas=[0.01, 0.1, 1.0, 10.0]):
    """Computes Multi-Kernel Gaussian RBF Maximum Mean Discrepancy (MMD) with adaptive bandwidths."""
    def rbf_kernel(A, B, gamma):
        dist_sq = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * np.dot(A, B.T)
        return np.exp(-gamma * dist_sq)
        
    mmd_total = 0.0
    for g in gammas:
        K_XX = rbf_kernel(X, X, g)
        K_YY = rbf_kernel(Y, Y, g)
        K_XY = rbf_kernel(X, Y, g)
        mmd_g = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        mmd_total += np.sqrt(np.maximum(mmd_g, 0.0))
        
    return float(mmd_total / len(gammas))

def run_correlation_suite(model_dir="results/models", env_id="SingleObjectTracking-v0", seeds=[0, 42, 100, 123, 999]):
    print(f"=== Multi-Seed Representation Distance & Statistical Correlation Suite across {len(seeds)} seeds ===")
    os.makedirs("results/tables", exist_ok=True)
    
    test_spectrum = BENCHMARK_PARTITIONS["validation"]["shift_spectrum"] + BENCHMARK_PARTITIONS["test"]["shift_spectrum"]
    
    all_seed_results = []
    
    for seed in seeds:
        model_path = os.path.join(model_dir, f"PPO_{env_id}_s{seed}.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, f"PPO_{env_id}_s42.zip")
        if not os.path.exists(model_path):
            # Fall back to training a temporary PPO model for this seed if checkpoint does not exist
            vec_env = gym.make(env_id)
            model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01)
            model.learn(total_timesteps=20000)
            vec_env.close()
        else:
            model = PPO.load(model_path)
            
        feature_extractor = model.policy.features_extractor
        feature_extractor.eval()
        
        # 1. Collect Clean Features
        env_clean = gym.make(env_id)
        clean_feats = []
        for ep in range(10):
            obs, _ = env_clean.reset(seed=seed * 100 + ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    feat = feature_extractor(obs_tensor).numpy()[0]
                clean_feats.append(feat)
                obs, _, terminated, truncated, _ = env_clean.step(action)
                done = terminated or truncated
        env_clean.close()
        
        clean_feats = np.array(clean_feats)
        clean_centroid = np.mean(clean_feats, axis=0)
        
        # 2. Evaluate across spectrum
        for name, wrapper_cls, kwargs in test_spectrum:
            metrics_shift = evaluate_policy_canonical(model, env_id, n_episodes=10, seed=seed, wrapper_cls=wrapper_cls, wrapper_kwargs=kwargs)
            
            # Extract shifted features for representation distance
            env_shift = wrapper_cls(gym.make(env_id), **kwargs)
            shifted_feats = []
            for ep in range(5):
                obs, _ = env_shift.reset(seed=seed * 100 + ep)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs_tensor = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
                    with torch.no_grad():
                        feat = feature_extractor(obs_tensor).numpy()[0]
                    shifted_feats.append(feat)
                    obs, _, terminated, truncated, _ = env_shift.step(action)
                    done = terminated or truncated
            env_shift.close()
            
            shifted_feats = np.array(shifted_feats)
            shifted_centroid = np.mean(shifted_feats, axis=0)
            
            d_euc = float(np.linalg.norm(clean_centroid - shifted_centroid))
            
            norm_clean = np.linalg.norm(clean_centroid)
            norm_shift = np.linalg.norm(shifted_centroid)
            d_cos = float(1.0 - np.dot(clean_centroid, shifted_centroid) / (norm_clean * norm_shift)) if norm_clean > 0 and norm_shift > 0 else 0.0
            
            d_mmd = compute_multi_kernel_rbf_mmd(clean_feats[:300], shifted_feats[:300])
            
            all_seed_results.append({
                "Seed": seed,
                "Shift_Condition": name,
                "Euclidean_Distance": d_euc,
                "Cosine_Distance": d_cos,
                "MMD_Distance": d_mmd,
                "Mean_CLE": metrics_shift['mean_cle'],
                "Success_Rate": metrics_shift['success_rate']
            })
            
    df_raw = pd.DataFrame(all_seed_results)
    df_raw.to_csv("results/tables/representation_distance_comparison.csv", index=False)
    
    # Compute statistical correlations across all seeds
    euc_dists = df_raw['Euclidean_Distance'].values
    cos_dists = df_raw['Cosine_Distance'].values
    mmd_dists = df_raw['MMD_Distance'].values
    cles = df_raw['Mean_CLE'].values
    
    pearson_euc, p_p_euc = stats.pearsonr(euc_dists, cles)
    spearman_euc, p_s_euc = stats.spearmanr(euc_dists, cles)
    
    pearson_cos, p_p_cos = stats.pearsonr(cos_dists, cles)
    spearman_cos, p_s_cos = stats.spearmanr(cos_dists, cles)
    
    pearson_mmd, p_p_mmd = stats.pearsonr(mmd_dists, cles)
    spearman_mmd, p_s_mmd = stats.spearmanr(mmd_dists, cles)
    
    summary = {
        "Evaluated_Seeds": len(seeds),
        "Total_Evaluated_Points": len(df_raw),
        "Euclidean_Distance_vs_CLE": {
            "Pearson_r": float(pearson_euc),
            "Pearson_p": float(p_p_euc),
            "Spearman_rho": float(spearman_euc),
            "Spearman_p": float(p_s_euc)
        },
        "Cosine_Distance_vs_CLE": {
            "Pearson_r": float(pearson_cos),
            "Pearson_p": float(p_p_cos),
            "Spearman_rho": float(spearman_cos),
            "Spearman_p": float(p_s_cos)
        },
        "MMD_Distance_vs_CLE": {
            "Pearson_r": float(pearson_mmd),
            "Pearson_p": float(p_p_mmd),
            "Spearman_rho": float(spearman_mmd),
            "Spearman_p": float(p_s_mmd)
        }
    }
    
    with open("results/tables/correlation_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n================ STATISTICAL CORRELATION SUMMARY (5 SEEDS) ================")
    print(f"Euclidean Dist vs CLE  -> Pearson r: {pearson_euc:.4f} (p={p_p_euc:.4e}), Spearman rho: {spearman_euc:.4f} (p={p_s_euc:.4e})")
    print(f"Cosine Dist vs CLE     -> Pearson r: {pearson_cos:.4f} (p={p_p_cos:.4e}), Spearman rho: {spearman_cos:.4f} (p={p_s_cos:.4e})")
    print(f"MMD Dist vs CLE        -> Pearson r: {pearson_mmd:.4f} (p={p_p_mmd:.4e}), Spearman rho: {spearman_mmd:.4f} (p={p_s_mmd:.4e})")
    print("==========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="results/models")
    args = parser.parse_args()
    
    run_correlation_suite(args.model_dir)
