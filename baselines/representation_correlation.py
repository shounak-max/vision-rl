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
from utils.metrics import TrackingMetricsLogger

def compute_rbf_mmd(X, Y, gamma=1.0):
    """Computes Gaussian RBF Maximum Mean Discrepancy (MMD) between feature matrices X and Y."""
    def rbf_kernel(A, B):
        # A: (N, D), B: (M, D)
        dist_sq = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * np.dot(A, B.T)
        return np.exp(-gamma * dist_sq)
        
    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)
    
    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return float(np.sqrt(np.maximum(mmd, 0.0)))

def run_correlation_suite(model_path, env_id="SingleObjectTracking-v0"):
    print(f"=== Multi-Metric Representation Distance & Statistical Correlation Suite ===")
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} not found.")
        return
        
    model = PPO.load(model_path)
    feature_extractor = model.policy.features_extractor
    feature_extractor.eval()
    
    # 1. Collect Clean Features & Baseline Performance
    print("Collecting clean environment baseline features...")
    env_clean = gym.make(env_id)
    
    clean_feats = []
    logger = TrackingMetricsLogger()
    for _ in range(30):
        obs, _ = env_clean.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                feat = feature_extractor(obs_tensor).numpy()[0]
            clean_feats.append(feat)
            
            obs, _, terminated, truncated, info = env_clean.step(action)
            logger.add_step_info(info)
            done = terminated or truncated
            
    clean_feats = np.array(clean_feats)
    clean_centroid = np.mean(clean_feats, axis=0)
    clean_metrics = logger.get_episode_metrics()
    env_clean.close()
    
    print(f"Clean Baseline CLE: {clean_metrics['mean_cle']:.2f} px, Success: {clean_metrics['success_rate']*100:.2f}%")
    
    # 2. Spectrum of 16 Continuous Corruption Levels
    shift_spectrum = [
        ("Noise (sigma=0.02)", NoiseWrapper, {"noise_std": 0.02}),
        ("Noise (sigma=0.05)", NoiseWrapper, {"noise_std": 0.05}),
        ("Noise (sigma=0.08)", NoiseWrapper, {"noise_std": 0.08}),
        ("Noise (sigma=0.10)", NoiseWrapper, {"noise_std": 0.10}),
        ("Noise (sigma=0.15)", NoiseWrapper, {"noise_std": 0.15}),
        ("Noise (sigma=0.20)", NoiseWrapper, {"noise_std": 0.20}),
        ("Noise (sigma=0.30)", NoiseWrapper, {"noise_std": 0.30}),
        ("Noise (sigma=0.40)", NoiseWrapper, {"noise_std": 0.40}),
        ("Distractors (N=1)", DistractorWrapper, {"num_distractors": 1}),
        ("Distractors (N=2)", DistractorWrapper, {"num_distractors": 2}),
        ("Distractors (N=3)", DistractorWrapper, {"num_distractors": 3}),
        ("Distractors (N=4)", DistractorWrapper, {"num_distractors": 4}),
        ("Viewpoint (angle=10deg)", ViewpointWrapper, {"max_angle": 10}),
        ("Viewpoint (angle=20deg)", ViewpointWrapper, {"max_angle": 20}),
        ("Viewpoint (angle=30deg)", ViewpointWrapper, {"max_angle": 30}),
        ("Viewpoint (angle=45deg)", ViewpointWrapper, {"max_angle": 45}),
    ]
    
    results = []
    
    for name, wrapper_cls, kwargs in shift_spectrum:
        env = wrapper_cls(gym.make(env_id), **kwargs)
        shifted_feats = []
        logger_shift = TrackingMetricsLogger()
        
        for _ in range(20):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    feat = feature_extractor(obs_tensor).numpy()[0]
                shifted_feats.append(feat)
                
                obs, _, terminated, truncated, info = env.step(action)
                logger_shift.add_step_info(info)
                done = terminated or truncated
                
        shifted_feats = np.array(shifted_feats)
        shifted_centroid = np.mean(shifted_feats, axis=0)
        metrics_shift = logger_shift.get_episode_metrics()
        env.close()
        
        # Metrics Calculation
        d_euc = float(np.linalg.norm(clean_centroid - shifted_centroid))
        
        # Cosine Distance
        norm_clean = np.linalg.norm(clean_centroid)
        norm_shift = np.linalg.norm(shifted_centroid)
        if norm_clean > 0 and norm_shift > 0:
            d_cos = float(1.0 - np.dot(clean_centroid, shifted_centroid) / (norm_clean * norm_shift))
        else:
            d_cos = 0.0
            
        # MMD Distance
        d_mmd = compute_rbf_mmd(clean_feats[:100], shifted_feats[:100])
        
        # Feature Variance
        feat_var = float(np.var(np.linalg.norm(shifted_feats, axis=1)))
        
        results.append({
            "Shift_Condition": name,
            "Euclidean_Distance": d_euc,
            "Cosine_Distance": d_cos,
            "MMD_Distance": d_mmd,
            "Feature_Variance": feat_var,
            "Mean_CLE": metrics_shift['mean_cle'],
            "Success_Rate": metrics_shift['success_rate']
        })
        print(f"[{name}] d_Euc: {d_euc:.2f}, d_Cos: {d_cos:.4f}, MMD: {d_mmd:.4f} -> CLE: {metrics_shift['mean_cle']:.2f} px")
        
    df = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/representation_distance_comparison.csv", index=False)
    
    # 3. Statistical Correlation Analysis (Pearson r & Spearman rho)
    euc_dists = df['Euclidean_Distance'].values
    cos_dists = df['Cosine_Distance'].values
    mmd_dists = df['MMD_Distance'].values
    cles = df['Mean_CLE'].values
    successes = df['Success_Rate'].values
    
    pearson_euc_cle, p_p_euc = stats.pearsonr(euc_dists, cles)
    spearman_euc_cle, p_s_euc = stats.spearmanr(euc_dists, cles)
    
    pearson_cos_cle, p_p_cos = stats.pearsonr(cos_dists, cles)
    spearman_cos_cle, p_s_cos = stats.spearmanr(cos_dists, cles)

    pearson_mmd_cle, p_p_mmd = stats.pearsonr(mmd_dists, cles)
    spearman_mmd_cle, p_s_mmd = stats.spearmanr(mmd_dists, cles)
    
    correlation_summary = {
        "Euclidean_Distance_vs_CLE": {
            "Pearson_r": float(pearson_euc_cle),
            "Pearson_p": float(p_p_euc),
            "Spearman_rho": float(spearman_euc_cle),
            "Spearman_p": float(p_s_euc)
        },
        "Cosine_Distance_vs_CLE": {
            "Pearson_r": float(pearson_cos_cle),
            "Pearson_p": float(p_p_cos),
            "Spearman_rho": float(spearman_cos_cle),
            "Spearman_p": float(p_s_cos)
        },
        "MMD_Distance_vs_CLE": {
            "Pearson_r": float(pearson_mmd_cle),
            "Pearson_p": float(p_p_mmd),
            "Spearman_rho": float(spearman_mmd_cle),
            "Spearman_p": float(p_s_mmd)
        }
    }
    
    with open("results/tables/correlation_analysis.json", "w") as f:
        json.dump(correlation_summary, f, indent=2)
        
    print("\n================ STATISTICAL CORRELATION SUMMARY ================")
    print(f"Euclidean Dist vs CLE  -> Pearson r: {pearson_euc_cle:.4f} (p={p_p_euc:.4e}), Spearman rho: {spearman_euc_cle:.4f} (p={p_s_euc:.4e})")
    print(f"Cosine Dist vs CLE     -> Pearson r: {pearson_cos_cle:.4f} (p={p_p_cos:.4e}), Spearman rho: {spearman_cos_cle:.4f} (p={p_s_cos:.4e})")
    print(f"MMD Dist vs CLE        -> Pearson r: {pearson_mmd_cle:.4f} (p={p_p_mmd:.4e}), Spearman rho: {spearman_mmd_cle:.4f} (p={p_s_mmd:.4e})")
    print("==================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/models/PPO_SingleObjectTracking-v0_s42.zip")
    args = parser.parse_args()
    
    run_correlation_suite(args.model)
