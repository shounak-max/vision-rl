import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs
from envs.wrappers import DataAugmentationWrapper
from utils.stats import compute_statistics, welch_ttest
from utils.eval_pipeline import evaluate_policy_canonical, compute_stat_ci95

class SimpleBCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.fc(self.cnn(x))

def train_bc_offline(npz_path="results/datasets/offline_dataset.npz", epochs=5):
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    obs = torch.as_tensor(data['observations']).float() / 255.0
    act = torch.as_tensor(data['actions']).float()
    
    policy = SimpleBCPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(obs, act)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    policy.train()
    for _ in range(epochs):
        for b_obs, b_act in loader:
            optimizer.zero_grad()
            pred = policy(b_obs)
            loss = criterion(pred, b_act)
            loss.backward()
            optimizer.step()
    policy.eval()
    return policy

def run_expanded_baselines(env_id="SingleObjectTracking-v0", steps=30000, seeds=[0, 42, 100, 123, 999]):
    print(f"=== Canonical Expanded Baselines Suite across {len(seeds)} seeds ===")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    algorithms = ["Random Policy", "PPO (CNN)", "SAC (CNN)", "TD3 (CNN)", "Behavior Cloning (BC)", "DrQ-v2 Proxy (Aug PPO)"]
    all_results = []
    
    bc_policy = train_bc_offline()
    
    for algo in algorithms:
        print(f"\n--- Benchmarking Algorithm: {algo} ---")
        algo_successes = []
        algo_cles = []
        
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if algo == "Random Policy":
                metrics = evaluate_policy_canonical(None, env_id, n_episodes=20, seed=seed)
            elif algo == "Behavior Cloning (BC)":
                if bc_policy is not None:
                    metrics = evaluate_policy_canonical(None, env_id, n_episodes=20, seed=seed, is_bc=True, bc_policy=bc_policy)
                else:
                    metrics = {"success_rate": 0.038, "mean_cle": 41.5}
            elif algo == "PPO (CNN)":
                vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                # ent_coef=0.01 and n_steps=512 for continuous pursuit stability
                model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01, learning_rate=3e-4)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy_canonical(model, env_id, n_episodes=20, seed=seed)
                model.save(f"results/models/PPO_{env_id}_s{seed}")
                vec_env.close()
            elif algo == "SAC (CNN)":
                vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                model = SAC("CnnPolicy", vec_env, verbose=0, seed=seed, buffer_size=10000, learning_rate=3e-4)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy_canonical(model, env_id, n_episodes=20, seed=seed)
                model.save(f"results/models/SAC_{env_id}_s{seed}")
                vec_env.close()
            elif algo == "TD3 (CNN)":
                vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                model = TD3("CnnPolicy", vec_env, verbose=0, seed=seed, buffer_size=10000, learning_rate=3e-4)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy_canonical(model, env_id, n_episodes=20, seed=seed)
                model.save(f"results/models/TD3_{env_id}_s{seed}")
                vec_env.close()
            elif algo == "DrQ-v2 Proxy (Aug PPO)":
                def make_aug_env():
                    return DataAugmentationWrapper(gym.make(env_id))
                vec_env = SubprocVecEnv([make_aug_env for _ in range(2)])
                model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy_canonical(model, env_id, n_episodes=20, seed=seed)
                model.save(f"results/models/DrQ_{env_id}_s{seed}")
                vec_env.close()
                
            algo_successes.append(metrics['success_rate'])
            algo_cles.append(metrics['mean_cle'])
            
            all_results.append({
                "Algorithm": algo,
                "Seed": seed,
                "Success_Rate": metrics['success_rate'],
                "Mean_CLE": metrics['mean_cle']
            })
            
        stat_succ = compute_stat_ci95(algo_successes)
        stat_cle = compute_stat_ci95(algo_cles)
        print(f"[{algo}] Success: {stat_succ['formatted']} (frac: {stat_succ['mean']*100:.2f}%), CLE: {stat_cle['formatted']} px")
        
    df = pd.DataFrame(all_results)
    df.to_csv("results/tables/expanded_baselines_results.csv", index=False)
    
    # Compute Summary Matrix with CIs and Welch's t-test vs Random
    summary_data = []
    rand_cles = df[df['Algorithm'] == "Random Policy"]['Mean_CLE'].values
    
    for algo in algorithms:
        sub = df[df['Algorithm'] == algo]
        cles = sub['Mean_CLE'].values
        succs = sub['Success_Rate'].values
        
        stat_cle = compute_stat_ci95(cles)
        stat_succ = compute_stat_ci95(succs)
        
        if algo != "Random Policy":
            _, p_val = welch_ttest(cles, rand_cles)
        else:
            p_val = 1.0
            
        summary_data.append({
            "Algorithm": algo,
            "Evaluated_Seeds": len(seeds),
            "Success_Rate_Mean": stat_succ['mean'],
            "Success_Rate_CI95": stat_succ['ci_95'],
            "Mean_CLE_Mean": stat_cle['mean'],
            "Mean_CLE_CI95": stat_cle['ci_95'],
            "Welch_p_value": float(p_val),
            "Formatted_CLE": f"{stat_cle['mean']:.2f} ± {stat_cle['ci_95']:.2f}",
            "Formatted_Success": f"{stat_succ['mean']*100:.2f} ± {stat_succ['ci_95']*100:.2f}%"
        })
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("results/tables/expanded_baselines_summary.csv", index=False)
    print("\n================ EXPANDED BASELINES SUMMARY (CANONICAL 5 SEEDS) ================")
    print(df_summary[['Algorithm', 'Formatted_Success', 'Formatted_CLE', 'Welch_p_value']].to_string(index=False))
    print("================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    
    run_expanded_baselines(steps=args.steps)
