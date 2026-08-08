import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs
from envs.wrappers import DataAugmentationWrapper
from utils.eval_pipeline import evaluate_policy_canonical, compute_stat_ci95
from utils.stats import welch_ttest

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
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def forward(self, obs):
        feat = self.cnn(obs)
        return self.fc(feat)

def train_bc_offline(dataset_path="results/datasets/vd4rl_sot_random.npz", epochs=3, batch_size=64):
    """Trains a Behavior Cloning policy on offline trajectory data."""
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Generating offline random dataset...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        env = gym.make("SingleObjectTracking-v0")
        obs_list, act_list = [], []
        for _ in range(50):
            obs, _ = env.reset()
            for _ in range(100):
                act = env.action_space.sample()
                obs_list.append(obs)
                act_list.append(act)
                obs, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
        env.close()
        np.savez_compressed(dataset_path, observations=np.array(obs_list), actions=np.array(act_list))
        
    print(f"Loading offline dataset from {dataset_path} for Behavior Cloning...")
    data = np.load(dataset_path)
    obs = torch.tensor(data['observations'], dtype=torch.float32) / 255.0
    act = torch.tensor(data['actions'], dtype=torch.float32)
    
    dataset = TensorDataset(obs, act)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    policy = SimpleBCPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    policy.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            pred_act = policy(batch_obs)
            loss = criterion(pred_act, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"BC Training Complete. Final MSE Loss: {total_loss/len(loader):.4f}")
    return policy

def run_expanded_baselines(env_id="SingleObjectTracking-v0", steps=30000, seeds=[0, 42, 100, 123, 999]):
    """
    Executes the 6-algorithm benchmark suite across 5 seeds using the canonical evaluation pipeline.
    Calculates mean, std, 95% CIs, and Welch's t-test p-values versus Random Policy.
    """
    print(f"=== Canonical Expanded Baselines Suite across {len(seeds)} seeds ===")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    algorithms = [
        "Random Policy",
        "PPO (CNN)",
        "SAC (CNN)",
        "TD3 (CNN)",
        "Behavior Cloning (BC)",
        "DrQ-v2 Proxy (Aug PPO)"
    ]
    
    results = []
    summary_rows = []
    
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
                metrics = evaluate_policy_canonical(None, env_id, n_episodes=20, seed=seed, is_bc=True, bc_policy=bc_policy)
            elif algo == "PPO (CNN)":
                vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
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
                aug_env_fn = lambda: DataAugmentationWrapper(gym.make(env_id))
                vec_env = make_vec_env(aug_env_fn, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01, learning_rate=3e-4)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy_canonical(model, env_id, n_episodes=20, seed=seed)
                model.save(f"results/models/DrQ_{env_id}_s{seed}")
                vec_env.close()
                
            algo_successes.append(metrics['success_rate'])
            algo_cles.append(metrics['mean_cle'])
            
            results.append({
                "Algorithm": algo,
                "Seed": seed,
                "Success_Rate": metrics['success_rate'],
                "Mean_CLE": metrics['mean_cle']
            })
            
            print(f"[{algo}] Seed {seed}: Success = {metrics['success_rate']*100:.2f}%, CLE = {metrics['mean_cle']:.2f} px")

    df_raw = pd.DataFrame(results)
    df_raw.to_csv("results/tables/expanded_baselines_results.csv", index=False)
    
    random_cles = df_raw[df_raw['Algorithm'] == 'Random Policy']['Mean_CLE'].values
    
    for algo in algorithms:
        sub_df = df_raw[df_raw['Algorithm'] == algo]
        succ_stats = compute_stat_ci95(sub_df['Success_Rate'].values)
        cle_stats = compute_stat_ci95(sub_df['Mean_CLE'].values)
        
        if algo == "Random Policy":
            p_val = 1.0
        else:
            _, p_val = welch_ttest(sub_df['Mean_CLE'].values, random_cles)
            
        summary_rows.append({
            "Algorithm": algo,
            "Evaluated_Seeds": len(seeds),
            "Success_Rate_Mean": succ_stats['mean'],
            "Success_Rate_CI95": succ_stats['ci_95'],
            "Mean_CLE_Mean": cle_stats['mean'],
            "Mean_CLE_CI95": cle_stats['ci_95'],
            "Welch_p_value": p_val,
            "Formatted_CLE": cle_stats['formatted'],
            "Formatted_Success": succ_stats['formatted']
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/tables/expanded_baselines_summary.csv", index=False)
    
    print("\n================ CANONICAL BENCHMARK SUMMARY (5 SEEDS) ================")
    print(df_summary[['Algorithm', 'Formatted_Success', 'Formatted_CLE', 'Welch_p_value']])
    print("========================================================================\n")

if __name__ == "__main__":
    run_expanded_baselines()
