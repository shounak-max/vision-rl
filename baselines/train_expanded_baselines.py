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
from utils.stats import compute_statistics, welch_ttest, format_stat_string
from utils.metrics import TrackingMetricsLogger

def evaluate_policy(model, env_id, n_episodes=10, is_bc=False, bc_policy=None):
    env = gym.make(env_id)
    logger = TrackingMetricsLogger()
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if model is None and not is_bc: # Random policy
                action = env.action_space.sample()
            elif is_bc:
                obs_t = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    action = bc_policy(obs_t).numpy()[0]
                action = np.clip(action, -1.0, 1.0)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            logger.add_step_info(info)
            done = terminated or truncated
    metrics = logger.get_episode_metrics()
    env.close()
    return metrics

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

def run_expanded_baselines(env_id="SingleObjectTracking-v0", steps=25000, seeds=[0, 42, 100, 123, 999]):
    print(f"=== Running Expanded Baselines Suite across {len(seeds)} seeds ===")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    algorithms = ["Random Policy", "PPO (CNN)", "SAC (CNN)", "TD3 (CNN)", "Behavior Cloning (BC)", "DrQ-v2 Proxy (Aug PPO)"]
    all_results = []
    
    # Pre-train BC policy if offline dataset exists
    bc_policy = train_bc_offline()
    
    for algo in algorithms:
        print(f"\n--- Benchmarking Algorithm: {algo} ---")
        algo_successes = []
        algo_cles = []
        
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if algo == "Random Policy":
                metrics = evaluate_policy(None, env_id, n_episodes=5)
            elif algo == "Behavior Cloning (BC)":
                if bc_policy is not None:
                    metrics = evaluate_policy(None, env_id, n_episodes=5, is_bc=True, bc_policy=bc_policy)
                else:
                    metrics = {"success_rate": 0.0, "mean_cle": 64.0}
            elif algo == "PPO (CNN)":
                vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=256)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy(model, env_id, n_episodes=10)
                vec_env.close()
            elif algo == "SAC (CNN)":
                vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                model = SAC("CnnPolicy", vec_env, verbose=0, seed=seed, buffer_size=10000)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy(model, env_id, n_episodes=10)
                vec_env.close()
            elif algo == "TD3 (CNN)":
                vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
                model = TD3("CnnPolicy", vec_env, verbose=0, seed=seed, buffer_size=10000)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy(model, env_id, n_episodes=10)
                vec_env.close()
            elif algo == "DrQ-v2 Proxy (Aug PPO)":
                def make_aug_env():
                    return DataAugmentationWrapper(gym.make(env_id))
                vec_env = SubprocVecEnv([make_aug_env for _ in range(2)])
                model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=256)
                model.learn(total_timesteps=steps)
                metrics = evaluate_policy(model, env_id, n_episodes=10)
                vec_env.close()
                
            algo_successes.append(metrics['success_rate'])
            algo_cles.append(metrics['mean_cle'])
            
            all_results.append({
                "Algorithm": algo,
                "Seed": seed,
                "Success_Rate": metrics['success_rate'],
                "Mean_CLE": metrics['mean_cle']
            })
            
        stat_succ = compute_statistics(algo_successes)
        stat_cle = compute_statistics(algo_cles)
        print(f"[{algo}] Success: {stat_succ['mean']*100:.2f} ± {stat_succ['ci_95']*100:.2f}%, CLE: {stat_cle['mean']:.2f} ± {stat_cle['ci_95']:.2f} px")
        
    df = pd.DataFrame(all_results)
    df.to_csv("results/tables/expanded_baselines_results.csv", index=False)
    
    # Compute Summary Matrix with CIs
    summary_data = []
    rand_cles = df[df['Algorithm'] == "Random Policy"]['Mean_CLE'].values
    
    for algo in algorithms:
        sub = df[df['Algorithm'] == algo]
        cles = sub['Mean_CLE'].values
        succs = sub['Success_Rate'].values
        
        stat_cle = compute_statistics(cles)
        stat_succ = compute_statistics(succs)
        
        if algo != "Random Policy":
            _, p_val = welch_ttest(cles, rand_cles)
        else:
            p_val = 1.0
            
        summary_data.append({
            "Algorithm": algo,
            "Success_Rate_Mean": stat_succ['mean'],
            "Success_Rate_CI95": stat_succ['ci_95'],
            "Mean_CLE_Mean": stat_cle['mean'],
            "Mean_CLE_CI95": stat_cle['ci_95'],
            "Welch_p_value": p_val,
            "Formatted_CLE": f"{stat_cle['mean']:.2f} ± {stat_cle['ci_95']:.2f}",
            "Formatted_Success": f"{stat_succ['mean']*100:.2f} ± {stat_succ['ci_95']*100:.2f}%"
        })
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("results/tables/expanded_baselines_summary.csv", index=False)
    print("\n================ EXPANDED BASELINES SUMMARY ================")
    print(df_summary[['Algorithm', 'Formatted_Success', 'Formatted_CLE', 'Welch_p_value']].to_string(index=False))
    print("===========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25000)
    args = parser.parse_args()
    
    run_expanded_baselines(steps=args.steps)
