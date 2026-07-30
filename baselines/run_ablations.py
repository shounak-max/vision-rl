import os
import argparse
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs
from envs.wrappers import DataAugmentationWrapper, NoiseWrapper
from utils.stats import compute_statistics, format_stat_string
from utils.metrics import TrackingMetricsLogger

def train_and_eval_ablation(config_name, wrapper_class=None, steps=25000, seed=42):
    print(f"\n--- Running Ablation Config: {config_name} ---")
    env_id = "SingleObjectTracking-v0"
    
    def make_env():
        env = gym.make(env_id)
        if wrapper_class:
            env = wrapper_class(env)
        return env
        
    vec_env = SubprocVecEnv([make_env for _ in range(2)])
    model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=256)
    model.learn(total_timesteps=steps)
    
    # Evaluate under noise shift to see if ablation affected robustness
    eval_env = NoiseWrapper(gym.make(env_id), noise_std=0.2)
    logger = TrackingMetricsLogger()
    
    for _ in range(10):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            logger.add_step_info(info)
            done = terminated or truncated
            
    metrics = logger.get_episode_metrics()
    eval_env.close()
    vec_env.close()
    
    print(f"[{config_name}] OOD Noise Success Rate: {metrics['success_rate']:.4f}, Mean CLE: {metrics['mean_cle']:.2f}")
    return {
        "Configuration": config_name,
        "OOD_Success_Rate": metrics['success_rate'],
        "OOD_Mean_CLE": metrics['mean_cle']
    }

def run_all_ablations(steps=25000):
    results = []
    
    # 1. Full Model (with Data Augmentation)
    results.append(train_and_eval_ablation("Full Model (with DataAug)", DataAugmentationWrapper, steps=steps))
    
    # 2. No Augmentation
    results.append(train_and_eval_ablation("No Data Augmentation", None, steps=steps))
    
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ablation_results.csv", index=False)
    print("\n================ ABLATION RESULTS ================")
    print(df.to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25000)
    args = parser.parse_args()
    
    run_all_ablations(args.steps)
