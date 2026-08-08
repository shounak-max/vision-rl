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
from utils.eval_pipeline import evaluate_policy_canonical, compute_stat_ci95

def train_and_eval_ablation(config_name, wrapper_class=None, steps=30000, seeds=[0, 42, 100, 123, 999]):
    print(f"\n--- Running Ablation Config: {config_name} across {len(seeds)} seeds ---")
    env_id = "SingleObjectTracking-v0"
    
    succs = []
    cles = []
    
    for seed in seeds:
        def make_env():
            env = gym.make(env_id)
            if wrapper_class:
                env = wrapper_class(env)
            return env
            
        vec_env = SubprocVecEnv([make_env for _ in range(2)])
        model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01)
        model.learn(total_timesteps=steps)
        
        # Evaluate under noise shift to see if ablation affected robustness
        metrics = evaluate_policy_canonical(model, env_id, n_episodes=20, seed=seed, wrapper_cls=NoiseWrapper, wrapper_kwargs={"noise_std": 0.2})
        succs.append(metrics['success_rate'])
        cles.append(metrics['mean_cle'])
        vec_env.close()
        
    stat_succ = compute_stat_ci95(succs)
    stat_cle = compute_stat_ci95(cles)
    
    print(f"[{config_name}] OOD Noise Success: {stat_succ['formatted']} (frac: {stat_succ['mean']*100:.2f}%), CLE: {stat_cle['formatted']} px")
    return {
        "Configuration": config_name,
        "OOD_Success_Rate_Mean": stat_succ['mean'],
        "OOD_Success_Rate_CI95": stat_succ['ci_95'],
        "OOD_Mean_CLE_Mean": stat_cle['mean'],
        "OOD_Mean_CLE_CI95": stat_cle['ci_95'],
        "Formatted_Success": f"{stat_succ['mean']*100:.2f} ± {stat_succ['ci_95']*100:.2f}%",
        "Formatted_CLE": f"{stat_cle['mean']:.2f} ± {stat_cle['ci_95']:.2f}"
    }

def run_all_ablations(steps=30000, seeds=[0, 42, 100, 123, 999]):
    results = []
    
    # 1. Full Model (with Data Augmentation)
    results.append(train_and_eval_ablation("Full Model (with DataAug)", DataAugmentationWrapper, steps=steps, seeds=seeds))
    
    # 2. No Augmentation
    results.append(train_and_eval_ablation("No Data Augmentation", None, steps=steps, seeds=seeds))
    
    df = pd.DataFrame(results)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/ablation_results.csv", index=False)
    print("\n================ ABLATION RESULTS (5 SEEDS) ================")
    print(df[['Configuration', 'Formatted_Success', 'Formatted_CLE']].to_string(index=False))
    print("===========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    
    run_all_ablations(args.steps)
