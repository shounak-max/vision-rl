import os
import argparse
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs
from utils.stats import compute_statistics, welch_ttest, format_stat_string
from utils.metrics import TrackingMetricsLogger

def evaluate_policy(model, env_id, n_episodes=10):
    env = gym.make(env_id)
    logger = TrackingMetricsLogger()
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if model is None: # Random policy
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            logger.add_step_info(info)
            done = terminated or truncated
    metrics = logger.get_episode_metrics()
    env.close()
    return metrics

def run_multiseed_experiment(env_id="SingleObjectTracking-v0", steps=30000, seeds=[0, 42, 100, 123, 999]):
    print(f"=== Multi-Seed Evaluation for {env_id} across {len(seeds)} seeds ===")
    os.makedirs("results/models", exist_ok=True)
    
    # 1. Random Agent Baseline Across Seeds
    random_successes = []
    random_cles = []
    print("Evaluating Random Policy baseline...")
    for s in seeds:
        np.random.seed(s)
        m = evaluate_policy(None, env_id, n_episodes=5)
        random_successes.append(m['success_rate'])
        random_cles.append(m['mean_cle'])
        
    rand_succ_stat = compute_statistics(random_successes)
    rand_cle_stat = compute_statistics(random_cles)
    
    # 2. Train PPO across seeds
    ppo_results = []
    ppo_successes = []
    ppo_cles = []
    
    for seed in seeds:
        print(f"\nTraining PPO Seed {seed}...")
        vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=256, batch_size=64)
        model.learn(total_timesteps=steps)
        
        save_path = f"results/models/PPO_{env_id}_s{seed}"
        model.save(save_path)
        
        eval_metrics = evaluate_policy(model, env_id, n_episodes=10)
        ppo_successes.append(eval_metrics['success_rate'])
        ppo_cles.append(eval_metrics['mean_cle'])
        
        ppo_results.append({
            "Algorithm": "PPO",
            "Seed": seed,
            "Success_Rate": eval_metrics['success_rate'],
            "Mean_CLE": eval_metrics['mean_cle']
        })
        vec_env.close()
        
    ppo_succ_stat = compute_statistics(ppo_successes)
    ppo_cle_stat = compute_statistics(ppo_cles)
    
    # 3. Compute Statistical Significance (Welch's t-test PPO vs Random)
    t_stat, p_val = welch_ttest(ppo_successes, random_successes)
    
    # Save CSV
    df = pd.DataFrame(ppo_results)
    df.to_csv("results/multiseed_results.csv", index=False)
    
    # Save Summary JSON
    summary = {
        "Environment": env_id,
        "Seeds": seeds,
        "Timesteps": steps,
        "PPO_Success": {
            "mean": ppo_succ_stat['mean'],
            "std": ppo_succ_stat['std'],
            "ci_95": ppo_succ_stat['ci_95'],
            "formatted": format_stat_string(ppo_succ_stat['mean'] * 100, ppo_succ_stat['std'] * 100) + "%"
        },
        "PPO_CLE": {
            "mean": ppo_cle_stat['mean'],
            "std": ppo_cle_stat['std'],
            "ci_95": ppo_cle_stat['ci_95'],
            "formatted": format_stat_string(ppo_cle_stat['mean'], ppo_cle_stat['std'])
        },
        "Random_Success": {
            "mean": rand_succ_stat['mean'],
            "std": rand_succ_stat['std'],
            "formatted": format_stat_string(rand_succ_stat['mean'] * 100, rand_succ_stat['std'] * 100) + "%"
        },
        "Statistical_Significance_vs_Random": {
            "p_value": p_val,
            "significant_at_0.05": bool(p_val < 0.05)
        }
    }
    
    with open("results/multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n================ MULTI-SEED SUMMARY ================")
    print(f"PPO Success Rate:   {summary['PPO_Success']['formatted']}")
    print(f"Random Success Rate:{summary['Random_Success']['formatted']}")
    print(f"PPO vs Random p-val:{p_val:.4e} (Significant: {summary['Statistical_Significance_vs_Random']['significant_at_0.05']})")
    print("====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SingleObjectTracking-v0")
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    
    run_multiseed_experiment(args.env, args.steps)
