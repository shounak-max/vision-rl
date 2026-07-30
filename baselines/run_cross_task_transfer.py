import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs
from utils.stats import compute_statistics, format_stat_string
from utils.metrics import TrackingMetricsLogger

def eval_policy(model, env_id, n_episodes=10):
    env = gym.make(env_id)
    logger = TrackingMetricsLogger()
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            logger.add_step_info(info)
            done = terminated or truncated
    metrics = logger.get_episode_metrics()
    env.close()
    return metrics

def run_transfer_experiment(source_env="SingleObjectTracking-v0", target_env="ActiveTracking-v0", steps=20000, seeds=[0, 42, 100]):
    print(f"=== Cross-Task Transfer Benchmark: {source_env} -> {target_env} ===")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed}: Training Source Task ({source_env}) ---")
        src_vec_env = make_vec_env(source_env, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_src = PPO("CnnPolicy", src_vec_env, verbose=0, seed=seed, n_steps=256)
        model_src.learn(total_timesteps=steps)
        
        src_save_path = f"results/models/PPO_source_s{seed}.zip"
        model_src.save(src_save_path)
        src_vec_env.close()
        
        # 1. Evaluate Zero-Shot Jumpstart on Target Task
        print(f"Evaluating Zero-Shot Jumpstart Transfer on {target_env}...")
        model_transferred = PPO.load(src_save_path)
        m_jumpstart = eval_policy(model_transferred, target_env)
        
        # 2. Fine-Tune Transferred Model on Target Task
        print(f"Fine-Tuning Transferred Model on {target_env}...")
        tgt_vec_env = make_vec_env(target_env, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_transferred.set_env(tgt_vec_env)
        model_transferred.learn(total_timesteps=steps)
        m_finetuned = eval_policy(model_transferred, target_env)
        tgt_vec_env.close()
        
        # 3. Train Target Task from Scratch (Baseline)
        print(f"Training Target Task ({target_env}) from Scratch...")
        scratch_vec_env = make_vec_env(target_env, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_scratch = PPO("CnnPolicy", scratch_vec_env, verbose=0, seed=seed, n_steps=256)
        model_scratch.learn(total_timesteps=steps)
        m_scratch = eval_policy(model_scratch, target_env)
        scratch_vec_env.close()
        
        results.append({
            "Seed": seed,
            "Scratch_Target_CLE": m_scratch['mean_cle'],
            "ZeroShot_Jumpstart_CLE": m_jumpstart['mean_cle'],
            "FineTuned_Target_CLE": m_finetuned['mean_cle'],
            "Scratch_Success": m_scratch['success_rate'],
            "FineTuned_Success": m_finetuned['success_rate']
        })
        
    df = pd.DataFrame(results)
    df.to_csv("results/tables/cross_task_transfer_results.csv", index=False)
    
    stat_scratch_cle = compute_statistics(df['Scratch_Target_CLE'].values)
    stat_transfer_cle = compute_statistics(df['FineTuned_Target_CLE'].values)
    
    print("\n================ CROSS-TASK TRANSFER SUMMARY ================")
    print(f"Scratch Policy Target CLE:    {stat_scratch_cle['mean']:.2f} ± {stat_scratch_cle['ci_95']:.2f} px")
    print(f"Fine-Tuned Transferred CLE:  {stat_transfer_cle['mean']:.2f} ± {stat_transfer_cle['ci_95']:.2f} px")
    print(f"Error Reduction Delta:      {stat_scratch_cle['mean'] - stat_transfer_cle['mean']:.2f} px improvement")
    print("=============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    
    run_transfer_experiment(steps=args.steps)
