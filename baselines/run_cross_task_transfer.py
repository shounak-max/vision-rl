import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs
from utils.eval_pipeline import evaluate_policy_canonical, compute_stat_ci95
from utils.stats import welch_ttest

def run_transfer_experiment(source_env="SingleObjectTracking-v0", target_env="ActiveTracking-v0", steps=30000, seeds=[0, 42, 100, 123, 999]):
    print(f"=== Cross-Task Transfer Benchmark ({len(seeds)} seeds): {source_env} -> {target_env} ===")
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed}: Training Source Task ({source_env}) ---")
        src_vec_env = make_vec_env(source_env, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_src = PPO("CnnPolicy", src_vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01)
        model_src.learn(total_timesteps=steps)
        
        src_save_path = f"results/models/PPO_source_s{seed}.zip"
        model_src.save(src_save_path)
        src_vec_env.close()
        
        # 1. Evaluate Zero-Shot Jumpstart on Target Task
        print(f"Evaluating Zero-Shot Jumpstart Transfer on {target_env}...")
        model_transferred = PPO.load(src_save_path)
        m_jumpstart = evaluate_policy_canonical(model_transferred, target_env, n_episodes=20, seed=seed)
        
        # 2. Fine-Tune Transferred Model on Target Task
        print(f"Fine-Tuning Transferred Model on {target_env}...")
        tgt_vec_env = make_vec_env(target_env, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_transferred.set_env(tgt_vec_env)
        model_transferred.learn(total_timesteps=steps)
        m_finetuned = evaluate_policy_canonical(model_transferred, target_env, n_episodes=20, seed=seed)
        tgt_vec_env.close()
        
        # 3. Train Target Task from Scratch (Baseline)
        print(f"Training Target Task ({target_env}) from Scratch...")
        scratch_vec_env = make_vec_env(target_env, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_scratch = PPO("CnnPolicy", scratch_vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01)
        model_scratch.learn(total_timesteps=steps)
        m_scratch = evaluate_policy_canonical(model_scratch, target_env, n_episodes=20, seed=seed)
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
    
    stat_scratch_cle = compute_stat_ci95(df['Scratch_Target_CLE'].values)
    stat_transfer_cle = compute_stat_ci95(df['FineTuned_Target_CLE'].values)
    stat_jumpstart_cle = compute_stat_ci95(df['ZeroShot_Jumpstart_CLE'].values)
    
    print("\n================ CROSS-TASK TRANSFER CANONICAL SUMMARY ================")
    print(f"Scratch Policy Target CLE:    {stat_scratch_cle['formatted']} px")
    print(f"Zero-Shot Jumpstart CLE:      {stat_jumpstart_cle['formatted']} px")
    print(f"Fine-Tuned Transferred CLE:   {stat_transfer_cle['formatted']} px")
    print(f"Error Reduction Delta:        {stat_scratch_cle['mean'] - stat_transfer_cle['mean']:.2f} px improvement")
    print("=======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    
    run_transfer_experiment(steps=args.steps)
