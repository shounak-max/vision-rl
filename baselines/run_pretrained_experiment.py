import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs
from baselines.pretrained_policy import policy_kwargs_resnet
from utils.eval_pipeline import evaluate_policy_canonical, compute_stat_ci95
from utils.stats import welch_ttest

def run_comparative_experiment(env_id="SingleObjectTracking-v0", steps=30000, seeds=[0, 42, 100, 123, 999]):
    print(f"=== Canonical Comparative Evaluation: Scratch CNN vs Pre-Trained ResNet-18 ({len(seeds)} seeds) ===")
    os.makedirs("results/tables", exist_ok=True)
    
    # 1. Random Baseline
    rand_cles = [evaluate_policy_canonical(None, env_id, seed=s)['mean_cle'] for s in seeds]
    rand_cle_mean = float(np.mean(rand_cles))
    print(f"Random Baseline Mean CLE: {rand_cle_mean:.2f} px")
    
    results = []
    
    # 2. Scratch CNN Policy across seeds
    scratch_cles = []
    scratch_succs = []
    for seed in seeds:
        print(f"\nTraining Scratch CNN Policy (Seed {seed})...")
        vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_scratch = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=512, ent_coef=0.01)
        model_scratch.learn(total_timesteps=steps)
        
        m_scratch = evaluate_policy_canonical(model_scratch, env_id, seed=seed)
        delta_cle_scratch = 1.0 - (m_scratch['mean_cle'] / rand_cle_mean)
        
        scratch_cles.append(m_scratch['mean_cle'])
        scratch_succs.append(m_scratch['success_rate'])
        
        results.append({
            "Architecture": "Scratch NatureCNN (3-Layer)",
            "Seed": seed,
            "Success_Rate": m_scratch['success_rate'],
            "Mean_CLE": m_scratch['mean_cle'],
            "Delta_CLE_Reduction": delta_cle_scratch
        })
        vec_env.close()
        
    # 3. Pre-Trained ResNet-18 Policy across seeds
    resnet_cles = []
    resnet_succs = []
    for seed in seeds:
        print(f"\nTraining Pre-Trained ResNet-18 Policy (Seed {seed})...")
        vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_resnet = PPO("CnnPolicy", vec_env, policy_kwargs=policy_kwargs_resnet, verbose=0, seed=seed, n_steps=512, ent_coef=0.01)
        model_resnet.learn(total_timesteps=steps)
        
        m_resnet = evaluate_policy_canonical(model_resnet, env_id, seed=seed)
        delta_cle_resnet = 1.0 - (m_resnet['mean_cle'] / rand_cle_mean)
        
        resnet_cles.append(m_resnet['mean_cle'])
        resnet_succs.append(m_resnet['success_rate'])
        
        results.append({
            "Architecture": "Pre-Trained ResNet-18 (Frozen Backbone)",
            "Seed": seed,
            "Success_Rate": m_resnet['success_rate'],
            "Mean_CLE": m_resnet['mean_cle'],
            "Delta_CLE_Reduction": delta_cle_resnet
        })
        vec_env.close()
        
    df = pd.DataFrame(results)
    df.to_csv("results/tables/pretrained_vs_scratch_results.csv", index=False)
    
    stat_scratch_cle = compute_stat_ci95(scratch_cles)
    stat_scratch_succ = compute_stat_ci95(scratch_succs)
    
    stat_resnet_cle = compute_stat_ci95(resnet_cles)
    stat_resnet_succ = compute_stat_ci95(resnet_succs)
    
    t_stat, p_val = welch_ttest(resnet_cles, scratch_cles)
    
    summary = {
        "Scratch_CNN_CLE": stat_scratch_cle['formatted'],
        "Scratch_CNN_Success": f"{stat_scratch_succ['mean']*100:.2f} ± {stat_scratch_succ['ci_95']*100:.2f}%",
        "ResNet18_CLE": stat_resnet_cle['formatted'],
        "ResNet18_Success": f"{stat_resnet_succ['mean']*100:.2f} ± {stat_resnet_succ['ci_95']*100:.2f}%",
        "Welch_p_value": float(p_val)
    }
    
    print("\n================ COMPARATIVE BACKBONE RESULTS (5 SEEDS) ================")
    print(f"Scratch CNN CLE:  {summary['Scratch_CNN_CLE']} px (Success: {summary['Scratch_CNN_Success']})")
    print(f"ResNet-18 CLE:    {summary['ResNet18_CLE']} px (Success: {summary['ResNet18_Success']})")
    print(f"Welch t-test p:   {p_val:.4e}")
    print("=========================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()
    
    run_comparative_experiment(steps=args.steps)
