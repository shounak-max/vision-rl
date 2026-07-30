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
from utils.stats import compute_statistics, format_stat_string
from utils.metrics import TrackingMetricsLogger

def evaluate_model(model, env_id, n_episodes=10):
    env = gym.make(env_id)
    logger = TrackingMetricsLogger()
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            logger.add_step_info(info)
            done = terminated or truncated
    metrics = logger.get_episode_metrics()
    env.close()
    return metrics

def run_comparative_experiment(env_id="SingleObjectTracking-v0", steps=20000, seeds=[0, 42, 100]):
    print(f"=== Comparative Evaluation: Scratch CNN vs Pre-Trained ResNet-18 on {env_id} ===")
    os.makedirs("results", exist_ok=True)
    
    # 1. Random Baseline CLE
    rand_metrics = [evaluate_model(None, env_id) for _ in range(5)]
    rand_cle_mean = np.mean([m['mean_cle'] for m in rand_metrics])
    print(f"Random Baseline Mean CLE: {rand_cle_mean:.2f} px")
    
    results = []
    
    # 2. Scratch CNN Policy
    for seed in seeds:
        print(f"\nTraining Scratch CNN Policy (Seed {seed})...")
        vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_scratch = PPO("CnnPolicy", vec_env, verbose=0, seed=seed, n_steps=256)
        model_scratch.learn(total_timesteps=steps)
        
        m_scratch = evaluate_model(model_scratch, env_id)
        delta_cle_scratch = 1.0 - (m_scratch['mean_cle'] / rand_cle_mean)
        
        results.append({
            "Architecture": "Scratch CNN (3-Layer)",
            "Seed": seed,
            "Success_Rate": m_scratch['success_rate'],
            "Mean_CLE": m_scratch['mean_cle'],
            "Delta_CLE_Reduction": delta_cle_scratch
        })
        vec_env.close()
        
    # 3. Pre-Trained ResNet-18 Policy
    for seed in seeds:
        print(f"\nTraining Pre-Trained ResNet-18 Policy (Seed {seed})...")
        vec_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
        model_resnet = PPO("CnnPolicy", vec_env, policy_kwargs=policy_kwargs_resnet, verbose=0, seed=seed, n_steps=256)
        model_resnet.learn(total_timesteps=steps)
        
        m_resnet = evaluate_model(model_resnet, env_id)
        delta_cle_resnet = 1.0 - (m_resnet['mean_cle'] / rand_cle_mean)
        
        results.append({
            "Architecture": "Pre-Trained ResNet-18 (Frozen Backbone)",
            "Seed": seed,
            "Success_Rate": m_resnet['success_rate'],
            "Mean_CLE": m_resnet['mean_cle'],
            "Delta_CLE_Reduction": delta_cle_resnet
        })
        vec_env.close()
        
    df = pd.DataFrame(results)
    df.to_csv("results/pretrained_vs_scratch_results.csv", index=False)
    print("\n================ COMPARATIVE BACKBONE RESULTS ================")
    print(df.to_string(index=False))
    print("===============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    
    run_comparative_experiment(steps=args.steps)
