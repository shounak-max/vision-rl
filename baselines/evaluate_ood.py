import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
import envs.tracking_envs
from envs.wrappers import NoiseWrapper, DistractorWrapper, ViewpointWrapper
from utils.metrics import TrackingMetricsLogger

def evaluate_severity(model, env_id, wrapper_cls, param_name, param_values):
    results = []
    for val in param_values:
        kwargs = {param_name: val}
        env = gym.make(env_id)
        if wrapper_cls:
            env = wrapper_cls(env, **kwargs)
            
        logger = TrackingMetricsLogger()
        for _ in range(10):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                logger.add_step_info(info)
                done = terminated or truncated
                
        metrics = logger.get_episode_metrics()
        results.append({
            "Corruption": wrapper_cls.__name__ if wrapper_cls else "Clean",
            "Severity_Param": param_name,
            "Severity_Value": val,
            "Success_Rate": metrics['success_rate'],
            "Mean_CLE": metrics['mean_cle']
        })
        env.close()
    return results

def run_ood_evaluations(model_path, env_id="SingleObjectTracking-v0"):
    print(f"=== OOD Corruption Evaluation for {model_path} ===")
    model = PPO.load(model_path)
    
    all_results = []
    
    # 1. Noise Severities
    print("Evaluating Noise Severities...")
    noise_vals = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    all_results.extend(evaluate_severity(model, env_id, NoiseWrapper, "noise_std", noise_vals))
    
    # 2. Distractor Severities
    print("Evaluating Distractor Severities...")
    distractor_vals = [0, 1, 2, 3, 4]
    all_results.extend(evaluate_severity(model, env_id, DistractorWrapper, "num_distractors", distractor_vals))
    
    # 3. Viewpoint Severities
    print("Evaluating Viewpoint Severities...")
    viewpoint_vals = [0, 10, 20, 30, 45]
    all_results.extend(evaluate_severity(model, env_id, ViewpointWrapper, "max_angle", viewpoint_vals))
    
    df = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ood_corruption_results.csv", index=False)
    print("Saved OOD corruption results to results/ood_corruption_results.csv")
    print(df.head(15))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--env", type=str, default="SingleObjectTracking-v0")
    args = parser.parse_args()
    
    run_ood_evaluations(args.model, args.env)
