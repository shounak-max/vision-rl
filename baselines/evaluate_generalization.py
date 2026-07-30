import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import envs.tracking_envs
from envs.wrappers import NoiseWrapper, DistractorWrapper, ViewpointWrapper
from utils.metrics import TrackingMetricsLogger
import pandas as pd

def evaluate_model(model_path, env_id, n_eval_episodes=10, wrapper_class=None, wrapper_kwargs={}):
    print(f"Evaluating {model_path} on {env_id} with wrapper {wrapper_class.__name__ if wrapper_class else 'None'}")
    
    def make_env():
        env = gym.make(env_id)
        if wrapper_class:
            env = wrapper_class(env, **wrapper_kwargs)
        return env
        
    env = make_env()
    model = PPO.load(model_path)
    logger = TrackingMetricsLogger()
    
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            logger.add_step_info(info)
            done = terminated or truncated
            
    metrics = logger.get_episode_metrics()
    env.close()
    return metrics

def run_generalization_tests(model_path, env_id, output_csv="results/tables/generalization_results.csv"):
    os.makedirs("results/tables", exist_ok=True)
    results = []
    
    # Baseline (no shift)
    metrics = evaluate_model(model_path, env_id)
    results.append({"Condition": "Baseline", "Mean_CLE": metrics["mean_cle"], "Success_Rate": metrics["success_rate"]})
    
    # Noise Shift
    metrics = evaluate_model(model_path, env_id, wrapper_class=NoiseWrapper, wrapper_kwargs={"noise_std": 0.2})
    results.append({"Condition": "Noise (std=0.2)", "Mean_CLE": metrics["mean_cle"], "Success_Rate": metrics["success_rate"]})
    
    # Distractor Shift
    metrics = evaluate_model(model_path, env_id, wrapper_class=DistractorWrapper, wrapper_kwargs={"num_distractors": 2})
    results.append({"Condition": "Distractors (n=2)", "Mean_CLE": metrics["mean_cle"], "Success_Rate": metrics["success_rate"]})
    
    # Viewpoint Shift
    metrics = evaluate_model(model_path, env_id, wrapper_class=ViewpointWrapper, wrapper_kwargs={"max_angle": 30})
    results.append({"Condition": "Viewpoint (max_angle=30)", "Mean_CLE": metrics["mean_cle"], "Success_Rate": metrics["success_rate"]})
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--env", type=str, default="SingleObjectTracking-v0")
    args = parser.parse_args()
    
    run_generalization_tests(args.model, args.env)
