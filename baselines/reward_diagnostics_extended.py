import os
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.navigation_envs

def make_monitored_env(reward_type, seed):
    def _init():
        env = gym.make("MultiStageNavigation-v0", reward_type=reward_type)
        env = Monitor(env)
        return env
    return _init

def run_extended_reward_diagnostics(steps=150000, seed=42):
    print(f"=== Extended Reward Hacking Diagnostic ({steps} steps) ===")
    os.makedirs("results/logs_reward_ext", exist_ok=True)
    
    results = []
    for reward_type in ["sparse", "hackable_shaped"]:
        print(f"\nTraining PPO with [{reward_type}] reward for {steps} steps...")
        env = SubprocVecEnv([make_monitored_env(reward_type, seed + i) for i in range(2)])
        model = PPO("CnnPolicy", env, verbose=0, seed=seed, n_steps=512, batch_size=64)
        model.learn(total_timesteps=steps)
        
        # Evaluate
        eval_env = gym.make("MultiStageNavigation-v0", reward_type=reward_type)
        successes = []
        key_pickups = []
        
        for _ in range(20):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            successes.append(info['success'])
            key_pickups.append(info['has_key'])
            
        eval_env.close()
        vec_succ = np.mean(successes)
        vec_key = np.mean(key_pickups)
        print(f"[{reward_type}] Final Success Rate: {vec_succ:.4f}, Key Picked Rate: {vec_key:.4f}")
        results.append({
            "RewardType": reward_type,
            "Timesteps": steps,
            "SuccessRate": vec_succ,
            "KeyPickedRate": vec_key
        })
        env.close()
        
    df = pd.DataFrame(results)
    df.to_csv("results/extended_reward_hacking_results.csv", index=False)
    print("\nSaved extended reward hacking results to results/extended_reward_hacking_results.csv")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150000)
    args = parser.parse_args()
    
    run_extended_reward_diagnostics(args.steps)
