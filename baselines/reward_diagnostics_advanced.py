import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import pandas as pd
import envs.navigation_envs
import envs # for registration

def train_and_eval(reward_type, steps=50000, seed=42):
    print(f"Training with {reward_type} reward...")
    log_dir = f"./results/logs_reward_diagnostics_adv/{reward_type}"
    
    def make_env():
        return gym.make("MultiStageNavigation-v0", reward_type=reward_type)
        
    env = SubprocVecEnv([make_env for _ in range(4)])
    model = PPO("CnnPolicy", env, verbose=0, tensorboard_log=log_dir, seed=seed, n_steps=512, batch_size=64)
    model.learn(total_timesteps=steps)
    
    # Eval
    eval_env = gym.make("MultiStageNavigation-v0", reward_type=reward_type)
    successes = []
    keys_picked = []
    
    for _ in range(20):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        successes.append(info['success'])
        keys_picked.append(info['has_key'])
        
    eval_env.close()
    
    success_rate = sum(successes) / len(successes)
    key_rate = sum(keys_picked) / len(keys_picked)
    print(f"[{reward_type}] Success Rate: {success_rate:.2f}, Key Picked Rate: {key_rate:.2f}")
    return {"RewardType": reward_type, "SuccessRate": success_rate, "KeyPickedRate": key_rate}

if __name__ == "__main__":
    os.makedirs("./results/logs_reward_diagnostics_adv", exist_ok=True)
    results = []
    
    # 1. Sparse Reward (Hard exploration, might fail to learn without intrinsic motivation)
    res_sparse = train_and_eval("sparse", steps=50000)
    results.append(res_sparse)
    
    # 2. Hackable Shaped Reward (Agent might just hover near key/door for living reward)
    res_shaped = train_and_eval("hackable_shaped", steps=50000)
    results.append(res_shaped)
    
    df = pd.DataFrame(results)
    df.to_csv("./results/reward_hacking_results.csv", index=False)
    print("Saved results to ./results/reward_hacking_results.csv")
    print(df)
