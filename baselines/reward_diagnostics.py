import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.wrappers import SparseRewardStepWrapper
import envs.tracking_envs

def make_sparse_env(env_id, success_threshold):
    def _init():
        env = gym.make(env_id)
        env = SparseRewardStepWrapper(env, success_threshold=success_threshold)
        return env
    return _init

def train_reward_diagnostics(env_id="ActiveTracking-v0", steps=20000, seed=42):
    print("Running Reward Diagnostics...")
    log_dir = "./results/logs_reward_diagnostics"
    
    # 1. Train with Dense Shaped Reward (default in ActiveTracking-v0)
    print("Training with Dense Shaped Reward...")
    dense_env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
    model_dense = PPO("CnnPolicy", dense_env, verbose=0, tensorboard_log=log_dir, seed=seed)
    model_dense.learn(total_timesteps=steps, tb_log_name="DenseReward_PPO")
    
    # 2. Train with Sparse Terminal Reward
    print("Training with Sparse Terminal Reward...")
    sparse_env = SubprocVecEnv([make_sparse_env(env_id, success_threshold=10.0) for _ in range(2)])
    model_sparse = PPO("CnnPolicy", sparse_env, verbose=0, tensorboard_log=log_dir, seed=seed)
    model_sparse.learn(total_timesteps=steps, tb_log_name="SparseReward_PPO")

if __name__ == "__main__":
    train_reward_diagnostics()
