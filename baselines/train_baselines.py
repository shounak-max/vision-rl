import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs # register envs

def train_agent(algo_name, env_id, total_timesteps, seed, log_dir):
    print(f"Training {algo_name} on {env_id} for {total_timesteps} steps (Seed: {seed})")
    
    # Create vectorized environments
    env = make_vec_env(env_id, n_envs=4, seed=seed, vec_env_cls=SubprocVecEnv)
    
    if algo_name.upper() == "PPO":
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, seed=seed,
                    learning_rate=3e-4, n_steps=512, batch_size=64)
    elif algo_name.upper() == "SAC":
        # SAC requires continuous action space
        model = SAC("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, seed=seed,
                    learning_rate=3e-4, buffer_size=50000, batch_size=64)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
        
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"{algo_name}_{env_id}_s{seed}")
    
    # Save model
    save_dir = os.path.join(log_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"{algo_name}_{env_id}_s{seed}"))
    
    print(f"Training completed. Model saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--env", type=str, default="SingleObjectTracking-v0")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="./results/logs")
    args = parser.parse_args()
    
    train_agent(args.algo, args.env, args.steps, args.seed, args.log_dir)
