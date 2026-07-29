import os
import argparse
import gymnasium as gym
import numpy as np
import envs.tracking_envs

def generate_offline_dataset(env_id="ActiveTracking-v0", num_transitions=5000, output_file="results/offline_dataset.npz"):
    print(f"Generating V-D4RL-lite offline dataset using {env_id}...")
    env = gym.make(env_id)
    
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    obs, _ = env.reset()
    
    for i in range(num_transitions):
        # Random policy for baseline dataset
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        terminals.append(terminated or truncated)
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} / {num_transitions} transitions")
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(
        output_file,
        observations=np.array(observations, dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        next_observations=np.array(next_observations, dtype=np.uint8),
        terminals=np.array(terminals, dtype=bool)
    )
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ActiveTracking-v0")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    
    generate_offline_dataset(args.env, args.steps)
