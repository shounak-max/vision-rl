import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.tracking_envs

def train_ire_vla_lite(env_id="ActiveTracking-v0", offline_data_path="results/offline_dataset.npz", total_iterations=10, rl_steps_per_iter=2048, sl_epochs_per_iter=2, seed=42):
    print("Starting iRe-VLA-lite Alternating Training...")
    
    # 1. Load Offline Data
    if not os.path.exists(offline_data_path):
        print(f"Error: Offline data not found at {offline_data_path}")
        return
        
    data = np.load(offline_data_path)
    obs_data = data['observations']
    act_data = data['actions']
    
    dataset_size = len(obs_data)
    print(f"Loaded {dataset_size} offline transitions.")
    
    # 2. Init Env and Model
    env = make_vec_env(env_id, n_envs=2, seed=seed, vec_env_cls=SubprocVecEnv)
    model = PPO("CnnPolicy", env, verbose=0, seed=seed, n_steps=rl_steps_per_iter // 2)
    
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
    
    for iteration in range(total_iterations):
        print(f"--- Iteration {iteration+1}/{total_iterations} ---")
        
        # Phase A: Online RL Update
        print(f"  [RL Phase] Training for {rl_steps_per_iter} steps...")
        model.learn(total_timesteps=rl_steps_per_iter, reset_num_timesteps=False)
        
        # Phase B: Supervised Learning (Behavior Cloning) on Offline Data
        print(f"  [SL Phase] Training for {sl_epochs_per_iter} epochs...")
        model.policy.train()
        
        batch_size = 64
        indices = np.arange(dataset_size)
        
        for epoch in range(sl_epochs_per_iter):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            num_batches = 0
            
            for start_idx in range(0, dataset_size, batch_size):
                batch_idx = indices[start_idx:start_idx+batch_size]
                
                # SB3 uses (C, H, W) and expects inputs in [0, 255] scaled internally, or floats.
                # Actually SB3 extracts float in [0, 255] and internally divides by 255 if it's an image.
                obs_batch = torch.as_tensor(obs_data[batch_idx]).float().to(model.device)
                target_acts = torch.as_tensor(act_data[batch_idx]).float().to(model.device)
                
                # Predict action
                distribution = model.policy.get_distribution(obs_batch)
                pred_acts = distribution.mode()
                
                loss = F.mse_loss(pred_acts, target_acts)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            print(f"    Epoch {epoch+1} Loss: {epoch_loss/num_batches:.4f}")
            
    # Save final model
    os.makedirs("results/models", exist_ok=True)
    save_path = f"results/models/iRe_VLA_{env_id}"
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ActiveTracking-v0")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    
    train_ire_vla_lite(args.env, total_iterations=args.iters)
