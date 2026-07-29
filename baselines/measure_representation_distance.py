import os
import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
import envs.tracking_envs
from envs.wrappers import NoiseWrapper, DistractorWrapper, ViewpointWrapper

def measure_distance(model_path, env_id):
    print(f"Measuring representation distance for {model_path} on {env_id}...")
    model = PPO.load(model_path)
    
    # We will use the CNN feature extractor of the loaded policy
    feature_extractor = model.policy.features_extractor
    feature_extractor.eval()
    
    env_clean = gym.make(env_id)
    env_noise = NoiseWrapper(gym.make(env_id), noise_std=0.2)
    env_distractor = DistractorWrapper(gym.make(env_id), num_distractors=2)
    env_viewpoint = ViewpointWrapper(gym.make(env_id), max_angle=30)
    
    def get_features(env, n_obs=100):
        features = []
        for _ in range(n_obs):
            obs, _ = env.reset()
            # convert to torch tensor, add batch dim
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                feat = feature_extractor(obs_tensor)
            features.append(feat.numpy()[0])
        return np.array(features)
        
    print("Collecting clean features...")
    feat_clean = get_features(env_clean)
    
    distances = []
    
    def calc_distance(feat_a, feat_b):
        # Mean L2 distance between paired observations is not possible because environments are stochastic.
        # We compute the distance between the mean feature vectors (centroid distance).
        centroid_a = np.mean(feat_a, axis=0)
        centroid_b = np.mean(feat_b, axis=0)
        return np.linalg.norm(centroid_a - centroid_b)
        
    conditions = [
        ("Noise (std=0.2)", env_noise),
        ("Distractors (n=2)", env_distractor),
        ("Viewpoint (max_angle=30)", env_viewpoint)
    ]
    
    for name, env_shifted in conditions:
        print(f"Collecting features for {name}...")
        feat_shifted = get_features(env_shifted)
        dist = calc_distance(feat_clean, feat_shifted)
        distances.append({"Condition": name, "Representation_Distance": dist})
        
    df = pd.DataFrame(distances)
    df.to_csv("results/representation_distances.csv", index=False)
    print("Saved distances to results/representation_distances.csv")
    print(df)
    
    # merge with generalization results if it exists
    if os.path.exists("results/generalization_results.csv"):
        gen_df = pd.read_csv("results/generalization_results.csv")
        merged = pd.merge(gen_df, df, on="Condition", how="left")
        merged.to_csv("results/generalization_degradation.csv", index=False)
        print("Merged with generalization results -> results/generalization_degradation.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--env", type=str, default="SingleObjectTracking-v0")
    args = parser.parse_args()
    
    measure_distance(args.model, args.env)
