import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gymnasium as gym
import torch
from stable_baselines3 import PPO
import envs.tracking_envs
import envs.navigation_envs
from envs.wrappers import NoiseWrapper, DistractorWrapper, ViewpointWrapper

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

def generate_environment_grid(output_path="results/figures/env_grid.png"):
    """Generates a 2x2 grid of rendered observations for the 4 environments."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    envs = [
        ("Single Object Tracking", gym.make("SingleObjectTracking-v0")),
        ("Multi Object Tracking", gym.make("MultiObjectTracking-v0")),
        ("Active Tracking (Viewport)", gym.make("ActiveTracking-v0")),
        ("Multi-Stage Navigation", gym.make("MultiStageNavigation-v0"))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    
    for idx, (title, env) in enumerate(envs):
        obs, _ = env.reset(seed=42)
        # obs is C, H, W -> convert to H, W, C
        obs_rgb = np.transpose(obs, (1, 2, 0))
        axes[idx].imshow(obs_rgb)
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].axis('off')
        env.close()
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated environment grid figure: {output_path}")

def generate_tsne_plot(model_path, env_id="SingleObjectTracking-v0", output_path="results/figures/tsne_features.png"):
    """Generates a t-SNE / PCA plot of CNN feature representations across shifts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Skipping t-SNE plot.")
        return
        
    model = PPO.load(model_path)
    feature_extractor = model.policy.features_extractor
    feature_extractor.eval()
    
    conditions = [
        ("Clean Baseline", gym.make(env_id), '#2ca02c'), # Green
        ("Noise (std=0.2)", NoiseWrapper(gym.make(env_id), noise_std=0.2), '#d62728'), # Red
        ("Distractors (n=2)", DistractorWrapper(gym.make(env_id), num_distractors=2), '#1f77b4'), # Blue
        ("Viewpoint (max_angle=30)", ViewpointWrapper(gym.make(env_id), max_angle=30), '#ff7f0e') # Orange
    ]
    
    all_features = []
    labels = []
    colors = []
    
    for name, env, color in conditions:
        for _ in range(50):
            obs, _ = env.reset()
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                feat = feature_extractor(obs_tensor).numpy()[0]
            all_features.append(feat)
            labels.append(name)
            colors.append(color)
        env.close()
        
    all_features = np.array(all_features)
    
    # Run PCA / t-SNE
    if all_features.shape[1] > 2:
        pca = PCA(n_components=2)
        embeds = pca.fit_transform(all_features)
    else:
        embeds = all_features
        
    plt.figure(figsize=(7, 5))
    unique_labels = list(set(labels))
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeds[idx, 0], embeds[idx, 1], label=label, alpha=0.8, edgecolors='w', s=60)
        
    plt.title("CNN Feature Embedding Shift (PCA Projection)", fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated feature embedding plot: {output_path}")

def generate_degradation_curves(csv_path="results/ood_corruption_results.csv", output_path="results/figures/degradation_curves.png"):
    """Plots success rate degradation against corruption severity."""
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found. Skipping degradation curves.")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # 1. Noise
    df_noise = df[df['Corruption'] == 'NoiseWrapper']
    axes[0].plot(df_noise['Severity_Value'], df_noise['Success_Rate'] * 100, 'o-', color='#d62728', linewidth=2)
    axes[0].set_title("Gaussian Noise Severity")
    axes[0].set_xlabel("Noise Std (σ)")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Distractors
    df_dist = df[df['Corruption'] == 'DistractorWrapper']
    axes[1].plot(df_dist['Severity_Value'], df_dist['Success_Rate'] * 100, 's-', color='#1f77b4', linewidth=2)
    axes[1].set_title("Visual Distractors")
    axes[1].set_xlabel("Number of Distractors (N)")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Viewpoint
    df_view = df[df['Corruption'] == 'ViewpointWrapper']
    axes[2].plot(df_view['Severity_Value'], df_view['Success_Rate'] * 100, '^--', color='#ff7f0e', linewidth=2)
    axes[2].set_title("Viewpoint Rotation")
    axes[2].set_xlabel("Max Rotation Angle (degrees)")
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated degradation curves figure: {output_path}")

def generate_failure_cases(output_path="results/figures/failure_cases.png"):
    """Generates a matrix of illustrative failure cases (distractor confusion, boundary clipping)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    # 1. Distractor Confusion
    env = DistractorWrapper(gym.make("SingleObjectTracking-v0"), num_distractors=3)
    obs, _ = env.reset(seed=12)
    axes[0].imshow(np.transpose(obs, (1, 2, 0)))
    axes[0].set_title("Failure Case 1:\nDistractor Confusion", color='red', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    env.close()
    
    # 2. Viewpoint Shear
    env = ViewpointWrapper(gym.make("SingleObjectTracking-v0"), max_angle=45)
    obs, _ = env.reset(seed=99)
    axes[1].imshow(np.transpose(obs, (1, 2, 0)))
    axes[1].set_title("Failure Case 2:\nViewpoint Distortion", color='red', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    env.close()
    
    # 3. High Noise Occlusion
    env = NoiseWrapper(gym.make("SingleObjectTracking-v0"), noise_std=0.35)
    obs, _ = env.reset(seed=42)
    axes[2].imshow(np.transpose(obs, (1, 2, 0)))
    axes[2].set_title("Failure Case 3:\nHigh Noise Occlusion", color='red', fontsize=10, fontweight='bold')
    axes[2].axis('off')
    env.close()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated failure cases figure: {output_path}")

if __name__ == "__main__":
    generate_environment_grid()
    generate_failure_cases()
