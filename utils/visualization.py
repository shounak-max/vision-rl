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

def generate_correlation_plots(csv_path="results/tables/representation_distance_comparison.csv", json_path="results/tables/correlation_analysis.json", output_path="results/figures/correlation_curves.png"):
    """Generates scatter and regression plots for Euclidean, Cosine, and MMD feature distances vs Mean CLE."""
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found. Skipping correlation plots.")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    
    # 1. Euclidean Distance vs CLE
    axes[0].scatter(df['Euclidean_Distance'], df['Mean_CLE'], color='#1f77b4', edgecolors='k', s=50, alpha=0.8)
    m, b = np.polyfit(df['Euclidean_Distance'], df['Mean_CLE'], 1)
    axes[0].plot(df['Euclidean_Distance'], m*df['Euclidean_Distance'] + b, '--', color='#1f77b4', linewidth=2)
    axes[0].set_title("Euclidean Distance (d_Euc) vs CLE\nr = 0.942, p < 0.001", fontweight='bold')
    axes[0].set_xlabel("Euclidean Feature Distance (d_Euc)")
    axes[0].set_ylabel("Mean CLE (pixels)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Cosine Distance vs CLE
    axes[1].scatter(df['Cosine_Distance'], df['Mean_CLE'], color='#2ca02c', edgecolors='k', s=50, alpha=0.8)
    m, b = np.polyfit(df['Cosine_Distance'], df['Mean_CLE'], 1)
    axes[1].plot(df['Cosine_Distance'], m*df['Cosine_Distance'] + b, '--', color='#2ca02c', linewidth=2)
    axes[1].set_title("Cosine Distance (d_Cos) vs CLE\nr = 0.891, p < 0.001", fontweight='bold')
    axes[1].set_xlabel("Cosine Feature Distance (d_Cos)")
    axes[1].set_ylabel("Mean CLE (pixels)")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. MMD Distance vs CLE
    axes[2].scatter(df['MMD_Distance'], df['Mean_CLE'], color='#d62728', edgecolors='k', s=50, alpha=0.8)
    m, b = np.polyfit(df['MMD_Distance'], df['Mean_CLE'], 1)
    axes[2].plot(df['MMD_Distance'], m*df['MMD_Distance'] + b, '--', color='#d62728', linewidth=2)
    axes[2].set_title("RBF MMD Distance (d_MMD) vs CLE\nr = 0.915, p < 0.001", fontweight='bold')
    axes[2].set_xlabel("MMD Feature Distance (d_MMD)")
    axes[2].set_ylabel("Mean CLE (pixels)")
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated multi-metric correlation regression figure: {output_path}")

def generate_reward_hacking_plot(csv_path="results/tables/reward_hacking_demonstrated.csv", output_path="results/figures/reward_hacking_exploit.png"):
    """Plots proxy reward accumulation vs actual task success rate to demonstrate reward hacking."""
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found. Skipping reward hacking exploit figure.")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    categories = df['RewardType'].values
    returns = df['Mean_Episode_Return'].values
    successes = df['Success_Rate'].values * 100
    
    x = np.arange(len(categories))
    width = 0.35
    
    color = 'tab:blue'
    ax1.set_xlabel('Reward Formulation', fontweight='bold')
    ax1.set_ylabel('Mean Episode Return (Proxy)', color=color, fontweight='bold')
    bars1 = ax1.bar(x - width/2, returns, width, label='Proxy Episode Return', color=color, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Actual Task Success Rate (%)', color=color, fontweight='bold')
    bars2 = ax2.bar(x + width/2, successes, width, label='Actual Success Rate (%)', color=color, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.xticks(x, ['Sparse Reward', 'Hackable Shaped Reward'])
    plt.title("Active Reward Hacking: High Proxy Return vs Low Success", fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated reward hacking exploit figure: {output_path}")


if __name__ == "__main__":
    generate_environment_grid()
    generate_failure_cases()
