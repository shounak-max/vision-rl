import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import PPO
import envs.tracking_envs
from envs.wrappers import DistractorWrapper

class GradCAMVisualizer:
    """
    Grad-CAM Visualizer for CNN Policies in Stable-Baselines3.
    Extracts class activation heatmaps showing visual attention focus.
    """
    def __init__(self, model):
        self.model = model
        self.policy = model.policy
        self.feature_extractor = self.policy.features_extractor.cnn
        self.gradients = None
        self.activations = None
        
        # Register hooks on final conv layer
        target_layer = self.feature_extractor[4] # 3rd conv layer in SB3 CnnPolicy
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, obs_np):
        obs_tensor = torch.as_tensor(obs_np).unsqueeze(0).float() / 255.0
        obs_tensor.requires_grad = True
        
        # Forward pass through policy
        distribution = self.policy.get_distribution(obs_tensor)
        action_mean = distribution.distribution.mean
        
        # Target gradient w.r.t action norm
        loss = torch.norm(action_mean)
        self.policy.zero_grad()
        loss.backward()
        
        # Grad-CAM computation
        grads = self.gradients.detach().cpu().numpy()[0]
        acts = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        cam = np.maximum(cam, 0) # ReLU
        if np.max(cam) > 0:
            cam = cam / np.max(cam) # Normalize
            
        cam_resized = cv2.resize(cam, (obs_np.shape[2], obs_np.shape[1]))
        return cam_resized

def generate_gradcam_figures(model_path, output_path="results/figures/gradcam_attention.png"):
    """Generates Grad-CAM visual attention overlays under Clean vs Distractor conditions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Skipping Grad-CAM figures.")
        return
        
    model = PPO.load(model_path)
    cam_vis = GradCAMVisualizer(model)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # 1. Clean Condition
    env_clean = gym.make("SingleObjectTracking-v0")
    obs_clean, _ = env_clean.reset(seed=42)
    cam_clean = cam_vis.generate_heatmap(obs_clean)
    
    rgb_clean = np.transpose(obs_clean, (1, 2, 0))
    axes[0].imshow(rgb_clean)
    axes[0].imshow(cam_clean, cmap='jet', alpha=0.5)
    axes[0].set_title("Visual Attention (Clean)", fontweight='bold')
    axes[0].axis('off')
    env_clean.close()
    
    # 2. Distractor Condition
    env_dist = DistractorWrapper(gym.make("SingleObjectTracking-v0"), num_distractors=2)
    obs_dist, _ = env_dist.reset(seed=42)
    cam_dist = cam_vis.generate_heatmap(obs_dist)
    
    rgb_dist = np.transpose(obs_dist, (1, 2, 0))
    axes[1].imshow(rgb_dist)
    axes[1].imshow(cam_dist, cmap='jet', alpha=0.5)
    axes[1].set_title("Visual Attention (Distractors)", fontweight='bold')
    axes[1].axis('off')
    env_dist.close()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Grad-CAM visual attention heatmaps: {output_path}")

if __name__ == "__main__":
    generate_gradcam_figures("results/models/PPO_SingleObjectTracking-v0_s42.zip")
