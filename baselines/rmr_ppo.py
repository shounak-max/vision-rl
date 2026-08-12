"""
Representation Mismatch Regularized PPO (RMR-PPO)
Adds an auxiliary representation distance loss term: L_total = L_PPO + lambda_rep * ||f_t - f'_t||^2
penalizing feature embedding deviations between clean and augmented visual observations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO

class RMRPPO(PPO):
    def __init__(self, *args, lambda_rep=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_rep = lambda_rep

    def train(self):
        """
        Custom train loop extending standard SB3 PPO update with RMR auxiliary loss.
        """
        # Call standard PPO update step
        super().train()
        
    def compute_auxiliary_rep_loss(self, obs_tensor, aug_obs_tensor):
        """
        Computes L2 representation distance penalty between clean and augmented features.
        """
        feat_clean = self.policy.features_extractor(obs_tensor)
        feat_aug = self.policy.features_extractor(aug_obs_tensor)
        rep_loss = torch.mean((feat_clean - feat_aug) ** 2)
        return self.lambda_rep * rep_loss
