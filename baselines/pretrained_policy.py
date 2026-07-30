import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class PretrainedVisionFeatureExtractor(BaseFeaturesExtractor):
    """
    Deep Residual Vision Feature Extractor for Visual RL.
    Uses deep residual spatial blocks with initialized feature embeddings 
    to bypass the 3-layer CNN representation bottleneck.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Deep Residual Architecture (ResNet-like block)
        self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResBlock 1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.shortcut1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        
        # ResBlock 2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.shortcut2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            x = observations.float() / 255.0
        else:
            x = observations
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        r1 = self.res_block1(x) + self.shortcut1(x)
        x = self.relu(r1)
        
        r2 = self.res_block2(x) + self.shortcut2(x)
        x = self.relu(r2)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

policy_kwargs_resnet = dict(
    features_extractor_class=PretrainedVisionFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)
