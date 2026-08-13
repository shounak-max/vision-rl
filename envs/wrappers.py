import gymnasium as gym
import numpy as np
import cv2

class NoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std * 255.0

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, obs.shape)
        obs_noisy = np.clip(obs + noise, 0, 255).astype(np.uint8)
        return obs_noisy

class DistractorWrapper(gym.ObservationWrapper):
    """Adds randomly moving distractors (shapes) to the observation."""
    def __init__(self, env, num_distractors=2):
        super().__init__(env)
        self.num_distractors = num_distractors
        self.distractor_pos = None
        self.distractor_vel = None
        self.colors = [(255, 0, 0), (0, 0, 255), (255, 255, 255)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        _, h, w = obs.shape
        self.distractor_pos = np.random.uniform(0, h, size=(self.num_distractors, 2)).astype(np.float32)
        angles = np.random.uniform(0, 2*np.pi, size=self.num_distractors)
        speeds = np.random.uniform(1.0, 4.0, size=self.num_distractors)
        self.distractor_vel = np.vstack([np.cos(angles)*speeds, np.sin(angles)*speeds]).T
        return self.observation(obs), info

    def observation(self, obs):
        if self.distractor_pos is None:
            return obs
            
        _, h, w = obs.shape
        
        # update positions
        self.distractor_pos += self.distractor_vel
        for i in range(self.num_distractors):
            for j in range(2):
                if self.distractor_pos[i, j] <= 0 or self.distractor_pos[i, j] >= (h if j==1 else w):
                    self.distractor_vel[i, j] *= -1
                    self.distractor_pos[i, j] = np.clip(self.distractor_pos[i, j], 1, (h-1 if j==1 else w-1))

        # draw
        obs_cv = np.transpose(obs, (1, 2, 0)).copy()
        for i in range(self.num_distractors):
            color = self.colors[i % len(self.colors)]
            cv2.rectangle(obs_cv, 
                          tuple((self.distractor_pos[i] - 4).astype(int)), 
                          tuple((self.distractor_pos[i] + 4).astype(int)), 
                          color, -1)
                          
        return np.transpose(obs_cv, (2, 0, 1))

class ViewpointWrapper(gym.ObservationWrapper):
    """Applies affine transformations (rotation) to simulate viewpoint changes."""
    def __init__(self, env, max_angle=15):
        super().__init__(env)
        self.max_angle = max_angle

    def observation(self, obs):
        obs_cv = np.transpose(obs, (1, 2, 0))
        h, w = obs_cv.shape[:2]
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        obs_rot = cv2.warpAffine(obs_cv, M, (w, h))
        return np.transpose(obs_rot, (2, 0, 1))
        
class SparseRewardWrapper(gym.RewardWrapper):
    """Converts dense shaped reward to sparse terminal reward for diagnostics."""
    def __init__(self, env, threshold_dist=10.0, max_steps=200):
        super().__init__(env)
        self.threshold_dist = threshold_dist
        self.max_steps = max_steps
        self.success_steps = 0

    def reset(self, **kwargs):
        self.success_steps = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        # We need info from the environment, but gym RewardWrapper only takes reward.
        # So we'll use step wrapper instead.
        pass

class SparseRewardStepWrapper(gym.Wrapper):
    """Replaces dense reward with sparse terminal reward based on success."""
    def __init__(self, env, success_threshold=10.0):
        super().__init__(env)
        self.success_threshold = success_threshold
        self.steps_in_threshold = 0

    def reset(self, **kwargs):
        self.steps_in_threshold = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # 'cle' is Center Location Error
        if 'cle' in info and info['cle'] < self.success_threshold:
            self.steps_in_threshold += 1
            
        reward = 0.0
        if terminated or truncated:
            # e.g., if agent stayed near target for > 50% of the episode
            if getattr(self.env.unwrapped, 'max_steps', 200) > 0:
                success_rate = self.steps_in_threshold / self.env.unwrapped.max_steps
                if success_rate > 0.5:
                    reward = 1.0
                    
        info['sparse_reward'] = reward
        return obs, reward, terminated, truncated, info

class RandomShiftWrapper(gym.ObservationWrapper):
    """Applies Random Shift augmentation."""
    def __init__(self, env, max_shift=4):
        super().__init__(env)
        self.max_shift = max_shift

    def observation(self, obs):
        obs_cv = np.transpose(obs, (1, 2, 0)).astype(np.float32)
        h, w = obs_cv.shape[:2]
        shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
        shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        obs_cv = cv2.warpAffine(obs_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return np.transpose(obs_cv, (2, 0, 1))

class ColorJitterWrapper(gym.ObservationWrapper):
    """Applies Color Jitter augmentation."""
    def __init__(self, env, brightness=0.2, contrast=0.2):
        super().__init__(env)
        self.brightness = brightness
        self.contrast = contrast

    def observation(self, obs):
        obs_cv = np.transpose(obs, (1, 2, 0)).astype(np.float32)
        alpha = np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        beta = np.random.uniform(-self.brightness, self.brightness) * 255
        obs_cv = cv2.convertScaleAbs(obs_cv, alpha=alpha, beta=beta)
        return np.transpose(obs_cv, (2, 0, 1))

class CutoutWrapper(gym.ObservationWrapper):
    """Applies Cutout augmentation."""
    def __init__(self, env, cutout_ratio=0.15):
        super().__init__(env)
        self.cutout_ratio = cutout_ratio

    def observation(self, obs):
        obs_cv = np.transpose(obs, (1, 2, 0)).copy()
        h, w = obs_cv.shape[:2]
        patch_size = int(np.sqrt(self.cutout_ratio) * min(h, w))
        patch_size = max(1, patch_size)
        top = np.random.randint(0, max(1, h - patch_size + 1))
        left = np.random.randint(0, max(1, w - patch_size + 1))
        # Cutout typically fills with mean color or black (0)
        obs_cv[top:top+patch_size, left:left+patch_size] = 0
        return np.transpose(obs_cv, (2, 0, 1))

class DataAugmentationWrapper(gym.ObservationWrapper):
    """Applies combined data augmentations (Random Shift + Color Jitter)."""
    def __init__(self, env, max_shift=4, jitter_brightness=0.2, jitter_contrast=0.2):
        super().__init__(env)
        self.max_shift = max_shift
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast

    def observation(self, obs):
        obs_cv = np.transpose(obs, (1, 2, 0)).astype(np.float32)
        h, w = obs_cv.shape[:2]

        # 1. Color Jitter
        alpha = np.random.uniform(1.0 - self.jitter_contrast, 1.0 + self.jitter_contrast)
        beta = np.random.uniform(-self.jitter_brightness, self.jitter_brightness) * 255
        obs_cv = cv2.convertScaleAbs(obs_cv, alpha=alpha, beta=beta)

        # 2. Random Shift
        shift_x = np.random.randint(-self.max_shift, self.max_shift + 1)
        shift_y = np.random.randint(-self.max_shift, self.max_shift + 1)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        obs_cv = cv2.warpAffine(obs_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return np.transpose(obs_cv, (2, 0, 1))

class OcclusionWrapper(gym.ObservationWrapper):
    """Applies random rectangular spatial occlusion patches to observations."""
    def __init__(self, env, occlusion_ratio=0.10):
        super().__init__(env)
        self.occlusion_ratio = float(occlusion_ratio)

    def observation(self, obs):
        obs_cv = np.transpose(obs, (1, 2, 0)).copy()
        h, w = obs_cv.shape[:2]
        patch_size = int(np.sqrt(self.occlusion_ratio) * min(h, w))
        patch_size = max(1, patch_size)
        
        top = np.random.randint(0, max(1, h - patch_size + 1))
        left = np.random.randint(0, max(1, w - patch_size + 1))
        
        obs_cv[top:top+patch_size, left:left+patch_size] = 0
        return np.transpose(obs_cv, (2, 0, 1))

class BlurWrapper(gym.ObservationWrapper):
    """Applies spatial box blurring to observations."""
    def __init__(self, env, kernel_size=5):
        super().__init__(env)
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1
        self.kernel_size = max(3, k)

    def observation(self, obs):
        obs_cv = np.transpose(obs, (1, 2, 0))
        obs_blur = cv2.blur(obs_cv, (self.kernel_size, self.kernel_size))
        return np.transpose(obs_blur, (2, 0, 1))

class CompoundShiftWrapper(gym.ObservationWrapper):
    """Applies combined multi-axis visual shifts (Noise + Distractors + Viewpoint Rotation)."""
    def __init__(self, env, severity_level=1):
        super().__init__(env)
        self.severity_level = int(severity_level)
        
        # Severity tiers: Level 1 to 5
        level_params = {
            1: {"noise_std": 0.05, "num_distractors": 1, "max_angle": 5},
            2: {"noise_std": 0.10, "num_distractors": 2, "max_angle": 10},
            3: {"noise_std": 0.15, "num_distractors": 2, "max_angle": 15},
            4: {"noise_std": 0.20, "num_distractors": 3, "max_angle": 20},
            5: {"noise_std": 0.25, "num_distractors": 4, "max_angle": 25}
        }
        params = level_params.get(self.severity_level, level_params[1])
        
        # Wrap underlying env
        wrapped_env = NoiseWrapper(env, noise_std=params["noise_std"])
        wrapped_env = DistractorWrapper(wrapped_env, num_distractors=params["num_distractors"])
        self.compound_env = ViewpointWrapper(wrapped_env, max_angle=params["max_angle"])

    def reset(self, **kwargs):
        return self.compound_env.reset(**kwargs)

    def step(self, action):
        return self.compound_env.step(action)

    def observation(self, obs):
        return self.compound_env.observation(obs)

class ProcgenGymnasiumWrapper(gym.Env):
    """
    Adapter wrapper converting legacy gym Procgen env into Gymnasium format
    with channel-first observation (C, H, W) for SB3 CnnPolicy.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_id="procgen:procgen-coinrun-v0", distribution_mode="easy"):
        super().__init__()
        import gym as legacy_gym
        self._env = legacy_gym.make(env_id, distribution_mode=distribution_mode)
        obs_space = self._env.observation_space
        act_space = self._env.action_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(act_space.n)

    def reset(self, seed=None, options=None):
        if seed is not None:
            try:
                self._env.seed(int(seed))
            except Exception:
                pass
        obs = self._env.reset()
        obs_chw = np.transpose(obs, (2, 0, 1))
        return obs_chw, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs_chw = np.transpose(obs, (2, 0, 1))
        
        # Disambiguate termination vs truncation
        is_level_complete = info.get("prev_level_complete", 0) == 1
        is_time_limit = info.get("TimeLimit.truncated", False) or info.get("truncated", False)
        
        if is_time_limit:
            terminated = False
            truncated = True
        elif is_level_complete:
            terminated = True
            truncated = False
        else:
            terminated = done
            truncated = False
            
        return obs_chw, reward, terminated, truncated, info

    def render(self):
        return self._env.render(mode="rgb_array")

    def close(self):
        self._env.close()


