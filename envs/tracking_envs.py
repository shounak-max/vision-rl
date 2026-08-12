import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class SingleObjectTrackingEnv(gym.Env):
    """
    A single object (a circle) moves in 2D space.
    The agent controls a crosshair via continuous velocity (dx, dy).
    Goal: Keep the crosshair centered on the object.
    Observation: Rendered RGB image.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="rgb_array", canvas_size=84):
        super().__init__()
        self.canvas_size = canvas_size
        self.render_mode = render_mode
        self.max_steps = 200

        # Action: (dx, dy)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation: RGB image
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, canvas_size, canvas_size), dtype=np.uint8)

        self.obj_pos = np.array([canvas_size / 2, canvas_size / 2], dtype=np.float32)
        self.obj_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.tracker_pos = np.array([canvas_size / 2, canvas_size / 2], dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Random initial position
        self.obj_pos = self.np_random.uniform(low=10, high=self.canvas_size-10, size=(2,)).astype(np.float32)
        self.tracker_pos = self.np_random.uniform(low=10, high=self.canvas_size-10, size=(2,)).astype(np.float32)
        
        # Random velocity
        angle = self.np_random.uniform(0, 2 * np.pi)
        speed = self.np_random.uniform(1.0, 3.0)
        self.obj_vel = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.step_count += 1
        
        # Update object position with bounce
        self.obj_pos += self.obj_vel
        for i in range(2):
            if self.obj_pos[i] <= 5 or self.obj_pos[i] >= self.canvas_size - 5:
                self.obj_vel[i] *= -1
                self.obj_pos[i] = np.clip(self.obj_pos[i], 5, self.canvas_size - 5)

        # Update tracker position (max speed 5 px/step)
        self.tracker_pos += action * 5.0
        self.tracker_pos = np.clip(self.tracker_pos, 0, self.canvas_size - 1)

        # Reward: negative Euclidean distance
        dist = np.linalg.norm(self.obj_pos - self.tracker_pos)
        reward = -dist / self.canvas_size  # Normalized to roughly [-1, 0]

        # Termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {"cle": dist} # Center Location Error
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        # Draw object (green circle)
        cv2.circle(canvas, tuple(self.obj_pos.astype(int)), 5, (0, 255, 0), -1)
        # Draw tracker (red crosshair)
        tx, ty = self.tracker_pos.astype(int)
        cv2.line(canvas, (tx - 5, ty), (tx + 5, ty), (0, 0, 255), 1)
        cv2.line(canvas, (tx, ty - 5), (tx, ty + 5), (0, 0, 255), 1)
        
        # stable-baselines3 uses channel-first (C, H, W)
        return np.transpose(canvas, (2, 0, 1))

