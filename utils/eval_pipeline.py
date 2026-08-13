import os
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from scipy import stats
from utils.metrics import TrackingMetricsLogger, calculate_success_rate, calculate_cle

def compute_stat_ci95(data):
    """Computes mean, std, standard error, and 95% Student-t confidence interval."""
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return {
            "mean": mean,
            "std": 0.0,
            "sem": 0.0,
            "ci_95": 0.0,
            "formatted": f"{mean:.2f} ± 0.00"
        }
    std = float(np.std(arr, ddof=1))
    sem = float(stats.sem(arr))
    ci95 = float(sem * stats.t.ppf((1 + 0.95) / 2.0, n - 1))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_95": ci95,
        "formatted": f"{mean:.2f} ± {ci95:.2f}"
    }

def make_env_canonical(env_id):
    if env_id.startswith("procgen"):
        from envs.wrappers import ProcgenGymnasiumWrapper
        return ProcgenGymnasiumWrapper(env_id)
    return gym.make(env_id)

def evaluate_policy_canonical(model, env_id, n_episodes=20, seed=42, wrapper_cls=None, wrapper_kwargs=None, is_bc=False, bc_policy=None):
    """
    Standardized, canonical evaluation routine for all models across all environments.
    
    Returns:
        dict containing episode rewards, CLEs, success flags, mean metrics, and detailed step history.
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
        
    env = make_env_canonical(env_id)
    if wrapper_cls is not None:
        env = wrapper_cls(env, **wrapper_kwargs)
        
    logger = TrackingMetricsLogger()
    episode_returns = []
    episode_successes = []
    episode_mean_cles = []
    
    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx * 100
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0
        step_cles = []
        
        while not done:
            if model is None and not is_bc:
                action = env.action_space.sample()
            elif is_bc:
                obs_t = torch.as_tensor(obs).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    action = bc_policy(obs_t).numpy()[0]
                action = np.clip(action, -1.0, 1.0)
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            logger.add_step_info(info)
            if 'cle' in info:
                step_cles.append(info['cle'])
            done = terminated or truncated
            
        episode_returns.append(ep_return)
        
        if step_cles:
            ep_cle = float(np.mean(step_cles))
            # Episode success: mean CLE < 10.0 pixels across trajectory
            ep_succ = 1.0 if ep_cle < 10.0 else 0.0
            episode_mean_cles.append(ep_cle)
            episode_successes.append(ep_succ)
        elif 'success' in info:
            episode_successes.append(float(info['success']))
            episode_mean_cles.append(info.get('cle', 0.0))
        elif 'prev_level_complete' in info:
            episode_successes.append(1.0 if info.get('prev_level_complete') == 1 else 0.0)
            episode_mean_cles.append(0.0)
            
    env.close()
    
    overall_metrics = logger.get_episode_metrics()
    
    mean_cle = float(np.mean(episode_mean_cles)) if episode_mean_cles else overall_metrics['mean_cle']
    mean_succ = float(np.mean(episode_successes)) if episode_successes else overall_metrics['success_rate']
    mean_return = float(np.mean(episode_returns))
    
    return {
        "mean_cle": mean_cle,
        "success_rate": mean_succ,
        "mean_return": mean_return,
        "raw_cles": episode_mean_cles,
        "raw_successes": episode_successes,
        "raw_returns": episode_returns
    }
