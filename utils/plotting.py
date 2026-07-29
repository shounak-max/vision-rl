import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_scalars(log_dir, tag):
    """Extract scalars from a tensorboard log directory."""
    try:
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except Exception as e:
        print(f"Error extracting {tag} from {log_dir}: {e}")
        return [], []

def plot_learning_curves(logs_base_dir="results/logs", output_file="results/sample_efficiency.png"):
    plt.figure(figsize=(10, 6))
    
    # We expect subdirectories like PPO_SingleObjectTracking-v0_s42_1
    if not os.path.exists(logs_base_dir):
        print(f"Logs dir {logs_base_dir} not found.")
        return
        
    for run_dir in os.listdir(logs_base_dir):
        full_path = os.path.join(logs_base_dir, run_dir)
        if os.path.isdir(full_path):
            steps, rewards = extract_tensorboard_scalars(full_path, "rollout/ep_rew_mean")
            if steps:
                plt.plot(steps, rewards, label=run_dir.split('_')[0] + " " + run_dir.split('_')[1])
                
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Episode Reward")
    plt.title("Sample Efficiency Curves (Vision-Based RL)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")
    plt.close()

def plot_reward_diagnostics(logs_base_dir="results/logs_reward_diagnostics", output_file="results/reward_diagnostics.png"):
    plt.figure(figsize=(10, 6))
    if not os.path.exists(logs_base_dir):
        print(f"Logs dir {logs_base_dir} not found.")
        return
        
    for run_dir in os.listdir(logs_base_dir):
        full_path = os.path.join(logs_base_dir, run_dir)
        if os.path.isdir(full_path):
            # Plot reward
            steps, rewards = extract_tensorboard_scalars(full_path, "rollout/ep_rew_mean")
            if steps:
                plt.plot(steps, rewards, label=run_dir.split('_')[0])
                
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Episode Reward")
    plt.title("Reward Diagnostics (Sparse vs Dense)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")
    plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    plot_learning_curves()
    plot_reward_diagnostics()
