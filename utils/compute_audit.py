import os
import sys
import time
import platform
import psutil
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3

def run_hardware_compute_audit():
    print("=== Running Comprehensive Hardware Transparency & Compute Audit ===")
    os.makedirs("results/tables", exist_ok=True)
    
    # 1. Hardware Environment Inventory
    cpu_name = platform.processor() or "Multi-Core CPU"
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A (CPU Execution)"
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    
    sys_info = {
        "OS": platform.platform(),
        "Python_Version": sys.version.split()[0],
        "PyTorch_Version": torch.__version__,
        "CPU_Architecture": cpu_name,
        "Physical_CPU_Cores": physical_cores,
        "Logical_CPU_Cores": logical_cores,
        "System_RAM_GB": ram_gb,
        "CUDA_Available": cuda_available,
        "GPU_Model": gpu_name,
        "CUDA_Version": cuda_version
    }
    
    print("\n--- System Hardware Inventory ---")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")
        
    # 2. Benchmarking Algorithm Parameter Count, Footprint, and Latency
    env = gym.make("SingleObjectTracking-v0")
    dummy_obs = env.reset()[0]
    
    models = {
        "PPO (NatureCNN)": PPO("CnnPolicy", env, verbose=0, n_steps=256),
        "SAC (NatureCNN)": SAC("CnnPolicy", env, verbose=0, buffer_size=1000),
    }
    
    audit_data = []
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        
        # Measure inference latency over 500 steps
        start_time = time.time()
        for _ in range(500):
            _ = model.predict(dummy_obs, deterministic=True)
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000.0
        latency_ms = total_time_ms / 500.0
        fps = 500.0 / (end_time - start_time)
        
        # Save temp checkpoint to measure file footprint MB
        tmp_path = f"results/tables/tmp_{name.replace(' ', '_')}.zip"
        model.save(tmp_path)
        size_mb = round(os.path.getsize(tmp_path) / (1024 * 1024), 2)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        audit_data.append({
            "Algorithm": name,
            "Policy_Parameters": params,
            "Inference_Latency_ms": round(latency_ms, 2),
            "Inference_FPS": round(fps, 1),
            "Model_Size_MB": size_mb,
            "CPU_WallClock_1M_Steps_Est_Hours": round((1000000 / (fps * 60)) / 60, 2)
        })
        
    env.close()
    
    df_audit = pd.DataFrame(audit_data)
    df_audit.to_csv("results/tables/compute_efficiency_audit.csv", index=False)
    
    with open("results/tables/hardware_inventory.json", "w") as f:
        import json
        json.dump(sys_info, f, indent=2)
        
    print("\n================ COMPUTE EFFICIENCY AUDIT ================")
    print(df_audit.to_string(index=False))
    print("==========================================================")

if __name__ == "__main__":
    run_hardware_compute_audit()
