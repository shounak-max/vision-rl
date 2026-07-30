import os
import sys
import time
import paramiko

HOST = "10.0.24.7"
PORT = 2222
USERNAME = "piyush"
PASSWORD = "2£K40d#N"

REMOTE_DIR = "/home/piyush/vision-rl"

def connect_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=10)
        return ssh
    except Exception as e:
        print(f"SSH Error: {e}")
        return None

def execute_remote_cmd(ssh, cmd):
    print(f"\n[Remote Exec] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    
    # Clean non-ASCII for Windows CP1252 console printing
    out_clean = out.encode('ascii', 'ignore').decode('ascii')
    err_clean = err.encode('ascii', 'ignore').decode('ascii')
    
    if out_clean.strip():
        print(out_clean)
    if err_clean.strip():
        print(f"[Remote Message]: {err_clean}")
    return out, err

def sync_code_to_remote(ssh):
    print("\nSyncing local code repository to GPU server...")
    sftp = ssh.open_sftp()
    execute_remote_cmd(ssh, f"mkdir -p {REMOTE_DIR}/envs {REMOTE_DIR}/baselines {REMOTE_DIR}/utils {REMOTE_DIR}/results/tables {REMOTE_DIR}/results/figures {REMOTE_DIR}/results/models")
    
    local_files = [
        "requirements.txt",
        "envs/__init__.py", "envs/tracking_envs.py", "envs/navigation_envs.py", "envs/wrappers.py",
        "baselines/pretrained_policy.py", "baselines/train_multiseed.py", "baselines/reward_hacking_demonstration.py",
        "baselines/representation_correlation.py", "baselines/evaluate_ood.py", "baselines/run_ablations.py",
        "utils/metrics.py", "utils/stats.py", "utils/saliency.py", "utils/visualization.py"
    ]
    
    for rel_path in local_files:
        if os.path.exists(rel_path):
            remote_path = f"{REMOTE_DIR}/{rel_path}"
            sftp.put(rel_path, remote_path)
            
    sftp.close()
    print("Code Sync Complete!")

def main():
    ssh = connect_ssh()
    if not ssh:
        return
        
    # Sync repository
    sync_code_to_remote(ssh)
    
    # 1. Activate conda and test PyTorch on GPUs
    activate_cmd = "source /home/piyush/miniconda3/bin/activate && conda activate rl_env || conda activate darl_k80 || conda activate darl311"
    
    print("\nInstalling/Updating dependencies inside remote Conda environment...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && pip install torch torchvision stable-baselines3 gymnasium opencv-python numpy pandas scipy matplotlib tensorboard scikit-learn'")
    
    print("\nTesting PyTorch CUDA GPU acceleration...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && python3 -c \"import torch; print(\\\"CUDA Available:\\\", torch.cuda.is_available(), \\\"Device Count:\\\", torch.cuda.device_count(), \\\"Device Name:\\\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"None\\\")\"'")
    
    # Launch 100k-step High-Scale GPU Multi-Seed Training
    print("\nLaunching High-Scale Multi-Seed Training (100,000 steps across 5 seeds on GPU)...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && cd {REMOTE_DIR} && PYTHONPATH=. python3 baselines/train_multiseed.py --steps 100000'")
    
    # Fetch output summary
    print("\nFetching Remote Results...")
    execute_remote_cmd(ssh, f"cat {REMOTE_DIR}/results/tables/multiseed_summary.json")
    
    ssh.close()

if __name__ == "__main__":
    main()
