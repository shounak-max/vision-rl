import os
import sys
import time
import paramiko

HOST = "10.0.24.7"
PORT = 2222
USERNAME = "piyush"
PASSWORD = "2£K40d#N"

REMOTE_DIR = "/home/piyush/vision-rl"
LOCAL_WORKSPACE = "d:/gitfork/vision rl"

def connect_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
        transport = ssh.get_transport()
        if transport:
            transport.set_keepalive(15)
        return ssh
    except Exception as e:
        print(f"SSH Connection Error: {e}")
        return None

def execute_remote_cmd_safe(cmd):
    """Executes command on remote GPU server with individual SSH connection and keepalives."""
    ssh = connect_ssh()
    if not ssh:
        return "", "SSH connection failed"
        
    print(f"\n[Remote GPU Exec]: {cmd}")
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
        
        out_lines = []
        while True:
            line = stdout.readline()
            if not line:
                break
            clean_line = line.encode('ascii', 'ignore').decode('ascii').rstrip()
            if clean_line:
                print(clean_line)
                out_lines.append(clean_line)
                
        err = stderr.read().decode('utf-8', errors='ignore').encode('ascii', 'ignore').decode('ascii')
        if err.strip():
            print(f"[Remote Message]: {err}")
        ssh.close()
        return "\n".join(out_lines), err
    except Exception as e:
        print(f"Execution Exception: {e}")
        try:
            ssh.close()
        except:
            pass
        return "", str(e)

def sync_code_to_remote():
    ssh = connect_ssh()
    if not ssh:
        return False
        
    print("\nSyncing local codebase to remote GPU server...")
    sftp = ssh.open_sftp()
    
    execute_remote_cmd_safe(f"mkdir -p {REMOTE_DIR}/envs {REMOTE_DIR}/baselines {REMOTE_DIR}/utils {REMOTE_DIR}/results/tables {REMOTE_DIR}/results/figures {REMOTE_DIR}/results/models {REMOTE_DIR}/results/logs")
    
    local_files = [
        "requirements.txt",
        "envs/__init__.py", "envs/tracking_envs.py", "envs/navigation_envs.py", "envs/wrappers.py",
        "baselines/pretrained_policy.py", "baselines/train_multiseed.py", "baselines/reward_hacking_demonstration.py",
        "baselines/representation_correlation.py", "baselines/evaluate_ood.py", "baselines/run_ablations.py",
        "baselines/train_expanded_baselines.py", "baselines/run_cross_task_transfer.py", "baselines/run_pretrained_experiment.py",
        "baselines/fast_eval_suite.py", "baselines/rmr_ppo.py", "baselines/run_rep_distance_experiment.py",
        "utils/metrics.py", "utils/stats.py", "utils/saliency.py", "utils/visualization.py",
        "utils/eval_pipeline.py", "utils/dataset_partitions.py", "utils/compute_audit.py",
        "results/tables/rep_distance_manifest.json", "baselines/predictive_augmentation_selection.py",
        "baselines/run_scale_experiment.py"
    ]
    
    for rel_path in local_files:
        full_local = os.path.join(LOCAL_WORKSPACE, rel_path)
        if os.path.exists(full_local):
            remote_path = f"{REMOTE_DIR}/{rel_path}"
            sftp.put(full_local, remote_path)
            print(f"  Pushed: {rel_path} -> {remote_path}")
            
    sftp.close()
    ssh.close()
    print("Code Synchronization Complete!")
    return True

def launch_detached_remote_job():
    """Launches the 8.0M-step experiment under nohup so it persists across SSH disconnects."""
    if not sync_code_to_remote():
        print("Aborting launch due to sync error.")
        return

    nohup_cmd = f"nohup bash -c 'source /home/piyush/miniconda3/bin/activate && conda activate rl_env && cd {REMOTE_DIR} && PYTHONPATH=. python3 baselines/predictive_augmentation_selection.py' > {REMOTE_DIR}/results/logs/predictive_stdout.log 2>&1 &"
    
    print("\n=== Launching Detached (nohup) Predictive Script on Remote GPU Server ===")
    out, err = execute_remote_cmd_safe(nohup_cmd)
    print("Detached Remote Execution Command Issued!")
    
    # Wait 3 seconds and verify remote PID is running
    time.sleep(3)
    check_cmd = "ps aux | grep predictive_augmentation_selection | grep -v grep"
    pid_out, _ = execute_remote_cmd_safe(check_cmd)
    if pid_out.strip():
        print(f"VERIFIED REMOTE PID RUNNING:\n{pid_out}")
    else:
        print("Checking experiment_stdout.log...")
        cat_out, _ = execute_remote_cmd_safe(f"tail -n 20 {REMOTE_DIR}/results/logs/predictive_stdout.log")
        print(cat_out)

if __name__ == "__main__":
    launch_detached_remote_job()
