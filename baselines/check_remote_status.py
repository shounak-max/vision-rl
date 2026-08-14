import os
import sys
import json
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
        ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=15)
        return ssh
    except Exception as e:
        print(f"SSH Connection Error: {e}")
        return None

def check_remote_status():
    ssh = connect_ssh()
    if not ssh:
        print("Failed to connect to Remote GPU server.")
        return
        
    print(f"=== REMOTE GPU EXPERIMENT STATUS CHECK ({HOST}:{PORT}) ===")
    
    # 1. Check process PID
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep run_scale_experiment | grep -v grep")
    pid_output = stdout.read().decode('ascii', errors='ignore').strip()
    if pid_output:
        print(f"[PROCESS STATUS]: ACTIVE (RUNNING)")
        print(f"  PID Line: {pid_output.splitlines()[0]}")
    else:
        print(f"[PROCESS STATUS]: STOPPED / COMPLETED")
        
    # 2. Check saved checkpoints
    sftp = ssh.open_sftp()
    try:
        models = sftp.listdir(f"{REMOTE_DIR}/results/models")
        chkpt_count = len([m for m in models if m.endswith('.zip')])
        print(f"[CHECKPOINTS SAVED]: {chkpt_count} model checkpoint files in results/models/")
        for m in sorted(models)[:10]:
            print(f"  - {m}")
    except Exception as e:
        print(f"[CHECKPOINTS]: None saved yet ({e})")
        
    # 3. Check stdout log tail
    try:
        stdin, stdout, stderr = ssh.exec_command(f"tail -n 25 {REMOTE_DIR}/results/logs/experiment_stdout.log")
        log_lines = stdout.read().decode('ascii', errors='ignore').strip()
        print(f"\n[LATEST RUNNER LOG OUTPUT]:\n{log_lines}")
    except Exception as e:
        print(f"Error fetching log output: {e}")
        
    # 4. Sync manifest file back to local
    try:
        r_manifest = f"{REMOTE_DIR}/results/tables/rep_distance_manifest.json"
        l_manifest = os.path.join(LOCAL_WORKSPACE, "results/tables/rep_distance_manifest.json")
        sftp.get(r_manifest, l_manifest)
        print(f"\n[MANIFEST SYNC]: Updated local {l_manifest}")
        
        with open(l_manifest, "r") as f:
            manifest_data = json.load(f)
            
        completed_runs = sum(1 for m in manifest_data if m.get("Status") == "COMPLETED")
        running_runs = sum(1 for m in manifest_data if m.get("Status") == "RUNNING")
        pending_runs = sum(1 for m in manifest_data if m.get("Status") == "PENDING")
        print(f"Manifest Progress Summary: {completed_runs} Completed | {running_runs} Running | {pending_runs} Pending (Total: {len(manifest_data)})")
    except Exception as e:
        print(f"Manifest sync note: {e}")
        
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    check_remote_status()
