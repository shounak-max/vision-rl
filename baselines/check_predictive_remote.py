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
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep predictive_augmentation_selection | grep -v grep")
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
        stdin, stdout, stderr = ssh.exec_command(f"tail -n 25 {REMOTE_DIR}/results/logs/predictive_stdout.log")
        log_lines = stdout.read().decode('ascii', errors='ignore').strip()
        print(f"\n[LATEST RUNNER LOG OUTPUT]:\n{log_lines}")
    except Exception as e:
        print(f"Error fetching log output: {e}")
        
    # 4. Sync manifest file back to local
    try:
        r_summary = f"{REMOTE_DIR}/results/tables/predictive_selection_summary.json"
        l_summary = os.path.join(LOCAL_WORKSPACE, "results/tables/predictive_selection_summary.json")
        sftp.get(r_summary, l_summary)
        print(f"\n[SUMMARY SYNC]: Updated local {l_summary}")
        
        with open(l_summary, "r") as f:
            summary_data = json.load(f)
            
        print(f"Summary Results: N_Pairs={summary_data.get('N_Pairs')} | Pooled_Rho={summary_data.get('Pooled_Spearman_rho')} | Speedup={summary_data.get('Speedup_Factor')}")
        
        r_csv = f"{REMOTE_DIR}/results/tables/predictive_selection.csv"
        l_csv = os.path.join(LOCAL_WORKSPACE, "results/tables/predictive_selection.csv")
        sftp.get(r_csv, l_csv)
        print(f"[CSV SYNC]: Updated local {l_csv}")
    except Exception as e:
        print(f"Sync note (probably not finished yet): {e}")
        
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    check_remote_status()
