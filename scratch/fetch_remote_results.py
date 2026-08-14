import os
import paramiko

HOST = "10.0.24.7"
PORT = 2222
USERNAME = "piyush"
PASSWORD = "2£K40d#N"
REMOTE_DIR = "/home/piyush/vision-rl"
LOCAL_WORKSPACE = "d:/gitfork/vision rl"

print("=== CHECKING COMPLETED SCALE EXPERIMENT RESULTS & SYNCING ===")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD)

# 1. Process Status
stdin, stdout, stderr = ssh.exec_command("ps -ef | grep run_scale_experiment | grep -v grep")
proc_lines = stdout.read().decode('ascii', errors='ignore').strip()
if proc_lines:
    print(f"[PROCESS STATUS]: RUNNING\n  {proc_lines}")
else:
    print("[PROCESS STATUS]: COMPLETED / FINISHED!")

sftp = ssh.open_sftp()

# 2. Saved Models
models = sftp.listdir(f"{REMOTE_DIR}/results/models")
scale_models = [m for m in models if "procgen" in m or "MultiStageNavigation" in m]
print(f"\n[COMPLETED SCALE MODELS ({len(scale_models)})]:")
for m in sorted(scale_models):
    print(" -", m)

# 3. Saved Tables
tables = sftp.listdir(f"{REMOTE_DIR}/results/tables")
print(f"\n[SAVED TABLES ({len(tables)})]:")
for t in sorted(tables):
    print(" -", t)

# 4. Log Tail
stdin, stdout, stderr = ssh.exec_command(f"tail -n 60 {REMOTE_DIR}/results/logs/scale_experiment.log")
log_tail = stdout.read().decode('ascii', errors='ignore').strip()
print("\n[SCALE EXPERIMENT LOG TAIL]:\n", log_tail)

# 5. Download all tables to local workspace
os.makedirs(os.path.join(LOCAL_WORKSPACE, "results/tables"), exist_ok=True)
for t in tables:
    r_path = f"{REMOTE_DIR}/results/tables/{t}"
    l_path = os.path.join(LOCAL_WORKSPACE, "results/tables", t)
    sftp.get(r_path, l_path)
    print(f"Downloaded table: {t} -> {l_path}")

sftp.close()
ssh.close()
print("\nFetch and sync complete!")
