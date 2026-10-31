import os

HOST = os.environ.get("VISION_RL_SSH_HOST")
USERNAME = os.environ.get("VISION_RL_SSH_USERNAME")
PORT = int(os.environ.get("VISION_RL_SSH_PORT", "2222"))

def main():
    import paramiko

    if not HOST or not USERNAME:
        raise RuntimeError("Set VISION_RL_SSH_HOST and VISION_RL_SSH_USERNAME before connecting.")
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh.connect(HOST, port=PORT, username=USERNAME)

    cmd = "source /home/piyush/miniconda3/bin/activate && conda activate rl_env && cd /home/piyush/vision-rl && cat results/logs/scale_experiment.log"
    print("Reading scale_experiment.log...")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    print("LOG CONTENT:\n", stdout.read().decode('ascii', errors='ignore'))
    print("LOG ERR:\n", stderr.read().decode('ascii', errors='ignore'))

    cmd2 = "source /home/piyush/miniconda3/bin/activate && conda activate rl_env && cd /home/piyush/vision-rl && PYTHONPATH=. python3 -u baselines/run_scale_experiment.py"
    print("\nDirect execution output:")
    stdin, stdout, stderr = ssh.exec_command(cmd2, get_pty=True)
    for i in range(30):
        line = stdout.readline()
        if not line:
            break
        print(line.rstrip())

    ssh.close()

if __name__ == "__main__":
    main()
