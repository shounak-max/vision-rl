import os
HOST = os.environ.get("VISION_RL_SSH_HOST")
PORT = 2222
USERNAME = os.environ.get("VISION_RL_SSH_USERNAME")

if __name__ == "__main__":
    import paramiko

    if not HOST or not USERNAME:
        raise RuntimeError("Set VISION_RL_SSH_HOST and VISION_RL_SSH_USERNAME before connecting.")
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh.connect(HOST, port=PORT, username=USERNAME)

    cmd = "source /home/piyush/miniconda3/bin/activate && conda activate rl_env && cd /home/piyush/vision-rl && PYTHONPATH=. python3 baselines/predictive_augmentation_selection.py"
    print("Running cmd...")
    stdin, stdout, stderr = ssh.exec_command(cmd)

    print("STDOUT:", stdout.read().decode())
    print("STDERR:", stderr.read().decode())
    ssh.close()
