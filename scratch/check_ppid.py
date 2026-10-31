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

    sftp = ssh.open_sftp()
    try:
        models = sftp.listdir("/home/piyush/vision-rl/results/models")
        print("SAVED MODELS IN RESULTS/MODELS:")
        for m in models:
            print(" -", m)
    except Exception as e:
        print("Models check:", e)
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    main()
