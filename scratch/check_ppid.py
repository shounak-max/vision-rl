import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.0.24.7", port=2222, username="piyush", password="2£K40d#N")

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
