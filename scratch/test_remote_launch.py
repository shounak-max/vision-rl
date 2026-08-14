import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.0.24.7", port=2222, username="piyush", password="2£K40d#N")

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
