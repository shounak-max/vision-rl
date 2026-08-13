import paramiko

HOST = "10.0.24.7"
PORT = 2222
USERNAME = "piyush"
PASSWORD = "2£K40d#N"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD)

cmd = "source /home/piyush/miniconda3/bin/activate && conda activate rl_env && cd /home/piyush/vision-rl && PYTHONPATH=. python3 baselines/predictive_augmentation_selection.py"
print("Running cmd...")
stdin, stdout, stderr = ssh.exec_command(cmd)

print("STDOUT:", stdout.read().decode())
print("STDERR:", stderr.read().decode())
ssh.close()
