import paramiko

def execute_nvidia_smi(hostname):
    try:
        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the host without username/password (default key-based authentication)
        ssh_client.connect(hostname)

        # Execute nvidia-smi command
        stdin, stdout, stderr = ssh_client.exec_command("nvidia-smi")

        # Retrieve the output
        output = stdout.read().decode()
        error = stderr.read().decode()

        if error:
            print(f"Error on {hostname}: {error}")
        else:
            print(f"Output from {hostname}:")
            print(output)

        # Close the SSH connection
        ssh_client.close()

    except Exception as e:
        print(f"Failed to connect to {hostname}: {str(e)}")

# List of hosts to check
gpu_hosts = [f"gpu{str(i).zfill(3)}" for i in range(1, 17)]

# Execute nvidia-smi on each host
for host in gpu_hosts:
    execute_nvidia_smi(host)
