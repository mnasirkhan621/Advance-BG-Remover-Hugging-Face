# Deploying BiRefNet API to AWS EC2

This guide walks you through deploying the API to an Amazon EC2 instance.

## Prerequisites
- An AWS Account.
- AWS Key Pair file (`.pem`) for SSH access.

## 1. Launch EC2 Instance
1.  Go to **EC2 Dashboard** > **Launch Instances**.
2.  **Name**: `BiRefNet-API`
3.  **OS Image**: **Ubuntu Server 22.04 LTS** (or Amazon Linux 2023).
4.  **Instance Type**: 
    - **Recommended**: `t3.medium` (2 vCPU, 4GB RAM) or `t3.large`.
    - **Warning**: Do NOT use `t2.micro` (Free Tier), it only has 1GB RAM and will crash.
5.  **Key Pair**: Select your existing key pair or create a new one.
6.  **Network Settings**:
    - **Security Group**: Create a new one.
    - **Allow SSH traffic** from "My IP".
    - **Allow Custom TCP** port `8070` from "Anywhere" (`0.0.0.0/0`).
7.  **Storage**: Set to at least **15 GB** (PyTorch and Docker images are large).

## 2. Connect to Instance
Open your terminal (or PowerShell) and connect:
```bash
ssh -i "path/to/your-key.pem" ubuntu@<EC2-PUBLIC-IP>
```

## 3. Install Docker
Run these commands on the EC2 instance:
```bash
# Update and install Docker
sudo apt-get update
sudo apt-get install -y docker.io git

# Start Docker and enable it
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (so you don't need 'sudo' for docker commands)
sudo usermod -aG docker $USER
newgrp docker
```

## 4. Deploy Application
You can copy your files to the server or clone from a git repo.
Assuming you upload your files (app.py, requirements.txt, Dockerfile, weights folder, etc.):

### Option A: Upload via SCP (from your local machine)
```powershell
# Run this from your local folder containing the project
scp -i "path/to/your-key.pem" -r . ubuntu@<EC2-PUBLIC-IP>:~/birefnet-api
```

### Option B: Build and Run
On the EC2 instance:
```bash
cd ~/birefnet-api

# Build the Docker image
# This might take a few minutes (downloading PyTorch)
docker build -t birefnet-api .

# Run the container
# -d: Run in background
# -p 8070:8070: Map port 8070
docker run -d -p 8070:8070 --name api birefnet-api
```

## 5. Test
Open your browser or Postman and visit:
`http://<EC2-PUBLIC-IP>:8070/docs`

You should see the Swagger UI.

## Troubleshooting
- **API not reachable?** Check your Security Group inbound rules (Port 8070 must be open).
- **"Out of Memory" crash?** Check instance RAM. Use `docker logs api` to see errors.
