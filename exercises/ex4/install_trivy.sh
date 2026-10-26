#!/bin/bash
# Install Trivy on Ubuntu / Debian.
#
# Alternatively, via Homebrew (macOS / Linux):
#   brew install trivy
#
# Alternatively, run Trivy directly via Docker:
#   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image <image-name>

set -euo pipefail

sudo apt-get install wget apt-transport-https gnupg lsb-release -y
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/trivy.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update && sudo apt-get install trivy -y
