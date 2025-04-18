#!/bin/bash

# run podman for containers
if ! command -v podman &> /dev/null; then
    echo "Podman is not installed, installing..."
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is not installed. Please install it first."
        echo "Visit https://brew.sh for installation instructions."
        exit 1
    fi
    brew install podman
    echo "Podman installed"
    podman machine init > /dev/null 2>&1
    echo "Podman initialized"
fi
if ! podman machine list | grep -q "Currently running"; then
    podman machine set --user-mode-networking
    nohup podman machine start > /dev/null 2>&1
    echo "Podman machine started"
    podman info > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to initialize Podman. Please check the error messages above."
        exit 1
    fi
fi
export DOCKER_HOST=http+unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')

# Check if OPENAI_API_KEY is set
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

if [ -z "${SERPER_API_KEY}" ]; then
    echo "Error: SERPER_API_KEY environment variable is not set"
    exit 1
fi

# Run the Python script
echo "You should be able to run the GAIA agent now using this command:"
echo python "$(dirname "$0")/run_gaia.py"