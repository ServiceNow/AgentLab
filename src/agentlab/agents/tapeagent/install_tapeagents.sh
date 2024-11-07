#!/bin/bash

if [ ! -d "TapeAgents" ]; then
    # Clone the repository
    git clone https://github.com/ServiceNow/TapeAgents.git
    # Install the package in editable mode
    pip install -e TapeAgents
else
    echo "TapeAgents directory already exists. Skipping installation."
fi
