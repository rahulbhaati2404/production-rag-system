#!/bin/bash

# 1. Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# 2. Activate the environment
# On Windows use: source .venv/Scripts/activate
source .venv/bin/activate

# 3. Upgrade pip to the latest version
pip install --upgrade pip

# 4. Install dependencies from our requirements file
echo "Installing production-grade dependencies..."
pip install -r requirements.txt

echo "Setup Complete! Your environment is ready."