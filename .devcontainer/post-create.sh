#!/usr/bin/env bash
set -e

# Update apt
sudo apt-get update

# Upgrade pip
pip install --upgrade pip

# =============================================================================
# Poetry
# =============================================================================
# Install Poetry as the current user (if not already installed)
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Ensure Poetry is in PATH for this script
export PATH="$HOME/.local/bin:$PATH"

# Ensure Poetry will be in PATH for future terminal sessions
if ! grep -q "export PATH=\"$HOME/.local/bin:\$PATH\"" ~/.bashrc; then
    echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
fi

# Install the poetry-plugin-export plugin
# The plugin is no longer installed by default with Poetry 2.0.
poetry self add poetry-plugin-export

# =============================================================================
# Install dependencies
# =============================================================================
# Install all dependencies, including dev dependencies (synced to the lock file)
poetry sync --all-groups

# Install pre-commit hooks using Poetry's environment
poetry run pre-commit install
