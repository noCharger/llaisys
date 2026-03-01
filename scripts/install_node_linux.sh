#!/bin/bash

# Install Node.js and npm using nvm (Node Version Manager)
echo "Installing nvm..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install latest LTS version of Node.js
echo "Installing Node.js LTS..."
nvm install --lts

# Verify installation
echo "Node version:"
node -v
echo "NPM version:"
npm -v

echo "Installation complete. Please restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc) to use npm."
