#!/bin/bash
# EMR Bootstrap Script: Install Python dependencies on all nodes
# This runs on every node (master + workers) before the Spark job starts

set -e

echo "Installing Python dependencies..."
sudo pip3 install numpy pandas tqdm

echo "Bootstrap complete."
