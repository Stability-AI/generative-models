#!/bin/bash
set -euo pipefail
if nvcc --version 2&> /dev/null; then
    # Determine CUDA version using default nvcc binary
    CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p');
elif /usr/local/cuda/bin/nvcc --version 2&> /dev/null; then
    # Determine CUDA version using /usr/local/cuda/bin/nvcc binary
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p');
elif [ -f "/usr/local/cuda/version.txt" ]; then
    # Determine CUDA version using /usr/local/cuda/version.txt file
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.* \([0-9]\+\.[0-9]\+\).*/\1/')
else
    CUDA_VERSION=""
fi
echo $CUDA_VERSION
