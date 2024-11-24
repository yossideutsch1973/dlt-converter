#!/bin/bash

# Find gmlogger archive file
GMLOGGER_FILE=$(ls gmlogger*.zip gmlogger*.tar.gz 2>/dev/null | head -n 1)

if [ -z "$GMLOGGER_FILE" ]; then
    echo "Error: No gmlogger archive file (zip or tar.gz) found in current directory"
    exit 1
fi

# Check if file is a supported format
if [[ ! "$GMLOGGER_FILE" =~ \.(zip|tar\.gz)$ ]]; then
    echo "Error: File must be a .zip or .tar.gz archive"
    exit 1
fi

echo "Found archive file: $GMLOGGER_FILE"

# Check if required tools are installed
command -v dlt-convert >/dev/null 2>&1 || { echo "Error: dlt-convert is required but not installed"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed"; exit 1; }

# Check and install Python dependencies
echo "Checking Python dependencies..."
python3 -m pip install -U pip

# Uninstall existing packages to avoid conflicts
python3 -m pip uninstall -y torch torchvision torchaudio onnxruntime onnxruntime-gpu

# Install latest CUDA 12-compatible versions
python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install --no-cache-dir onnxruntime-gpu
python3 -m pip install -U transformers chromadb tqdm accelerate sentencepiece

# Unset CUDA_VISIBLE_DEVICES to ensure clean state
unset CUDA_VISIBLE_DEVICES

# Quick CUDA check
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    
    # Check CUDA installation
    if [ ! -d "/usr/local/cuda" ] && [ ! -d "/usr/cuda" ]; then
        echo "Warning: CUDA installation not found."
        echo "To enable GPU support, install CUDA toolkit from NVIDIA website."
    else
        echo "CUDA installation found."
        
        # Check for required libraries
        MISSING_LIBS=()
        
        # Check TensorRT with multiple possible paths (any version)
        if ! (ldconfig -p | grep -q "libnvinfer.so" || \
              [ -f "/usr/lib/libnvinfer.so" ] || \
              [ -f "/usr/lib/x86_64-linux-gnu/libnvinfer.so" ] || \
              [ -f "/usr/local/cuda/lib64/libnvinfer.so" ]); then
            MISSING_LIBS+=("tensorrt")
        fi
        
        # Check cuDNN with multiple possible paths (any version)
        if ! (ldconfig -p | grep -q "libcudnn.so" || \
              [ -f "/usr/lib/libcudnn.so" ] || \
              [ -f "/usr/lib/x86_64-linux-gnu/libcudnn.so" ] || \
              [ -f "/usr/local/cuda/lib64/libcudnn.so" ]); then
            MISSING_LIBS+=("cudnn")
        fi
        
        if [ ${#MISSING_LIBS[@]} -ne 0 ]; then
            echo "Warning: Some CUDA libraries are missing. The script will run in CPU-only mode."
            echo "To enable GPU support, install the following packages:"
            for lib in "${MISSING_LIBS[@]}"; do
                echo "  - $lib"
            done
            echo ""
            echo "For Ubuntu, you need to add NVIDIA repositories first:"
            echo "1. Add NVIDIA repository for cudnn:"
            echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin"
            echo "   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600"
            echo "   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub"
            echo "   sudo add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /\""
            echo ""
            echo "2. Add TensorRT repository:"
            echo "   sudo apt-get update"
            echo "   sudo apt-get install software-properties-common"
            echo "   sudo apt-get install ubuntu-keyring"
            echo "   wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb"
            echo "   sudo dpkg -i nvidia-machine-learning-repo-ubuntu2204_1.0.0-1_amd64.deb"
            echo ""
            echo "3. Then install the packages:"
            echo "   sudo apt update"
            echo "   sudo apt install libnvinfer8 libnvinfer-plugin8 tensorrt libcudnn8 libcudnn8-dev"
            echo ""
            echo "For other distributions, please consult your package manager"
            echo "or visit NVIDIA website for installation instructions."
        else
            echo "All required CUDA libraries found."
            
            # Test CUDA availability with Python
            echo "Testing CUDA with PyTorch..."
            if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA OK - Device: {torch.cuda.get_device_name(0)}')"; then
                echo "Warning: PyTorch cannot access CUDA. Will fall back to CPU mode."
                echo "Try running: nvidia-smi"
                echo "If that works but CUDA is still not available, your CUDA/PyTorch installation might need repair."
            fi
        fi
    fi
fi

# Set minimal CUDA environment
if nvidia-smi &> /dev/null; then
    # Check for required CUDA libraries
    if ! ldconfig -p | grep -q "libnvinfer.so" || ! ldconfig -p | grep -q "libcudnn.so"; then
        echo "Warning: Required CUDA libraries missing. Using CPU mode."
        export CUDA_VISIBLE_DEVICES=""
    else
        export CUDA_VISIBLE_DEVICES=0
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    fi
else
    export CUDA_VISIBLE_DEVICES=""
fi

# Run the Python script
python3 process_gmlogger.py "$GMLOGGER_FILE"
