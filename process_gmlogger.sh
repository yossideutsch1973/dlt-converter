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
python3 -m pip install -U onnxruntime torch transformers chromadb tqdm

# Check CUDA and related dependencies
echo "Checking CUDA dependencies..."

# Check for NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA driver not found. GPU support will not be available."
    echo "To enable GPU support, install NVIDIA drivers first."
elif ! nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA driver found but not working properly."
    echo "Please check your NVIDIA driver installation."
else
    echo "NVIDIA driver detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
    
    # Check CUDA installation
    if [ ! -d "/usr/local/cuda" ] && [ ! -d "/usr/cuda" ]; then
        echo "Warning: CUDA installation not found."
        echo "To enable GPU support, install CUDA toolkit from NVIDIA website."
    else
        echo "CUDA installation found."
        
        # Check for required libraries
        MISSING_LIBS=()
        
        # Check TensorRT
        if ! ldconfig -p | grep -q "libnvinfer.so.10"; then
            MISSING_LIBS+=("tensorrt")
        fi
        
        # Check cuDNN
        if ! ldconfig -p | grep -q "libcudnn_adv.so.9"; then
            MISSING_LIBS+=("libcudnn8")
        fi
        
        if [ ${#MISSING_LIBS[@]} -ne 0 ]; then
            echo "Warning: Some CUDA libraries are missing. The script will run in CPU-only mode."
            echo "To enable GPU support, install the following packages:"
            for lib in "${MISSING_LIBS[@]}"; do
                echo "  - $lib"
            done
            echo ""
            echo "For Ubuntu/Debian, you can install them using:"
            echo "sudo apt update"
            echo "sudo apt install ${MISSING_LIBS[*]}"
            echo ""
            echo "For other distributions, please consult your package manager"
            echo "or visit NVIDIA website for installation instructions."
        else
            echo "All required CUDA libraries found."
        fi
    fi
fi

# Run the Python script
python3 process_gmlogger.py "$GMLOGGER_FILE"
