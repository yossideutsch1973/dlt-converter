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

# Check for CUDA dependencies
echo "Checking CUDA dependencies..."
if ! ldconfig -p | grep -q "libcudnn_adv.so.9\|libnvinfer.so.10"; then
    echo "Warning: Some CUDA libraries are missing. The script will run in CPU-only mode."
    echo "To enable GPU support, install:"
    echo "  - libcudnn8"
    echo "  - tensorrt"
fi

# Run the Python script
python3 process_gmlogger.py "$GMLOGGER_FILE"
