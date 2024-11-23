#!/bin/bash

# Check if input file provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <gmlogger_archive_file>"
    exit 1
fi

# Check if file exists
if [ ! -f "$1" ]; then
    echo "Error: File $1 does not exist"
    exit 1
fi

# Check if file is a supported format
if [[ ! "$1" =~ \.(zip|tar\.gz)$ ]]; then
    echo "Error: File must be a .zip or .tar.gz archive"
    exit 1
fi

# Check if required tools are installed
command -v dlt-convert >/dev/null 2>&1 || { echo "Error: dlt-convert is required but not installed"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed"; exit 1; }

# Check and install Python dependencies
echo "Checking Python dependencies..."
python3 -m pip install -U pip
python3 -m pip install -U onnxruntime torch transformers chromadb tqdm

# Run the Python script
python3 process_gmlogger.py "$1"
