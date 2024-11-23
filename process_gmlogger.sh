#!/bin/bash

# Check if input file provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <gmlogger_zip_file>"
    exit 1
fi

# Check if file exists
if [ ! -f "$1" ]; then
    echo "Error: File $1 does not exist"
    exit 1
fi

# Check if required tools are installed
command -v dlt-convert >/dev/null 2>&1 || { echo "Error: dlt-convert is required but not installed"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed"; exit 1; }

# Run the Python script
python3 process_gmlogger.py "$1"