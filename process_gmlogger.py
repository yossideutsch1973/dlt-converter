import os
import zipfile
import chromadb
from pathlib import Path
import subprocess
import shutil
import tempfile

def extract_nested_zips(zip_path, extract_path):
    """Recursively extract nested zip files"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Look for more zip files in extracted content
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.zip'):
                nested_zip = os.path.join(root, file)
                nested_extract_path = os.path.join(root, Path(file).stem)
                extract_nested_zips(nested_zip, nested_extract_path)

def convert_dlt_file(dlt_file, output_file):
    """Convert DLT file to text format using dlt-convert"""
    try:
        subprocess.run(['dlt-convert', '-a', dlt_file, '-o', output_file], check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Error converting {dlt_file}")
        return False

def process_dlt_files(base_path):
    """Find and convert all DLT files"""
    converted_files = []
    for root, _, files in os.walk(base_path):
        if 'qnx' in root.lower():
            for file in files:
                if file.endswith('.dlt'):
                    dlt_file = os.path.join(root, file)
                    output_file = dlt_file + '.txt'
                    if convert_dlt_file(dlt_file, output_file):
                        converted_files.append(output_file)
    return converted_files

def load_into_chromadb(files):
    """Load converted files into ChromaDB"""
    client = chromadb.Client()
    collection = client.create_collection("gmlogger_data")
    
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            # Split content into chunks if needed
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": file, "chunk": i}],
                    ids=[f"{Path(file).stem}_chunk_{i}"]
                )

def main(gmlogger_zip):
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting {gmlogger_zip}...")
        extract_nested_zips(gmlogger_zip, temp_dir)
        
        print("Processing DLT files...")
        converted_files = process_dlt_files(temp_dir)
        
        if not converted_files:
            print("No DLT files found in QNX folder!")
            return
        
        print(f"Found {len(converted_files)} DLT files, loading into ChromaDB...")
        load_into_chromadb(converted_files)
        print("Processing complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process_gmlogger.py <gmlogger_zip_file>")
        sys.exit(1)
    main(sys.argv[1])
