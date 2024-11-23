import os
import zipfile
import tarfile
import chromadb
import torch
import logging
from pathlib import Path
import subprocess
import shutil
import tempfile
from tqdm import tqdm
from chromadb.utils import embedding_functions

def extract_nested_archives(archive_path, extract_path):
    """Recursively extract nested archives (zip and tar.gz)"""
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # Look for more archives in extracted content
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith(('.zip', '.tar.gz')):
                nested_archive = os.path.join(root, file)
                nested_extract_path = os.path.join(root, Path(file).stem)
                extract_nested_archives(nested_archive, nested_extract_path)

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
    dlt_files = []
    
    # First collect all DLT files
    for root, _, files in os.walk(base_path):
        if 'qnx' in root.lower():
            dlt_files.extend([
                (os.path.join(root, file), file)
                for file in files if file.endswith('.dlt')
            ])
    
    if dlt_files:
        print(f"Found {len(dlt_files)} DLT files")
        for dlt_path, file in tqdm(dlt_files, desc="Converting DLT files"):
            output_file = dlt_path + '.txt'
            if convert_dlt_file(dlt_path, output_file):
                converted_files.append(output_file)
    return converted_files

def load_into_chromadb(files):
    """Load converted files into ChromaDB with batched processing"""
    CHUNK_SIZE = 500  # Smaller chunks for better processing
    BATCH_SIZE = 20   # Process chunks in batches
    
    # Setup ChromaDB with new client format
    persist_dir = "./chroma_db"
    
    # Setup CUDA detection
    cuda_available = False
    try:
        if torch.cuda.is_available():
            torch.cuda.init()
            device_count = torch.cuda.device_count()
            if device_count > 0:
                cuda_available = True
                device_name = torch.cuda.get_device_name(0)
                print(f"CUDA is available - using GPU acceleration")
                print(f"Found {device_count} CUDA device(s): {device_name}")
    except Exception as e:
        print(f"CUDA initialization error: {str(e)}")
        
    if not cuda_available:
        print("CUDA not available or initialization failed - falling back to CPU")

    # Initialize ChromaDB client with new format
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("gmlogger_data")
    except:
        pass
        
    collection = client.create_collection(name="gmlogger_data")
    
    print("Loading files into ChromaDB...")
    for file_idx, file in enumerate(files):
        try:
            # Try UTF-8 first
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1
            with open(file, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Split content into smaller chunks
        chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
        print(f"\nProcessing file {file_idx + 1}/{len(files)}: {Path(file).name}")
        print(f"Split into {len(chunks)} chunks")
        
        # Process chunks in batches
        for batch_idx in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing batches"):
            batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]
            try:
                collection.add(
                    documents=batch_chunks,
                    metadatas=[{
                        "source": file,
                        "chunk": i + batch_idx,
                        "total_chunks": len(chunks)
                    } for i in range(len(batch_chunks))],
                    ids=[f"{Path(file).stem}_chunk_{i + batch_idx}" 
                         for i in range(len(batch_chunks))]
                )
            except Exception as e:
                print(f"\nError processing batch {batch_idx//BATCH_SIZE + 1}: {str(e)}")
                continue
    
    print("\nFinished loading all files into ChromaDB")
    return collection

def main(gmlogger_archive):
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting {gmlogger_archive}...")
        extract_nested_archives(gmlogger_archive, temp_dir)
        
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
