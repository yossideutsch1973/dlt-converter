import os
import zipfile
import tarfile
import chromadb
import torch
import logging
import onnxruntime as ort
from pathlib import Path
import subprocess
import shutil
import tempfile
from tqdm import tqdm
from chromadb.utils import embedding_functions

def extract_nested_archives(archive_path, extract_path):
    """Recursively extract nested archives (zip and tar.gz) with progress bar"""
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f"Extracting {Path(archive_path).name}"):
                tar_ref.extract(member, extract_path)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc=f"Extracting {Path(archive_path).name}"):
                zip_ref.extract(member, extract_path)
    
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
    CHUNK_SIZE = 250  # Reduced chunk size for better stability
    BATCH_SIZE = 10   # Smaller batch size for better memory management
    MAX_RETRIES = 3   # Number of retries for failed batches
    
    # Setup environment and providers
    providers = ['CPUExecutionProvider']  # Start with CPU as default
    
    try:
        # Check CUDA availability without initializing
        if torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(0)
            print(f"CUDA capable GPU detected: {cuda_device}")
            
            # Test CUDA initialization
            torch.cuda.init()
            torch.cuda.set_device(0)
            
            # Only add CUDA provider if initialization successful
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("CUDA initialization successful - GPU will be used if possible")
        else:
            print("No CUDA capable GPU detected - Using CPU only")
            
    except Exception as e:
        print(f"Warning: GPU initialization failed: {e}")
        print("Continuing with CPU only mode")
    
    # Setup ChromaDB with new client format and optimized settings
    persist_dir = "./chroma_db"
    
    # Initialize ChromaDB client with new format and settings
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
    )

    # Configure embedding function with CPU-safe settings
    try:
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        # Test the embedding function
        test_result = embedding_function(["test"])
        print("Embedding function initialized successfully")
    except Exception as e:
        print(f"Error initializing embedding function: {e}")
        print("Please ensure all required dependencies are installed:")
        print("pip install -U onnxruntime torch transformers")
        raise
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("gmlogger_data")
    except Exception as e:
        print(f"Error deleting collection: {str(e)}")
        
    # Create collection with configured embedding function
    collection = client.create_collection(
        name="gmlogger_data",
        metadata={"description": "Processed DLT log data"},
        embedding_function=embedding_function
    )
    
    print("Loading files into ChromaDB...")
    for file_idx, file in enumerate(files):
        print(f"\nProcessing file {file_idx + 1}/{len(files)}: {Path(file).name}")
        
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
        print(f"Split into {len(chunks)} chunks")
        
        # Process chunks in batches with retry logic
        for batch_idx in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing batches"):
            batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]
            retry_count = 0
            
            while retry_count < MAX_RETRIES:
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
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count == MAX_RETRIES:
                        print(f"\nFailed to process batch {batch_idx//BATCH_SIZE + 1} after {MAX_RETRIES} attempts: {str(e)}")
                    else:
                        print(f"\nRetrying batch {batch_idx//BATCH_SIZE + 1} (attempt {retry_count + 1})")
    
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
