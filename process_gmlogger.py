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
    
    def check_cuda_library(lib_name):
        """Check for CUDA library in common locations"""
        try:
            import ctypes
            # Common library paths
            lib_paths = [
                lib_name,  # Direct name for LD_LIBRARY_PATH
                f"/usr/lib/{lib_name}",
                f"/usr/lib/x86_64-linux-gnu/{lib_name}",
                f"/usr/local/cuda/lib64/{lib_name}",
                f"/usr/local/cuda/targets/x86_64-linux/{lib_name}"
            ]
            
            for path in lib_paths:
                try:
                    ctypes.CDLL(path)
                    return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    try:
        # Initialize CUDA early
        torch.cuda.init()
        
        # Check CUDA environment variables
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        
        if torch.cuda.is_available():
            # Force device synchronization to catch any initialization issues
            torch.cuda.synchronize()
            cuda_device = torch.cuda.get_device_name(0)
            print(f"CUDA capable GPU detected: {cuda_device}")
            print(f"CUDA version: {torch.version.cuda}")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print("CUDA is not available - Using CPU only")
            return ['CPUExecutionProvider']
            
    except RuntimeError as e:
        print(f"CUDA initialization error: {e}")
        print("\nDiagnostic information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print("\nThis might be due to:")
        print("1. CUDA_VISIBLE_DEVICES being changed after Python started")
        print("2. Incompatible CUDA driver version")
        print("3. Missing or incorrect NVIDIA driver installation")
        print("\nFalling back to CPU only mode")
        return ['CPUExecutionProvider']
            
            # Check specific libraries
            cuda_libs = {
                "libcudnn_adv.so.9": "cuDNN Advanced",
                "libnvinfer.so.10": "TensorRT"
            }
            
            missing_libs = []
            for lib, name in cuda_libs.items():
                if not check_cuda_library(lib):
                    missing_libs.append(f"{name} ({lib})")
            
            if missing_libs:
                print("\nWarning: The following CUDA libraries are missing:")
                for lib in missing_libs:
                    print(f"  - {lib}")
                print("\nFalling back to CPU only mode")
                print("See installation instructions in process_gmlogger.sh")
            else:
                try:
                    # Test CUDA initialization
                    torch.cuda.init()
                    torch.cuda.set_device(0)
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    print("CUDA initialization successful - GPU will be used")
                except Exception as e:
                    print(f"\nWarning: CUDA initialization failed: {e}")
                    print("Falling back to CPU only mode")
        else:
            print("No CUDA capable GPU detected - Using CPU only")
    except Exception as e:
        print(f"\nWarning: GPU detection failed: {e}")
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

    # Configure ONNX Runtime session options
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # Reduce logging noise
    
    # Configure embedding function with CPU-safe settings
    try:
        # Initialize with explicit provider order
        ort_providers = []
        if 'CUDAExecutionProvider' in providers:
            ort_providers.extend(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            ort_providers.append('CPUExecutionProvider')
            
        # Set environment variable to suppress CUDA initialization warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Test the embedding function
        test_result = embedding_function(["test"])
        print("Embedding function initialized successfully using providers:", ort_providers)
        
    except Exception as e:
        print(f"\nError initializing embedding function: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure all required dependencies are installed:")
        print("   pip install -U onnxruntime torch transformers")
        print("2. For GPU support, install CUDA dependencies:")
        print("   - CUDA 12.x")
        print("   - cuDNN 9.x")
        print("   - TensorRT")
        print("3. Check process_gmlogger.sh for detailed installation instructions")
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
