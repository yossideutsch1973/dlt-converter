import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm
import textwrap

def setup_llm():
    """Initialize the Phi-3.5 model"""
    model_name = "microsoft/phi-2"
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=device == "cuda"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        return pipe
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages:")
        print("   pip install transformers torch accelerate")
        print("2. Logged in to HuggingFace:")
        print("   huggingface-cli login")
        print("3. Accepted Llama 2 terms on HuggingFace website")
        raise

def get_relevant_logs(query, collection, max_results=5):
    """Query ChromaDB for relevant log entries"""
    results = collection.query(
        query_texts=[query],
        n_results=max_results
    )
    
    # Combine all retrieved documents into one text
    if results and 'documents' in results and results['documents']:
        return "\n".join(results['documents'][0])
    return ""

def generate_prompt(logs, query_type="summarize"):
    """Generate appropriate prompt based on query type"""
    if query_type == "summarize":
        return f"""Below are log entries from a vehicle system. Please summarize the key events and information in 10 clear sentences.
Focus on important system states, errors, and significant events.

Log entries:
{logs}

Summary in 10 sentences:"""
    
    elif query_type == "analyze":
        return f"""Below are log entries from a vehicle system. Please analyze these logs and provide:
1. Key system states and transitions
2. Any errors or warnings
3. Notable patterns or sequences
4. Potential issues or anomalies
5. Important timing information

Log entries:
{logs}

Analysis:"""
    
    return f"""Please analyze these vehicle system logs:

{logs}

Analysis:"""

def query_logs(query_type="summarize", custom_query=None):
    """Main function to query and analyze logs using Llama"""
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("gmlogger_data")
        
        # Initialize Llama model
        print("Initializing Llama model...")
        llm = setup_llm()
        
        # Define search queries for different analysis types
        search_queries = {
            "summarize": "important system events errors warnings",
            "analyze": "error warning critical failure system state",
        }
        
        # Get relevant logs
        search_query = custom_query if custom_query else search_queries.get(query_type, search_queries["summarize"])
        print("Retrieving relevant logs...")
        logs = get_relevant_logs(search_query, collection)
        
        if not logs:
            print("No relevant logs found!")
            return
        
        # Generate appropriate prompt
        prompt = generate_prompt(logs, query_type)
        
        # Generate response
        print("Analyzing logs with Llama...")
        response = llm(prompt, max_new_tokens=512)[0]['generated_text']
        
        # Extract the response part after the prompt
        response = response[len(prompt):].strip()
        
        # Format and print response
        print("\nAnalysis Results:")
        print("=" * 80)
        for line in textwrap.wrap(response, width=80):
            print(line)
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze vehicle logs using Llama")
    parser.add_argument('--type', choices=['summarize', 'analyze'], 
                      default='summarize', help='Type of analysis to perform')
    parser.add_argument('--query', type=str, help='Custom search query for logs')
    
    args = parser.parse_args()
    query_logs(args.type, args.query)
