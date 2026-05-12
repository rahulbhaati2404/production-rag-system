import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load environment variables
load_dotenv()

def run_ingestion():
    # Define paths
    DATA_DIR = "./data"
    PROCESSED_DIR = "./data/processed"
    
    # Ensure directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"--- Starting Ingestion from {DATA_DIR} ---")

    # 2. Load Documents
    # DirectoryLoader scans the folder for .txt files. 
    # In production, this can be scaled to S3 buckets or databases.
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: No files found in {DATA_DIR}. Please add a .txt file.")
        return

    loader = DirectoryLoader(DATA_DIR, glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} document(s).")

    # 3. Strategic Chunking
    # We use RecursiveCharacterTextSplitter to maintain semantic integrity.
    # chunk_size: 1000 characters. 
    # chunk_overlap: 100 characters to ensure context isn't cut off mid-sentence.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True, # Critical for tracking exactly where info came from
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 4. Persistence (The Production Way)
    # We save these chunks to a file so Step 2 can read them without re-processing.
    OUTPUT_FILE = os.path.join(PROCESSED_DIR, "chunks.pkl")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Success: Chunks saved to {OUTPUT_FILE}")
    
    # Simple verification of the first chunk
    if chunks:
        print("\nVerification (First Chunk Preview):")
        print(f"Content: {chunks[0].page_content[:150]}...")
        print(f"Metadata: {chunks[0].metadata}")

if __name__ == "__main__":
    run_ingestion()