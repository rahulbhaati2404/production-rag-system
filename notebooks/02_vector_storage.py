import os
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def run_vector_storage():
    # Define paths
    CHUNKS_FILE = "./data/processed/chunks.pkl"
    DB_PATH = "./vector_db"

    print("--- Starting Vector Storage Process ---")

    if not os.path.exists(CHUNKS_FILE):
        print(f"Error: {CHUNKS_FILE} not found. Please run Step 1 first.")
        return

    with open(CHUNKS_FILE, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"Loaded {len(chunks)} chunks from disk.")


    print("Initializing HuggingFace Embeddings (this may take a moment)...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Change to 'cuda' if you have a GPU
    )

    print(f"Creating vector database at {DB_PATH}...")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=DB_PATH
    )

    print(f"Success: Vector database created and persisted at {DB_PATH}")

    print("\n--- Testing Vector Search (Non-LLM) ---")
    test_query = "What information is available?" 
    docs = vector_db.similarity_search(test_query, k=2)
    
    for i, doc in enumerate(docs):
        print(f"Result {i+1} Source: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:100]}...\n")

if __name__ == "__main__":
    run_vector_storage()