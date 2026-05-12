import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Setup Environment
load_dotenv()

def run_rag_agent():
    DB_PATH = "./vector_db"
    
    print("--- Initializing RAG Agent ---")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found. Please run Step 2 first.")
        return

    vector_db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings_model
    )

    print("Connecting to local Llama 3.2 via Ollama...")
    llm = Ollama(model="llama3.2")

    # This "locks" the AI into being a grounded assistant
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Always mention which document source you are using.

Context: {context}

Question: {question}

Helpful Answer:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Get top 3 chunks
        return_source_documents=True, # Critical for Citations
        chain_type_kwargs={"prompt": QA_PROMPT}
    )


    query = "What is the main summary of the documents I provided?"
    print(f"\n--- Query: {query} ---")
    
    response = qa_chain.invoke({"query": query})

    # 8. Display Results with Production-Style Citations
    print("\n[AI Response]:")
    print(response["result"])
    
    print("\n[Sources Used]:")
    for i, doc in enumerate(response["source_documents"]):
        source_name = doc.metadata.get('source', 'Unknown')
        print(f"{i+1}. {source_name}")

if __name__ == "__main__":
    run_rag_agent()