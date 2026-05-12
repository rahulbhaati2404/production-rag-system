import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from sentence_transformers import CrossEncoder

load_dotenv()

def run_reranked_rag():
    DB_PATH = "./vector_db"
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings_model)

    print("Loading Cross-Encoder Re-ranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    llm = Ollama(model="llama3.2")

    query = "What specific instructions are mentioned in the docs?"
    print(f"\n--- Process Start for Query: {query} ---")


    initial_docs = vector_db.similarity_search(query, k=10)
    print(f"Initially retrieved {len(initial_docs)} chunks.")

    # Prepare pairs: [[query, doc1], [query, doc2], ...]
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)

    # Combine docs with their scores and sort by highest score
    reranked_results = sorted(list(zip(initial_docs, scores)), key=lambda x: x[1], reverse=True)
    
    top_3_docs = [res[0] for res in reranked_results[:3]]
    print("Re-ranking complete. Top chunks selected.")

    # Combine the top 3 chunks into one context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc in top_3_docs])
    
    final_prompt = f"""Answer the question based ONLY on the context below.
    
    Context: {context_text}
    
    Question: {query}
    
    Answer:"""

    print("\n[AI Response with Re-ranking]:")
    response = llm.invoke(final_prompt)
    print(response)

if __name__ == "__main__":
    run_reranked_rag()