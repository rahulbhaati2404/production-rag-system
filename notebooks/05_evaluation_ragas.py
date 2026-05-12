import os
import pickle
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Setup Environment
load_dotenv()

def run_evaluation():
    DB_PATH = "./vector_db"
    
    judge_llm = Ollama(model="llama3.2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    test_questions = [
        "What is the main topic of the uploaded documents?",
        "Are there any specific instructions mentioned?",
    ]
    
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": []
    }

    print(f"--- Running RAG on {len(test_questions)} test questions ---")

    for query in test_questions:
        # Retrieve context
        docs = retriever.invoke(query)
        context_strings = [doc.page_content for doc in docs]
        
        # Generate answer
        # Note: We use a simple prompt for the evaluation test
        context_joined = "\n".join(context_strings)
        response = judge_llm.invoke(f"Context: {context_joined}\nQuestion: {query}\nAnswer:")
        
        data_samples["question"].append(query)
        data_samples["answer"].append(response)
        data_samples["contexts"].append(context_strings)

    # 6. Convert to Dataset and Evaluate
    dataset = Dataset.from_dict(data_samples)
    
    print("\n--- Calculating RAGAS Metrics (Faithfulness & Relevance) ---")
    # This step uses the LLM to judge the results
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevance],
        llm=judge_llm,
        embeddings=embeddings
    )

    # 7. Final Report
    print("\n[Production Quality Report]")
    print("-" * 30)
    df = score.to_pandas()
    print(df[['question', 'faithfulness', 'answer_relevance']])
    print("-" * 30)
    print(f"Average Faithfulness: {df['faithfulness'].mean():.2f}")

if __name__ == "__main__":
    run_evaluation()