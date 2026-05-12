
Project Title: Production-Grade RAG.

The Problem: Solving hallucinations in technical document retrieval.

The Tech Stack: LangChain, ChromaDB, Ollama (Llama 3.2), Cross-Encoders, Ragas.

-------------------------------------------------------------
This system will be a domain-specific RAG pipeline (e.g., for technical docs or legal papers) that focuses on trust. Unlike a basic demo, this system will:

Hybrid Search: Combine semantic meaning with keyword matching.

Re-Ranking: Use a second model to verify the best context.

Self-Correction: Use an evaluation framework to ensure the answer is actually supported by the documents.
-------------------------------------------------------------------------------
Phase 0: Environment & Project Structure
Before we write logic, we must set up the workspace. We will use a modular folder structure to keep "Data Ingestion" separate from "Retrieval" and "Evaluation."

Folder & File Specification
Root Directory: production_rag_system/

data/: Folder to store your raw PDFs or Text files.

notebooks/: Where we will experiment with chunking and retrieval.

src/: Where we will move the final production Python scripts.

.env: To store your API keys safely.

requirements.txt: To manage our dependencies.

-------------------------------------------------------------------
Run This in Terminal : 
bash setup.sh

--------------------------------------------------------------------
Phase 1: Ingestion & Chunking
File Specification:

Location: notebooks/01_ingestion_and_chunking.ipynb

Purpose: Load documents, clean them, and split them into chunks with metadata.
----------------------------------------------------------
Phase 2: Vector Storage (The Brain)
File Specification:

Location: notebooks/02_vector_storage.ipynb

Purpose: Convert text chunks into vectors using an embedding model and save them into ChromaDB.
--------------------------------------------------------------------
Step 3: The Retrieval and Generation (RAG) Engine
File: 03_retrieval_agent.py

Purpose: This is the final piece of the core pipeline. It connects your Vector Database (from Step 2) to a Local LLM (via Ollama). When you ask a question, this script finds the right context and forces the LLM to answer using only that information.

-------------------------------------------------------------------------------
Phase 4: The Re-ranker (Precision Engineering)
File: 04_reranked_retrieval.py

Purpose: Improve accuracy by using a Cross-Encoder to re-sort results before the LLM sees them.
------------------------------------------------------------------------------------
Phase 5: Automated Evaluation (The Quality Gate)
File: 05_evaluation_ragas.py

Purpose: Quantify the quality of your RAG pipeline so you can catch regressions before deploying to GitHub.
-------------------------------------------------------------------------------------
File Specification:

Location: Dockerfile

Purpose: Packages your entire environment so it can be deployed to the cloud or another server.