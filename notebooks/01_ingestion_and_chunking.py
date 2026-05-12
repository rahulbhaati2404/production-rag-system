{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4541a571",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# %% [markdown]\n",
    "# # Step 1: Data Ingestion and Strategic Chunking\n",
    "# This notebook handles loading documents and splitting them into chunks.\n",
    "# In production, we use 'overlap' to ensure context isn't lost at the boundaries.\n",
    "\n",
    "# %%\n",
    "import os\n",
    "from dotenv import load_dotenv\n",
    "from langchain_community.document_loaders import TextLoader, DirectoryLoader\n",
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "\n",
    "# Load environment variables (for future API calls)\n",
    "load_dotenv()\n",
    "\n",
    "# %% [markdown]\n",
    "# ### 1. Load Documents\n",
    "# We use DirectoryLoader to scan the 'data' folder for any .txt files.\n",
    "# (In a real scenario, you could add PDF or Markdown loaders here).\n",
    "\n",
    "# Create data directory if it doesn't exist for testing\n",
    "os.makedirs('../data', exist_ok=True)\n",
    "\n",
    "# NOTE: Ensure you place at least one .txt file in the 'data' folder before running this!\n",
    "loader = DirectoryLoader('../data', glob=\"./*.txt\", loader_cls=TextLoader)\n",
    "documents = loader.load()\n",
    "\n",
    "print(f\"Loaded {len(documents)} documents.\")\n",
    "\n",
    "# %% [markdown]\n",
    "# ### 2. Strategic Chunking\n",
    "# We use RecursiveCharacterTextSplitter because it tries to split on \n",
    "# paragraphs and sentences first, keeping the text semantically coherent.\n",
    "\n",
    "# chunk_size: 1000 characters (~200-250 tokens)\n",
    "# chunk_overlap: 100 characters to keep context between chunks\n",
    "text_splitter = RecursiveCharacterTextSplitter(\n",
    "    chunk_size=1000,\n",
    "    chunk_overlap=100,\n",
    "    length_function=len,\n",
    "    add_start_index=True, # Critical for 'Citations' - tells us where in the doc the chunk is\n",
    ")\n",
    "\n",
    "chunks = text_splitter.split_documents(documents)\n",
    "\n",
    "# %%\n",
    "# Output the results to verify\n",
    "if chunks:\n",
    "    print(f\"Split documents into {len(chunks)} chunks.\")\n",
    "    print(\"\\n--- Example Chunk ---\")\n",
    "    print(chunks[0].page_content[:200] + \"...\")\n",
    "    print(\"\\n--- Metadata (for Citations) ---\")\n",
    "    print(chunks[0].metadata)\n",
    "else:\n",
    "    print(\"No chunks created. Check if your 'data' folder has .txt files.\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
