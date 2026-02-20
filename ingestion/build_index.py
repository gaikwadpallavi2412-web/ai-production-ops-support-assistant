# Step 4 — Build Vector Index (FAISS)

# Objective: Create a persistent vector database from chunked documents so the Ops Support Bot can perform fast semantic retrieval.
# What This Step Does
#    Input: * Chunked LangChain Documents * OpenAI embedding model
#    Process: * Generate embeddings * Store vectors in FAISS * Persist index locally
#    Output: * Saved FAISS index on disk * Ready-to-query vector store


import os
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from ingestion.loaders import load_all_documents
from ingestion.chunking import chunk_documents
from ingestion.embeddings import get_embedding_model


# ------------------------------------------------
# Main index builder
# ------------------------------------------------

def build_vector_index() -> FAISS:
    """
    Full offline pipeline:

    load → chunk → embed → FAISS → persist
    """

    #print("🔹 Loading raw documents...")
    documents: List[Document] = load_all_documents()
    #print(f"   Loaded: {len(documents)} documents")

    #print("🔹 Chunking documents...")
    chunked_docs: List[Document] = chunk_documents(documents)
    #print(f"   Chunks created: {len(chunked_docs)}")

    #print("🔹 Initializing embedding model...")
    embedding_model = get_embedding_model()

    #print("🔹 Building FAISS index...")
    vectorstore = FAISS.from_documents(
        chunked_docs,
        embedding_model,
    )

    # --------------------------------------------
    # Persist index
    # --------------------------------------------

    save_path = "vector_store"
    os.makedirs(save_path, exist_ok=True)

    #print(f"🔹 Saving FAISS index to: {save_path}")
    vectorstore.save_local(save_path)

    #print("✅ Vector index build complete!")

    return vectorstore


# ------------------------------------------------
# Optional loader (used later in retrieval)
# ------------------------------------------------

def load_vector_index() -> FAISS:
    """
    Load existing FAISS index from disk.
    """
    embedding_model = get_embedding_model()

    return FAISS.load_local(
        "vector_store",
        embedding_model,
        allow_dangerous_deserialization=True,  # required by FAISS
    )


# ------------------------------------------------
# CLI entry
# ------------------------------------------------

if __name__ == "__main__":
    build_vector_index()
