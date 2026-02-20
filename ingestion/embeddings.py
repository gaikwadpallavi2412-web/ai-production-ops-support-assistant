# Step 3 — Embeddings Generation (OpenAI)
# Objective:Convert chunked documents into vector embeddings using OpenAI so they can be stored in a vector database and retrieved semantically.

# ============================================
# ingestion/embeddings.py
# ============================================

from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------
# Embedding model singleton
# ------------------------------------------------

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Returns configured OpenAI embedding model.
    Singleton pattern recommended in production.
    """
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ------------------------------------------------
# Optional: sanity check helper
# ------------------------------------------------

def test_embedding():
    """Quick check that embeddings work."""
    model = get_embedding_model()
    vector = model.embed_query("trade booking failure")
    #print(f"✅ Embedding vector size: {len(vector)}")


# ------------------------------------------------
# (Used later by vector store)
# ------------------------------------------------

def embed_documents(
    docs: List[Document],
    embedding_model: OpenAIEmbeddings,
):
    """
    Placeholder helper for future vector store step.
    Vector DBs usually call embeddings internally,
    but we keep this for flexibility.
    """
    return embedding_model.embed_documents(
        [d.page_content for d in docs]
    )


