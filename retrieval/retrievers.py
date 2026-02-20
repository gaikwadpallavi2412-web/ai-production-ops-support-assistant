# retrieval/retrievers.py — Production RAG Retriever

# Objective:; Create the semantic retrieval layer that queries the FAISS vector store and returns the most relevant Ops documents.
# This file answers: **How do we search the knowledge base?**
# Role in the Architecture

# Flow reminder:
# User Query
#    ↓
# routing.py      (WHERE to search)
#    ↓
# filters.py      (WHAT subset)
#    ↓
# retrievers.py   (HOW to search)  ← YOU ARE HERE

# Responsibilities
# This module:
# * loads FAISS index
# * creates base retriever
# * supports metadata filtering
# * tunes top_k
# * prepares for runbook-first logic
# * keeps retrieval configurable

# Output
# Primary function returns a LangChain retriever:BaseRetriever
# Used later by the RAG chain.

# ============================================
# retrieval/retrievers.py
# ============================================

from functools import lru_cache
from typing import Optional, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from ingestion.build_index import load_vector_index

RETRIEVER_TOP_K: int = 5

# ------------------------------------------------
# Vector store singleton (VERY IMPORTANT)
# ------------------------------------------------

@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """
    Load FAISS index once and cache it.

    Why:
    - FAISS load is expensive
    - production best practice
    - avoids repeated disk reads
    """
    print("🔹 Loading FAISS vector store...")
    return load_vector_index()


# ------------------------------------------------
# Base retriever creator
# ------------------------------------------------

def get_base_retriever(
    top_k: Optional[int] = None,
) -> BaseRetriever:
    """
    Returns a standard similarity retriever.

    Args:
        top_k: number of documents to retrieve
    """
    vectorstore = get_vectorstore()

    k = top_k or RETRIEVER_TOP_K

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.75,  # ⭐ tune this
            "k": k,
        },
    )

    return retriever


# ------------------------------------------------
# Filtered retriever (used later by routing)
# ------------------------------------------------

def get_filtered_retriever(
    metadata_filter: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> BaseRetriever:
    """
    Returns retriever with metadata filtering.

    Example filter:
        {"service": "trading-db"}
    """

    vectorstore = get_vectorstore()
    k = top_k or RETRIEVER_TOP_K

    search_kwargs: Dict[str, Any] = {"k": k}

    # FAISS supports metadata filtering via kwargs
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    return retriever


# ------------------------------------------------
# Quick test helper (optional)
# ------------------------------------------------

def test_retrieval(query: str = "trade booking failure"):
    """
    Simple sanity test for retrieval.
    """
    retriever = get_base_retriever(top_k=3)
    docs = retriever.invoke(query)

   #print(f"✅ Retrieved {len(docs)} documents")

   #if docs:
   #    print("\n🔎 Top result metadata:")
   #    print(docs[0].metadata)
   #    print("\n🔎 Preview:")
   #    print(docs[0].page_content[:300])

