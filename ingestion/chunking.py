# Step 2 — Smart Chunking Strategy (LangChain)

# Objective :Split the enriched documents into semantically meaningful chunks optimized for retrieval quality, token efficiency, and Ops relevance.

# Strategy Overview

# | Source Type | Chunk Size | Overlap | Reason                  |
# | ----------- | ---------- | ------- | ----------------------- |
# | runbook     | 900        | 150     | Preserve procedures     |
# | alert       | 500        | 80      | Compact signal          |
# | incident    | 800        | 120     | Narrative context       |
# | ticket      | 700        | 120     | Human troubleshooting   |
# | log         | 300        | 50      | High-granularity events |


# ============================================
# Step 2 — Source-Aware Chunking (LangChain)
# ============================================
from ingestion import loaders
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ------------------------------------------------
# Splitter configurations per source
# ------------------------------------------------

SPLITTER_CONFIG = {
    "runbook": {"chunk_size": 900, "chunk_overlap": 150},
    "alert": {"chunk_size": 500, "chunk_overlap": 80},
    "incident": {"chunk_size": 800, "chunk_overlap": 120},
    "ticket": {"chunk_size": 700, "chunk_overlap": 120},
    "log": {"chunk_size": 300, "chunk_overlap": 50},
}

# cache splitters (performance best practice)
_splitter_cache = {}


def get_splitter(source_type: str) -> RecursiveCharacterTextSplitter:
    """Return cached splitter per source type."""
    if source_type not in _splitter_cache:
        cfg = SPLITTER_CONFIG.get(
            source_type,
            {"chunk_size": 700, "chunk_overlap": 100},
        )

        _splitter_cache[source_type] = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""],  # production-friendly order
        )

    return _splitter_cache[source_type]


# ------------------------------------------------
# Main chunking function
# ------------------------------------------------

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents using source-aware strategy."""
    chunked_docs: List[Document] = []

    for doc in documents:
        source_type = doc.metadata.get("source_type", "unknown")
        splitter = get_splitter(source_type)

        splits = splitter.split_documents([doc])
        chunked_docs.extend(splits)

    return chunked_docs


# ------------------------------------------------
# RUN STEP 2
# ------------------------------------------------

chunked_documents = chunk_documents(loaders.documents)

#print(f"✅ Original docs: {len(loaders.documents)}")
#print(f"✅ Chunked docs: {len(chunked_documents)}")

# preview one chunk
#if chunked_documents:
#    print("\n🔎 Sample chunk metadata:")
#    print(chunked_documents[0].metadata)
#    print("\n🔎 Sample chunk preview:")
#    print(chunked_documents[0].page_content[:300])

