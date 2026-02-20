# chains/rag_chain.py — Final RAG Answer Generation

# Objective : Generate the grounded Ops answer using:* user query * routed documents * LLM reasoning

# Responsibilities
#   This module:
#   * calls intent classifier
#   * calls router
#   * builds context
#   * invokes LLM
#   * returns grounded answer
#   * enforces “use context only” rule

# ============================================
# chains/rag_chain.py
# ============================================

from typing import List
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langsmith import traceable

from chains.intent_chain import classify_intent
from retrieval.routing import route_query

import os
MAX_CONTEXT_DOCS: int = 6
# ------------------------------------------------
# Prompt
# ------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a senior L2 production support assistant.
                     IMPORTANT RULES:
                     - Answer ONLY using the provided context
                     - If runbook steps exist → provide step-by-step actions
                     - If the context does not contain relevant information, say:"No relevant information found in runbooks, incidents, alerts, logs, or tickets."
                     - Be concise and actionable
                     - Use bullet points for steps
                     - If question out of scope like "what is date today?","what is whether today?"-> say "I am an L2 support assistant. Please ask questions related to production support issues.""",),
        ("human","""Conversation History:{history}
                    User Question:{query}
                    Context:{context}
                    Provide the best support response.""",),
    ]
)


# ------------------------------------------------
# LLM singleton
# ------------------------------------------------

@lru_cache(maxsize=1)
def get_rag_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )


def _extract_reference_ids(docs: List[Document]) -> List[str]:
    """
    Extract source filenames from retrieved documents.
    Hardened against null/empty metadata.
    """

    r_doc = set()

    for d in docs:
        md = d.metadata or {}

        doc_src = md.get("source")

        # 🔥 HARD GUARDS
        if not doc_src:
            continue
        if str(doc_src).lower() in {"none", "null", ""}:
            continue

        filename = os.path.basename(str(doc_src).strip())

        if filename and filename.lower() not in {"none", "null"}:
            r_doc.add(filename)

    return sorted(r_doc)
# ------------------------------------------------
# Context builder
# ------------------------------------------------

def _build_context(docs: List[Document]) -> str:
    """
    Convert retrieved docs into prompt context.
    """

    if not docs:
        return "No relevant documents found."

    context_parts = []

    for d in docs[: MAX_CONTEXT_DOCS]:
        source = d.metadata.get("source_type", "unknown")
        service = d.metadata.get("service", "unknown")

        block = f"""
                    Source: {source}
                    Service: {service}
                    Content:
                    {d.page_content}
                """
        context_parts.append(block.strip())

    return "\n\n---\n\n".join(context_parts)


# ------------------------------------------------
# Main RAG pipeline
# ------------------------------------------------
@traceable(name="rag_generate_answer")
def generate_answer(query: str,history: str = "",service: str | None = None,) -> tuple[str, list[str]]:
    """
    End-to-end RAG pipeline.
    """

    # ---- Step 1: classify intent ----
    intent = classify_intent(query)

    # ---- Step 2: retrieve docs via router ----
    docs = route_query(
        query=query,
        intent=intent,
        service=service,
    )

    reference_ids = _extract_reference_ids(docs)
    
    if not docs:
        return (
        "No relevant information found in runbooks, incidents, alerts, "
        "logs, or tickets. Please verify the issue or update the knowledge base.",
        [],
    )

    # ---- Step 3: build context ----
    context = _build_context(docs)

    # ---- Step 4: generate answer ----
    chain = RAG_PROMPT | get_rag_llm()
    response = chain.invoke(
        {
            "query": query,
            "context": context,
            "history": history,
        }
    )

    
    return response.content, reference_ids




# ------------------------------------------------
# Quick test helper
# ------------------------------------------------

#def test_rag():
#    query = "What should I do if trading DB connections are exhausted?"
#
#    answer = generate_answer(
#        query=query,
#        service="trading-db",
#    )
#
#    print("\n✅ RAG Response:\n")
#    print(answer)
