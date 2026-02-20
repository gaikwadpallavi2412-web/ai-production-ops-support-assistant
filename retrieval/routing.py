# retrieval/routing.py — Runbook-First Routing Brain

# Objective : Decide **where to search first** based on user intent and detected service.
#             This is the intelligence layer that makes your Ops bot behave like a real L2 assistant.

# Why Routing Matters
#           In enterprise support copilots, we **do not search everything blindly**.
#           We follow priority logic: Runbook → Alerts → Incidents → Tickets → Logs
#           Because: * runbooks are authoritative * alerts give live signals * incidents give history * tickets give human fixes * logs are noisy

# Routing Strategy (v1 Production Pattern)

### If intent == runbook_lookup
#           1. Search runbooks
#           2. If found → return
#           3. Else → fallback to alerts/incidents
### If intent == incident_analysis
#           Search incidents first.
### If intent == log_analysis
#           Search logs.
### Else (default)
#           Broad search with service filter.

# ============================================
# retrieval/routing.py
# ============================================

from typing import List, Optional
from langchain_core.documents import Document

from retrieval.retrievers import get_filtered_retriever
from retrieval.filters import build_metadata_filter

from langsmith import traceable

RUNBOOK_TOP_K: int = 3
ALERT_TOP_K: int = 3
INCIDENT_TOP_K: int = 3
# ------------------------------------------------
# Internal helper
# ------------------------------------------------

def _retrieve_with_filter(query: str,metadata_filter: dict,top_k: Optional[int] = None,) -> List[Document]:
    """
    Shared retrieval helper.
    """
    retriever = get_filtered_retriever(metadata_filter=metadata_filter,top_k=top_k,)
    return retriever.invoke(query)


# ------------------------------------------------
# Runbook-first routing
# ------------------------------------------------

def route_runbook_first(query: str,service: Optional[str],) -> List[Document]:
    """
    Primary production strategy.

    1. Try runbooks
    2. Fallback to alerts/incidents
    """

    # ---------- Step 1: runbooks ----------
    runbook_filter = build_metadata_filter(source_type="runbook",service=service,)

    docs = _retrieve_with_filter(query,runbook_filter,top_k=RUNBOOK_TOP_K,)

    if docs:
        return docs

    # ---------- Step 2: alerts fallback ----------
    alert_filter = build_metadata_filter(source_type="alert",service=service,)

    docs = _retrieve_with_filter(query,alert_filter,top_k=ALERT_TOP_K,)

    if docs:
        return docs

    # ---------- Step 3: incidents fallback ----------
    incident_filter = build_metadata_filter(source_type="incident",service=service,)

    docs = _retrieve_with_filter(query,incident_filter,top_k=INCIDENT_TOP_K,)

    return docs


# ------------------------------------------------
# Intent-aware router (used by RAG chain)
# ------------------------------------------------
@traceable(name="routing_decision")
def route_query(query: str,intent: str,service: Optional[str],) -> List[Document]:
    """
    Main routing entry point.
    """

    # ---- runbook questions ----
    if intent == "runbook_lookup":
        return route_runbook_first(query, service)

    # ---- incident focused ----
    if intent == "incident_analysis":
        flt = build_metadata_filter(source_type="incident",service=service,)
        return _retrieve_with_filter(query, flt)

    # ---- log focused ----
    if intent == "log_analysis":
        flt = build_metadata_filter(source_type="log",service=service,)
        return _retrieve_with_filter(query, flt)

    # ---- default broad search ----
    flt = build_metadata_filter(service=service)
    return _retrieve_with_filter(query, flt)


# ------------------------------------------------
# Quick test helper
# ------------------------------------------------

# def test_routing():
#     docs = route_query(query="database connection exhausted",intent="runbook_lookup",service="trading-db",)
# 
#     print(f"✅ Retrieved {len(docs)} docs")
# 
#     if docs:
#         print(docs[0].metadata)
# 