# retrieval/filters.py — Metadata Filtering Layer

# Objective : Create reusable metadata filters that narrow the search space before semantic retrieval.
# This answers: **What subset of documents should we search?**

# Why Filters Matter (Ops Reality)
#   Without filters, vector search scans the entire corpus.
#   In production support systems we must narrow by: service,source type,severity,region,priority

# Output
#   Each function returns: Dict[str, Any]
#   Example:{"service": "trading-db"}
#   Used by:get_filtered_retriever(...)

# Success Criteria
#   You should be able to:filter by service, filter by source type, combine filters, support runbook-first logic

# ============================================
# retrieval/filters.py
# ============================================


from typing import Dict, Any, Optional


# ------------------------------------------------
# Core reusable filter builder (PRIMARY)
# ------------------------------------------------

def build_metadata_filter(
    source_type: Optional[str] = None,
    service: Optional[str] = None,
    max_priority: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generic metadata filter builder.

    This is the main reusable function used across retrieval.

    Example outputs:

        {"source_type": "runbook", "service": "trading-db"}

        {"priority_order": {"$lte": 2}}
    """

    flt: Dict[str, Any] = {}

    # source filter
    if source_type:
        flt["source_type"] = source_type

    # service filter
    if service and service != "unknown":
        flt["service"] = service

    # priority filter
    if max_priority is not None:
        flt["priority_order"] = {"$lte": max_priority}

    return flt


# ------------------------------------------------
# Thin convenience wrappers (OPTIONAL but clean)
# ------------------------------------------------

def build_runbook_filter(service: Optional[str]) -> Dict[str, Any]:
    return build_metadata_filter(source_type="runbook", service=service)


def build_incident_filter(service: Optional[str]) -> Dict[str, Any]:
    return build_metadata_filter(source_type="incident", service=service)


def build_alert_filter(service: Optional[str]) -> Dict[str, Any]:
    return build_metadata_filter(source_type="alert", service=service)


def build_ticket_filter(service: Optional[str]) -> Dict[str, Any]:
    return build_metadata_filter(source_type="ticket", service=service)


def build_log_filter(service: Optional[str]) -> Dict[str, Any]:
    return build_metadata_filter(source_type="log", service=service)


# ------------------------------------------------
# Broad search fallback
# ------------------------------------------------

def build_all_sources_filter(service: Optional[str]) -> Dict[str, Any]:
    return build_metadata_filter(service=service)


# ------------------------------------------------
# Quick test
# ------------------------------------------------

#def test_filters():
#    print(build_metadata_filter("runbook", "trading-db"))
#    print(build_runbook_filter("payment-gateway"))
