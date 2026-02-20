# Step 1 — LangChain-Based Document Ingestion
# We use LangChain native components:
#       DirectoryLoader → scans folders
#       TextLoader → loads file content
#       Document → unified format

# Metadata Strategy
#       Common metadata (all documents):
#           doc_id,source_type,service,environment,priority_order,ingestion_time

# Source-specific metadata:
#       Runbooks → severity, owner_team
#       Incidents → incident_id, region
#       Alerts → alert_id
#       Tickets → ticket_id
#       Logs → log_type

# Priority Order (Runbook-first design)
#       runbook  → 1
#       alert    → 2
#       incident → 3
#       ticket   → 4
#       log      → 5

# ============================================
# Step 1 — LangChain Document Loader Pipeline
# ============================================

import re,os
from datetime import datetime
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

# --------------------------------------------
# CONFIG
# --------------------------------------------

# Get project root (one level up from ingestion/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATHS = {
    "runbooks": os.path.join(BASE_DIR, "data", "runbooks"),
    "incidents": os.path.join(BASE_DIR, "data", "incidents"),
    "alerts": os.path.join(BASE_DIR, "data", "alerts"),
    "tickets": os.path.join(BASE_DIR, "data", "tickets"),
    "logs": os.path.join(BASE_DIR, "data", "logs"),
}

SOURCE_PRIORITY = {
    "runbooks": 1,
    "alerts": 2,
    "incidents": 3,
    "tickets": 4,
    "logs": 5,
}

ENVIRONMENT_DEFAULT = "prod"


# --------------------------------------------
# Helper: regex extraction
# --------------------------------------------

def extract_field(pattern: str, text: str, default: str = "unknown"):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default


# --------------------------------------------
# Metadata enrichment
# --------------------------------------------

def enrich_metadata(doc: Document, source_folder: str, doc_index: int) -> Document:
    content = doc.page_content
    source_type = source_folder.rstrip("s")

    metadata = {
        "doc_id": f"{source_type}_{doc_index:04d}",
        "source_type": source_type,
        "environment": ENVIRONMENT_DEFAULT,
        "priority_order": SOURCE_PRIORITY[source_folder],
        "ingestion_time": datetime.utcnow().isoformat(),
        "service": extract_field(r"Service:\s*(.*)", content),
    }

    # ---- source-specific enrichment ----

    if source_type == "runbook":
        metadata["severity"] = extract_field(r"Severity:\s*(.*)", content)
        metadata["owner_team"] = extract_field(r"Owner Team:\s*(.*)", content)

    elif source_type == "incident":
        metadata["incident_id"] = extract_field(r"Incident ID:\s*(.*)", content)
        metadata["region"] = extract_field(r"Region:\s*(.*)", content)

    elif source_type == "alert":
        metadata["alert_id"] = extract_field(r"Alert ID:\s*(.*)", content)
        metadata["severity"] = extract_field(r"Severity:\s*(.*)", content)

    elif source_type == "ticket":
        metadata["ticket_id"] = extract_field(r"Ticket ID:\s*(.*)", content)
        metadata["assigned_group"] = extract_field(
            r"Assigned Group:\s*(.*)", content
        )

    elif source_type == "log":
        metadata["log_type"] = "application"

    # merge with existing metadata (keeps file path etc.)
    doc.metadata.update(metadata)
    return doc


# --------------------------------------------
# Main ingestion pipeline
# --------------------------------------------

def load_all_documents() -> List[Document]:
    all_docs: List[Document] = []
    doc_counter = 0

    for source_folder, path in DATA_PATHS.items():

        loader = DirectoryLoader(
            path,
            glob="**/*",
            loader_cls=TextLoader,
            show_progress=True,
        )

        docs = loader.load()

        for d in docs:
            doc_counter += 1
            enriched_doc = enrich_metadata(d, source_folder, doc_counter)
            all_docs.append(enriched_doc)

    return all_docs


# --------------------------------------------
# RUN STEP 1
# --------------------------------------------

documents = load_all_documents()

#print(f"✅ Total documents loaded: {len(documents)}")
## preview
#if documents:
#    print("\n🔎 Sample metadata:")
#    print(documents[12].metadata)
#    print(documents[12].metadata["source"][documents[12].metadata["source"].rfind('\\')+1:])

