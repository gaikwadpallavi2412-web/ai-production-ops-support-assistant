# chains/intent_chain.py — Intent Classification Layer

# Objective : Classify the user’s question into an Ops intent so the router knows where to search.
#             This is the first place your LLM adds intelligence.

# Supported Intents : * runbook_lookup * incident_analysis * log_analysis * general_question

# ============================================
# chains/intent_chain.py
# ============================================

from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable


# ------------------------------------------------
# Prompt
# ------------------------------------------------

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system","""You are an intent classifier for an L2 production support assistant.

                    Your task is to classify the user's query into EXACTLY ONE of these intents:
                    runbook_lookup → user is asking for troubleshooting steps, SOP, or what to do
                    incident_analysis → user is asking about a failure, root cause, or incident investigation
                    log_analysis → user is asking to analyze logs, errors, stack traces, or log patterns
                    alert_investigation → user is asking about monitoring alerts, spikes, or threshold breaches
                    ticket_investigation → user is asking about support tickets, user-reported issues, or SRs
                    general_question → question NOT related to L2 production support

                    INTENT DEFINITIONS

                    runbook_lookup:
                    User wants remediation steps, SOP, checklist, or how to fix an issue.

                    incident_analysis:
                    User asks what happened, why it failed, or requests incident/root cause analysis.

                    log_analysis:
                    User provides logs or asks to interpret errors, stack traces, or log messages.

                    alert_investigation:
                    User mentions alerts, monitoring warnings, spikes, or threshold breaches.
                    Typical terms:
                    alert, warning, critical alert, spike, high usage alert, threshold breach

                    ticket_investigation:
                    User refers to support tickets, SRs, user-reported problems, or service requests.
                    Typical terms:
                    ticket, SR, service request, user reported, customer issue, case number

                    general_question:
                    Anything unrelated to production support, such as:
                    weather, time/date, greetings, personal questions, general knowledge.

                    IMPORTANT BIAS RULE

                    If the query contains production/infra terms like:

                    disk, cpu, memory, mq, queue, database, db, latency,
                    timeout, kubernetes, pod, service down, error, failure,
                    trade, payment, settlement, risk engine, eod, batch,
                    alert, ticket, SR

                    → it is VERY LIKELY an L2 support query.

                    When unsure between general_question and any ops intent,
                    prefer the appropriate ops intent.

                    FEW-SHOT EXAMPLES

                    User: eod reconciliation failure steps
                    Intent: runbook_lookup

                    User: why did swift processing fail yesterday
                    Intent: incident_analysis

                    User: analyze this error from trade service logs
                    Intent: log_analysis

                    User: critical alert trade settlement queue depth high
                    Intent: alert_investigation

                    User: SR-Trade-003 users reporting payment timeouts
                    Intent: ticket_investigation

                    User: what is the date today
                    Intent: general_question

                    OUTPUT FORMAT (STRICT)

                    Return ONLY the intent label:

                    runbook_lookup
                    or
                    incident_analysis
                    or
                    log_analysis
                    or
                    alert_investigation
                    or
                    ticket_investigation
                    or
                    general_question"""),
        ("human", "{query}"),
    ]
)


# ------------------------------------------------
# LLM singleton
# ------------------------------------------------

@lru_cache(maxsize=1)
def get_intent_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini",temperature=0,)


# ------------------------------------------------
# Intent classifier
# ------------------------------------------------
from typing import Literal

@traceable(name="intent_classification")
def classify_intent(query: str) -> Literal[
    "runbook_lookup",
    "incident_analysis",
    "log_analysis",
    "alert_investigation",
    "ticket_investigation",
    "general_question",
]:
    """
    Classify user query into Ops intent.
    """

    chain = INTENT_PROMPT | get_intent_llm() | StrOutputParser()
    intent = chain.invoke({"query": query}).strip()

    # 🔒 Safety normalization (VERY IMPORTANT in prod)
    intent = intent.lower()

    # 🔒 Guardrail against LLM drift
    allowed = {
        "runbook_lookup",
        "incident_analysis",
        "log_analysis",
        "alert_investigation",
        "ticket_investigation",
        "general_question",
    }

    if intent not in allowed:
        # fallback bias toward ops (safer than blocking)
        intent = "runbook_lookup"

    return intent


# ------------------------------------------------
# Quick test helper
# ------------------------------------------------

#def test_intent():
#    test_queries = [
#        "What is the first step if disk usage spikes?",
#        "Why did payment fail yesterday?",
#        "Analyze these error logs",
#        "What is trade settlement?",
#    ]
#
#    for q in test_queries:
#        print(f"\nQuery: {q}")
#        print("Intent:", classify_intent(q))
#