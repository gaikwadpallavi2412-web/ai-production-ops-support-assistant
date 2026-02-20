# chains/structured_output.py — Enterprise Structured Response

# Objective : Convert free-text RAG answers into **structured, machine-readable output** suitable for dashboards, ticketing systems, and Ops automation.
# Structured Schema (v1)
#     Fields chosen for L2 usefulness:
#     * issue_summary
#     * impacted_service
#     * recommended_steps
#     * escalation_required
#     * confidence


# ============================================
# chains/structured_output.py
# ============================================

from functools import lru_cache
from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ------------------------------------------------
# Structured schema
# ------------------------------------------------

class SupportResponse(BaseModel):
    issue_summary: str = Field(description="Short summary of the detected issue")
    impacted_service: str = Field(description="Primary affected service")
    recommended_steps: List[str] = Field(description="Step-by-step remediation actions")
    escalation_required: bool = Field(description="Whether human escalation is required")
    confidence: str = Field(description="low | medium | high")
    reference_docs: List[str] = Field(description="List of runbook/incident/alert/ticket titles used for the answer")


# ------------------------------------------------
# Parser
# ------------------------------------------------

parser = PydanticOutputParser(pydantic_object=SupportResponse)


# ------------------------------------------------
# Prompt
# ------------------------------------------------

STRUCTURED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a senior production support assistant.
                    Convert the provided answer into structured JSON.
                    Rules:
                    - Be faithful to the answer
                    - Do NOT invent steps
                    - escalation_required = true if issue seems critical
                    - confidence must be: low, medium, or high
                    {format_instructions}""",),
        ("human","""Original Answer:{answer}""",),
    ]
).partial(format_instructions=parser.get_format_instructions())


# ------------------------------------------------
# LLM singleton
# ------------------------------------------------

@lru_cache(maxsize=1)
def get_structured_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )


# ------------------------------------------------
# Main transformer
# ------------------------------------------------

def to_structured_response(answer: str) -> SupportResponse:
    """
    Convert free-text RAG answer into structured output.
    """

    chain = STRUCTURED_PROMPT | get_structured_llm() | parser
    return chain.invoke({"answer": answer})


# ------------------------------------------------
# Quick test helper
# ------------------------------------------------

#def test_structured():
#    sample_answer = """
#    Database connections are exhausted.
#
#    Recommended steps:
#    1. Check active sessions
#    2. Kill long running queries
#    3. Restart connection pool
#    """
#
#    result = to_structured_response(sample_answer)
#
#    print("\n✅ Structured Output:\n")
#    print(result.model_dump())
