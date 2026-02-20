# evaluation/llm_judge.py — LLM-as-Judge Evaluation (Enterprise)

## 🎯 Objective

# Automatically grade answer quality using an LLM judge.
# This evaluates things traditional metrics miss:
#       * grounding correctness
#       * step usefulness
#       * hallucination risk
#       * completeness
# This is exactly how modern GenAI systems are evaluated.

# ============================================
# evaluation/llm_judge.py
# ============================================

from functools import lru_cache
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ------------------------------------------------
# Judge schema
# ------------------------------------------------

class JudgeResult(BaseModel):
    grounded: bool = Field(
        description="Whether answer is supported by provided context"
    )
    useful_steps: bool = Field(
        description="Whether remediation steps are actionable for L2"
    )
    hallucination: bool = Field(
        description="Whether answer contains invented information"
    )
    overall_score: int = Field(
        description="Overall quality score from 1 (poor) to 5 (excellent)"
    )


parser = PydanticOutputParser(pydantic_object=JudgeResult)


# ------------------------------------------------
# Prompt
# ------------------------------------------------

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a strict senior SRE evaluating an AI support assistant.
                     Evaluate the answer using the provided context.

                    Scoring rules:
                    - grounded = true only if answer is supported by context
                    - useful_steps = true only if steps are practical for L2 ops
                    - hallucination = true if any info is not in context
                    - overall_score from 1 to 5
                    Be strict and objective.
                    {format_instructions}""",),
        ("human","""User Query:{query}
                    Retrieved Context:{context}
                    Assistant Answer:{answer}""",),
    ]
).partial(format_instructions=parser.get_format_instructions())


# ------------------------------------------------
# LLM singleton
# ------------------------------------------------

@lru_cache(maxsize=1)
def get_judge_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )


# ------------------------------------------------
# Public API
# ------------------------------------------------

def judge_answer(query: str, context: str, answer: str) -> JudgeResult:
    """
    Evaluate answer quality using LLM judge.
    """

    chain = JUDGE_PROMPT | get_judge_llm() | parser

    return chain.invoke(
        {
            "query": query,
            "context": context,
            "answer": answer,
        }
    )