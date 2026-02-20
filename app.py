# app.py — Production Entry Point

# Objective
#       Provide a clean runnable interface for your Ops Support Assistant.
#       
#       This is the layer that:
#       
#       * accepts user queries
#       * calls the full RAG pipeline
#       * returns structured output
#       * demonstrates end-to-end functionality

# ============================================
# app.py — Ops Support Assistant CLI
# ============================================

import sys
from typing import Optional

from chains.rag_chain import generate_answer
from chains.structured_output import to_structured_response
from chains.intent_chain import classify_intent

import os

print("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))
print("Key exists:", bool(os.getenv("LANGCHAIN_API_KEY")))

chat_history = []
MAX_HISTORY_TURNS = 4
# ------------------------------------------------
# Pretty printer
# ------------------------------------------------

def print_structured_response(resp):
    print("\n" + "=" * 60)
    print("🛠️  OPS SUPPORT RESPONSE")
    print("=" * 60)

    print(f"\n📌 Issue Summary:\n{resp.issue_summary}")
    print(f"\n🔧 Impacted Service: {resp.impacted_service}")

    print("\n📋 Recommended Steps:")
    for i, step in enumerate(resp.recommended_steps, 1):
        print(f"  {i}. {step}")

    print(f"\n🚨 Escalation Required: {resp.escalation_required}")
    print(f"📊 Confidence: {resp.confidence}")
    if getattr(resp, "reference_docs", None):
        print("\n📚 Reference IDs:")
        for rid in resp.reference_docs:
            print(f"  - {rid}")

    print("=" * 60 + "\n")

# Add helper to format history
def format_history(history):
    if not history:
        return "No prior conversation."

    lines = []
    for h in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"User: {h['user']}")
        lines.append(f"Assistant: {h['assistant']}")
    return "\n".join(lines)

# ------------------------------------------------
# Main loop
# ------------------------------------------------

def main():
    
    print("\n🚀 AI Production Support Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query: Optional[str] = input("👤 Your question: ").strip()

            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                print("\n👋 Exiting assistant.")
                break

            # ---- Step 0: intent check ----
            intent = classify_intent(query)

            # 🚫 OUT-OF-SCOPE → simple response
            if intent == "general_question":
                print(
                        "\nI am an L2 production support assistant. "
                        "Please ask questions related to production support issues.\n"
                        )
                continue
            
            # ---- Step 1: build history text ----
            history_text = format_history(chat_history)

            # ---- Step 2: RAG ----
            answer_text, reference_docs = generate_answer(query=query,history=history_text,)
            
            # ---- Step 3: structured ----
            structured = to_structured_response(answer_text)
            # 🔥 inject references (deterministic, not LLM guessed)
            structured.reference_docs = reference_docs

            # ---- Step 4: display ----
            print_structured_response(structured)

            # ---- Step 5: update memory ----
            chat_history.append(
                {
                    "user": query,
                    "assistant": answer_text,
                }
            )

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting.")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# ------------------------------------------------
# Entry
# ------------------------------------------------

if __name__ == "__main__":
    main()
