# ui/app.py

from flask import Flask, render_template, request, jsonify


# 🔹 Import your existing pipeline pieces
from chains.intent_chain import classify_intent
from chains.rag_chain import generate_answer
from chains.structured_output import to_structured_response
from app import format_history  # if already exists

app = Flask(__name__)

# 🔹 simple in-memory chat history (per server)
chat_history = []


# ================================
# Home Page
# ================================
@app.route("/")
def home():
    return render_template("index.html")


# ================================
# Ask Endpoint
# ================================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Empty query"}), 400

        # --------------------------------
        # 1️⃣ Intent check (guardrail)
        # --------------------------------
        intent = classify_intent(query)

        if intent == "general_question":
            return jsonify(
                {
                    "message": "I am an L2 production support assistant. "
                    "Please ask questions related to production support issues."
                }
            )

        # --------------------------------
        # 2️⃣ Build history
        # --------------------------------
        history_text = format_history(chat_history)

        # --------------------------------
        # 3️⃣ RAG pipeline
        # --------------------------------
        answer_text, reference_docs = generate_answer(
        query=query,
        history=history_text,
        )

        structured = to_structured_response(answer_text)
        structured.reference_docs = list(reference_docs)
        

        # --------------------------------
        # 4️⃣ Update memory
        # --------------------------------
        chat_history.append(
            {
                "user": query,
                "assistant": answer_text,
            }
        )

        # --------------------------------
        # 5️⃣ Return JSON to UI
        # --------------------------------
        return jsonify(
            {
                "issue_summary": structured.issue_summary,
                "impacted_service": structured.impacted_service,
                "recommended_steps": structured.recommended_steps,
                "escalation_required": structured.escalation_required,
                "confidence": structured.confidence,
                "reference_docs": list(structured.reference_docs),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# Run
# ================================
if __name__ == "__main__":
    #app.run(debug=True, port=5000) #local
    app.run(host="0.0.0.0", port=5000) # render