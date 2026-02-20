# 🚀 AI Production Support Assistant (Ops Copilot)

An enterprise-style **L2 Production Support Assistant** powered by RAG, LangChain, and OpenAI.
The system helps operations teams quickly diagnose incidents, follow runbooks, and investigate alerts/tickets using natural language.

---

## 🧭 Overview

Modern production environments generate large volumes of runbooks, alerts, incidents, logs, and tickets. During outages, L2 engineers often lose time searching across multiple systems.

This project demonstrates an **Ops Copilot** that:

* understands support queries
* routes them intelligently
* retrieves the most relevant operational knowledge
* returns structured, actionable steps

The design mirrors real-world SRE / production support workflows in banking and trading environments.

---

## ✨ Key Features

* 🔎 **Multi-source RAG** over runbooks, incidents, alerts, logs, and tickets
* 🧠 **Intent-aware routing** for precise retrieval
* 📊 **Structured L2 responses** (steps, service, escalation, confidence)
* 🛡️ **Guardrails** for out-of-scope questions
* 💬 **Chat-style Flask UI**
* 📈 **LangSmith tracing** for observability
* 🧪 **Evaluation harness** with LLM judge
* 🏦 Realistic **banking/trade management scenarios**

---

## 🏗️ Architecture

**High-level flow**

User Query → Intent Classifier → Source Routing → Vector Retrieval → RAG Generation → Structured Output → UI

**Core capabilities**

* Semantic search using FAISS
* Metadata-aware filtering
* Prompt-engineered intent classification
* Deterministic reference attribution
* Production-style confidence scoring

---

## 📁 Project Structure

```
with_langchain/
│
├── ingestion/          # data loading and indexing
├── retrieval/          # retrievers, routing, filters
├── chains/             # intent + RAG chains
├── core/               # config, prompts, models
├── ui/                 # Flask web interface
│   ├── app.py
│   ├── templates/
│   └── static/
├── vector_store/       # FAISS index (generated)
├── evaluation/         # eval datasets and runner
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd with_langchain
```

---

### 2️⃣ Create and activate virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**

```bash
python -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=ops-support-assistant
```

> ⚠️ Never commit `.env` to GitHub.

---

### 5️⃣ Build the vector index (first time only)

```bash
python -m ingestion.build_index
```

---

### 6️⃣ Run the application

```bash
python -m ui.app
```

Open in browser:

```
http://localhost:5000
```

---

## 🧪 Example Queries

Try the following in the UI:

**Runbook**

```
disk usage high on trade server
```

**Alert**

```
critical alert trade settlement queue depth high
```

**Ticket**

```
SR-Trade-003 users reporting payment timeouts
```

**Incident**

```
why did eod reconciliation fail
```

**Guardrail**

```
what is the weather today
```

---

## 📊 Structured Response Format

The assistant returns:

* Issue Summary
* Impacted Service
* Recommended Steps
* Escalation Required
* Confidence Level
* Reference Documents

This mirrors real L2 support workflows.

---

## 🔍 Evaluation

An evaluation harness is included to measure:

* intent accuracy
* retrieval quality
* answer correctness
* confidence calibration

Run evaluation:

```bash
python evaluation/eval_runner.py
```

---

## 🧠 Observability (LangSmith)

Tracing is enabled via LangSmith for:

* prompt inspection
* retrieval debugging
* latency tracking
* token usage

Configure via environment variables.

---

## 🌐 Deployment

The application is designed to deploy easily on:

* Render (recommended for demo)
* Railway
* Docker + Cloud (advanced)

**Start command for production**

```bash
gunicorn ui.app:app
```

---

## 🛡️ Guardrails

The assistant intentionally refuses:

* weather queries
* personal questions
* non-ops chit-chat

This prevents hallucinations and keeps the system production-focused.

---

## 🔮 Future Enhancements

* Cross-encoder reranking
* Streaming responses
* RBAC / authentication
* Real log ingestion pipeline
* Kubernetes deployment
* FastAPI migration

---

## 👩‍💻 Author

**Pallavi**
L2 Production Support Specialist | GenAI Engineer

This project demonstrates how GenAI can augment real-world production support operations in banking/trading environments.

---

## 📜 License

This project is for educational and demonstration purposes.
