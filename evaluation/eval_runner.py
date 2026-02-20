# evaluation/eval_runner.py — Enterprise Evaluation Harness

## 🎯 Objective

#Run automated evaluation against your golden dataset and produce **production-style metrics**.
#This runner evaluates:
#   * intent accuracy
#   * routing correctness
#   * retrieval grounding
#   * service match
#   * guardrail behavior
#This is exactly the layer reviewers look for in serious GenAI systems.

# ============================================
# evaluation/eval_runner.py
# ============================================

import json
from pathlib import Path
from typing import List, Dict, Any

from chains.intent_chain import classify_intent
from retrieval.routing import route_query
from chains.rag_chain import _extract_reference_ids  # reuse your helper
from chains.rag_chain import generate_answer, _build_context 
from evaluation.llm_judge import judge_answer

DATA_PATH = Path("evaluation/eval_dataset.json")


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def load_dataset() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_source_types(docs) -> List[str]:
    return list({d.metadata.get("source_type") for d in docs})


def get_services(docs) -> List[str]:
    return list({d.metadata.get("service") for d in docs})


# ------------------------------------------------
# Metric accumulator
# ------------------------------------------------

class Metrics:
    def __init__(self):
        self.total = 0
        self.intent_correct = 0
        self.primary_source_correct = 0
        self.acceptable_source_correct = 0
        self.service_match = 0
        self.reference_recall_hits = 0
        self.guardrail_correct = 0
        self.guardrail_total = 0
        self.judge_runs = 0 
        self.judge_grounded = 0 
        self.judge_useful = 0 
        self.judge_hallucinations = 0 
        self.judge_score_sum = 0

    def report(self):
        def pct(x, y):
            return round(100 * x / y, 2) if y else 0.0

        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)

        print(f"Total Cases: {self.total}")
        print(f"Intent Accuracy: {pct(self.intent_correct, self.total)}%")
        print(
            f"Primary Source Accuracy: {pct(self.primary_source_correct, self.total)}%"
        )
        print(
            f"Acceptable Source Accuracy: {pct(self.acceptable_source_correct, self.total)}%"
        )
        print(f"Service Match Rate: {pct(self.service_match, self.total)}%")
        print(
            f"Reference Recall: {pct(self.reference_recall_hits, self.total)}%"
        )
        print(
            f"Guardrail Accuracy: {pct(self.guardrail_correct, self.guardrail_total)}%"
        )
        print( f"Grounded Rate: {pct(self.judge_grounded, self.judge_runs)}%" ) 
        print( f"Useful Steps Rate: {pct(self.judge_useful, self.judge_runs)}%" ) 
        print( f"Hallucination Rate: {pct(self.judge_hallucinations, self.judge_runs)}%" ) 
        avg_score = ( round(self.judge_score_sum / self.judge_runs, 2) if self.judge_runs else 0 ) 
        print(f"Avg Judge Score: {avg_score}")
        print("=" * 60 + "\n")


# ------------------------------------------------
# Core evaluation
# ------------------------------------------------

def evaluate():
    dataset = load_dataset()
    metrics = Metrics()

    for row in dataset:
        metrics.total += 1

        query = row["query"]
        expected_intent = row["expected_intent"]
        expected_primary = row.get("expected_primary_source")
        acceptable_sources = row.get("acceptable_sources", [])
        expected_services = row.get("expected_services", [])
        expected_refs = set(row.get("expected_reference_ids", []))
        is_oos = row.get("is_out_of_scope", False)

        # =================================================
        # Step 1: Intent Classification
        # =================================================
        pred_intent = classify_intent(query)

        if pred_intent == expected_intent:
            metrics.intent_correct += 1

        # =================================================
        # Step 2: Guardrail check (OUT OF SCOPE)
        # =================================================
        if is_oos:
            metrics.guardrail_total += 1
            if pred_intent == "general_question":
                metrics.guardrail_correct += 1
            continue  # skip retrieval + judge

        # =================================================
        # Step 3: Retrieval via router
        # =================================================
        docs = route_query(
            query=query,
            intent=pred_intent,
            service=None,
        )

        if not docs:
            continue

        source_types = get_source_types(docs)
        services = get_services(docs)
        retrieved_refs = set(_extract_reference_ids(docs))

        # =================================================
        # Step 4: Primary source accuracy
        # =================================================
        if expected_primary and expected_primary in source_types:
            metrics.primary_source_correct += 1

        # =================================================
        # Step 5: Acceptable source accuracy
        # =================================================
        if any(src in source_types for src in acceptable_sources):
            metrics.acceptable_source_correct += 1

        # =================================================
        # Step 6: Service match
        # =================================================
        if any(svc in services for svc in expected_services):
            metrics.service_match += 1

        # =================================================
        # Step 7: Reference recall
        # =================================================
        if expected_refs:
            if expected_refs.intersection(retrieved_refs):
                metrics.reference_recall_hits += 1
        else:
            metrics.reference_recall_hits += 1

        # =================================================
        # Step 8: ⭐ LLM Judge Evaluation
        # =================================================
        try:
            context = _build_context(docs)

            answer_text, _ = generate_answer(
                query=query,
                history="",
            )

            judge = judge_answer(
                query=query,
                context=context,
                answer=answer_text,
            )

            metrics.judge_runs += 1

            if judge.grounded:
                metrics.judge_grounded += 1

            if judge.useful_steps:
                metrics.judge_useful += 1

            if judge.hallucination:
                metrics.judge_hallucinations += 1

            metrics.judge_score_sum += judge.overall_score

        except Exception as e:
            print(f"⚠️ Judge skipped for query due to error: {e}")

    # =====================================================
    # Final Report
    # =====================================================
    metrics.report()


# ------------------------------------------------
# Entry
# ------------------------------------------------

if __name__ == "__main__":
    evaluate()