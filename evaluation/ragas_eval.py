"""
evaluation/ragas_eval.py
Evaluates the RAG system using RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

Also runs against a manually curated golden dataset.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

import requests
import mlflow
from datasets import Dataset

sys.path.append(str(Path(__file__).parent.parent))

API_URL = "http://localhost:8000"
RESULTS_DIR = Path("./evaluation")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Golden dataset — 50 curated diabetes Q&A pairs
# ---------------------------------------------------------------------------

GOLDEN_DATASET = [
    {
        "question": "What is the recommended HbA1c target for most non-pregnant adults with type 2 diabetes?",
        "ground_truth": "Less than 7.0% (53 mmol/mol) per ADA 2024 Standards of Care.",
    },
    {
        "question": "What is the first-line pharmacological treatment for type 2 diabetes?",
        "ground_truth": "Metformin is recommended as the preferred initial pharmacological agent for type 2 diabetes.",
    },
    {
        "question": "What blood pressure target is recommended for patients with diabetes?",
        "ground_truth": "A blood pressure target of less than 130/80 mmHg is recommended for most patients with diabetes.",
    },
    {
        "question": "What are the main microvascular complications of diabetes?",
        "ground_truth": "The main microvascular complications of diabetes are diabetic retinopathy, nephropathy, and neuropathy.",
    },
    {
        "question": "What is the role of SGLT2 inhibitors in type 2 diabetes with heart failure?",
        "ground_truth": "SGLT2 inhibitors reduce hospitalisation for heart failure and are recommended in T2DM patients with established heart failure.",
    },
    {
        "question": "When is metformin contraindicated in diabetes patients?",
        "ground_truth": "Metformin is contraindicated in patients with eGFR less than 30 mL/min/1.73m² due to risk of lactic acidosis.",
    },
    {
        "question": "What fasting plasma glucose level is used to diagnose diabetes?",
        "ground_truth": "A fasting plasma glucose of 126 mg/dL (7.0 mmol/L) or higher on two occasions is diagnostic of diabetes.",
    },
    {
        "question": "What lifestyle interventions are recommended for type 2 diabetes prevention?",
        "ground_truth": "Weight loss of 5-7% body weight and at least 150 minutes per week of moderate physical activity are recommended to prevent type 2 diabetes.",
    },
    {
        "question": "What is the recommended LDL cholesterol target for diabetic patients with cardiovascular disease?",
        "ground_truth": "An LDL cholesterol of less than 70 mg/dL is recommended for diabetic patients with established cardiovascular disease.",
    },
    {
        "question": "How often should HbA1c be monitored in stable type 2 diabetes?",
        "ground_truth": "HbA1c should be measured every 3 months initially and every 6 months once stable glycaemic targets are achieved.",
    },
    {
        "question": "What is diabetic ketoacidosis (DKA)?",
        "ground_truth": "DKA is a life-threatening complication of diabetes characterised by hyperglycaemia, ketosis, and metabolic acidosis, most common in type 1 diabetes.",
    },
    {
        "question": "What are GLP-1 receptor agonists used for in diabetes?",
        "ground_truth": "GLP-1 receptor agonists lower blood glucose, promote weight loss, and reduce cardiovascular risk in type 2 diabetes.",
    },
    {
        "question": "What screening is recommended for diabetic nephropathy?",
        "ground_truth": "Annual screening for albuminuria and eGFR is recommended for all patients with diabetes to detect diabetic nephropathy early.",
    },
    {
        "question": "At what HbA1c level should insulin therapy be initiated in type 2 diabetes?",
        "ground_truth": "Insulin therapy should be considered when HbA1c remains above 10% despite oral therapy or when hyperglycaemic symptoms are present.",
    },
    {
        "question": "What is the Diabetes Prevention Program (DPP)?",
        "ground_truth": "The DPP was a landmark RCT showing intensive lifestyle intervention reduced progression to type 2 diabetes by 58% in high-risk individuals.",
    },
    {
        "question": "What foot care is recommended for diabetic patients?",
        "ground_truth": "Annual comprehensive foot examination including assessment of sensation, pulses, and skin integrity is recommended for all diabetic patients.",
    },
    {
        "question": "What is the target BMI for overweight patients with type 2 diabetes?",
        "ground_truth": "Weight loss of 5-10% is beneficial; bariatric surgery may be considered for BMI ≥ 35 with inadequately controlled diabetes.",
    },
    {
        "question": "What is hyperosmolar hyperglycaemic state (HHS)?",
        "ground_truth": "HHS is a serious complication of type 2 diabetes with extreme hyperglycaemia, severe dehydration, and hyperosmolality without significant ketosis.",
    },
    {
        "question": "Which diabetes medication has shown cardiovascular mortality benefit?",
        "ground_truth": "SGLT2 inhibitors (empagliflozin, canagliflozin) and GLP-1 agonists (liraglutide, semaglutide) have shown cardiovascular mortality benefits in RCTs.",
    },
    {
        "question": "What is the recommended eye examination schedule for diabetic patients?",
        "ground_truth": "Dilated eye examination is recommended at diagnosis for type 2 diabetes and annually thereafter to screen for diabetic retinopathy.",
    },
    {
        "question": "What is the UKPDS study and its significance?",
        "ground_truth": "The UKPDS showed that intensive blood glucose control in newly diagnosed type 2 diabetes significantly reduces microvascular complications.",
    },
    {
        "question": "What dietary pattern is recommended for type 2 diabetes management?",
        "ground_truth": "Mediterranean, DASH, and low-carbohydrate dietary patterns are all effective for glycaemic control in type 2 diabetes.",
    },
    {
        "question": "What is the role of self-monitoring blood glucose (SMBG) in type 2 diabetes?",
        "ground_truth": "SMBG is recommended for insulin-treated patients; continuous glucose monitoring (CGM) is increasingly preferred for comprehensive glucose data.",
    },
    {
        "question": "How does chronic kidney disease affect diabetes drug selection?",
        "ground_truth": "Metformin should be dose-reduced at eGFR 30-45 and stopped below 30. SGLT2 inhibitors have reduced efficacy at low eGFR. GLP-1 agonists are generally safe.",
    },
    {
        "question": "What vaccinations are recommended for people with diabetes?",
        "ground_truth": "Annual influenza, pneumococcal, hepatitis B, and COVID-19 vaccines are recommended for adults with diabetes due to increased infection risk.",
    },
    {
        "question": "What is gestational diabetes mellitus (GDM)?",
        "ground_truth": "GDM is glucose intolerance first diagnosed during pregnancy, associated with increased risk of macrosomia, perinatal complications, and future type 2 diabetes.",
    },
    {
        "question": "What is the recommended screening for GDM?",
        "ground_truth": "Universal screening at 24-28 weeks gestation using a 75g oral glucose tolerance test (OGTT) is recommended.",
    },
    {
        "question": "What is the mechanism of action of SGLT2 inhibitors?",
        "ground_truth": "SGLT2 inhibitors block sodium-glucose co-transporter 2 in the proximal renal tubule, increasing urinary glucose excretion and lowering blood glucose.",
    },
    {
        "question": "What are the side effects of SGLT2 inhibitors?",
        "ground_truth": "Common side effects include genital mycotic infections, urinary tract infections, and rare risk of diabetic ketoacidosis and Fournier's gangrene.",
    },
    {
        "question": "What is prediabetes and how is it defined?",
        "ground_truth": "Prediabetes is defined as impaired fasting glucose (100-125 mg/dL) or impaired glucose tolerance (2h OGTT 140-199 mg/dL) or HbA1c 5.7-6.4%.",
    },
    {
        "question": "What physical activity is recommended for type 2 diabetes?",
        "ground_truth": "At least 150 minutes of moderate-intensity aerobic activity per week and resistance training 2-3 times per week are recommended.",
    },
    {
        "question": "What is the ACCORD trial and what did it show?",
        "ground_truth": "The ACCORD trial showed that intensive glycaemic control targeting HbA1c below 6% increased cardiovascular mortality in high-risk T2DM patients.",
    },
    {
        "question": "What causes diabetic peripheral neuropathy?",
        "ground_truth": "Diabetic peripheral neuropathy results from chronic hyperglycaemia causing oxidative stress, inflammation, and microvascular damage to peripheral nerves.",
    },
    {
        "question": "What are first-line treatments for painful diabetic neuropathy?",
        "ground_truth": "Pregabalin, duloxetine, and gabapentin are first-line pharmacological treatments for painful diabetic peripheral neuropathy.",
    },
    {
        "question": "What is the role of ACE inhibitors in diabetic nephropathy?",
        "ground_truth": "ACE inhibitors and ARBs reduce proteinuria and slow progression of diabetic nephropathy and are recommended in patients with albuminuria.",
    },
    {
        "question": "What is continuous glucose monitoring (CGM)?",
        "ground_truth": "CGM measures interstitial glucose continuously, providing real-time glucose values, trends, and alerts for hypoglycaemia and hyperglycaemia.",
    },
    {
        "question": "What is the diagnostic criterion for type 1 diabetes?",
        "ground_truth": "Type 1 diabetes is diagnosed by hyperglycaemia with positive islet autoantibodies (GAD, IA-2, ZnT8) and absolute insulin deficiency.",
    },
    {
        "question": "What is insulin resistance?",
        "ground_truth": "Insulin resistance is a condition where cells fail to respond normally to insulin, requiring higher insulin levels to maintain glucose homeostasis.",
    },
    {
        "question": "What is the dawn phenomenon in diabetes?",
        "ground_truth": "The dawn phenomenon is early morning hyperglycaemia caused by nocturnal growth hormone and cortisol surges increasing insulin resistance.",
    },
    {
        "question": "What is bariatric surgery's effect on type 2 diabetes?",
        "ground_truth": "Bariatric surgery achieves type 2 diabetes remission in 30-63% of patients and is recommended for BMI ≥ 35 with inadequately controlled diabetes.",
    },
    {
        "question": "What is the role of zinc in diabetes?",
        "ground_truth": "Zinc is essential for insulin synthesis, storage, and secretion; zinc transporter 8 (ZnT8) autoantibodies are a marker for type 1 diabetes.",
    },
    {
        "question": "What is latent autoimmune diabetes in adults (LADA)?",
        "ground_truth": "LADA is a slowly progressive autoimmune form of diabetes in adults that initially resembles type 2 diabetes but has positive GAD antibodies.",
    },
    {
        "question": "What is the recommended protein intake for patients with diabetic nephropathy?",
        "ground_truth": "A protein intake of 0.8 g/kg body weight per day is recommended for patients with diabetic nephropathy to slow renal progression.",
    },
    {
        "question": "What is the mechanism of metformin?",
        "ground_truth": "Metformin primarily reduces hepatic glucose production by activating AMPK and inhibiting mitochondrial complex I, with additional gut effects.",
    },
    {
        "question": "What are the symptoms of hypoglycaemia?",
        "ground_truth": "Symptoms include sweating, tremor, palpitations, anxiety, confusion, and loss of consciousness; blood glucose below 70 mg/dL defines hypoglycaemia.",
    },
    {
        "question": "What is the 15-15 rule for hypoglycaemia treatment?",
        "ground_truth": "Consume 15 grams of fast-acting carbohydrate, wait 15 minutes, recheck glucose; repeat if still below 70 mg/dL.",
    },
    {
        "question": "What are DPP-4 inhibitors used for?",
        "ground_truth": "DPP-4 inhibitors (gliptins) increase incretin levels to lower postprandial glucose; they are weight-neutral with low hypoglycaemia risk.",
    },
    {
        "question": "What is the role of diet in managing gestational diabetes?",
        "ground_truth": "Medical nutrition therapy controlling carbohydrate intake is first-line treatment for GDM; insulin is added if glucose targets are not met.",
    },
    {
        "question": "What causes diabetic retinopathy?",
        "ground_truth": "Chronic hyperglycaemia causes retinal microvascular damage through pericyte loss, basement membrane thickening, and neovascularisation.",
    },
    {
        "question": "What is the EMPA-REG OUTCOME trial?",
        "ground_truth": "EMPA-REG OUTCOME showed empagliflozin reduced cardiovascular death by 38% and hospitalisation for heart failure by 35% in T2DM patients with CVD.",
    },
    {
        "question": "What is the glycaemic index and its relevance to diabetes?",
        "ground_truth": "The glycaemic index ranks foods by how quickly they raise blood glucose; low-GI foods produce slower glucose rises and are preferred in diabetes management.",
    },
]


# ---------------------------------------------------------------------------
# Collect RAG responses for evaluation
# ---------------------------------------------------------------------------

def collect_rag_responses(questions: list[dict], use_local: bool = True) -> list[dict]:
    """
    Query the running API for each question and collect responses + contexts.
    """
    print(f"Collecting RAG responses for {len(questions)} questions…")
    results = []

    for i, item in enumerate(questions):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1}/{len(questions)}] {q[:70]}…")

        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": q, "use_local_llm": use_local, "top_k": 5},
                timeout=120,
            )
            if resp.status_code != 200:
                print(f"    API error {resp.status_code} — skipping")
                continue

            data = resp.json()
            contexts = [c["chunk_preview"] for c in data.get("citations", [])]

            results.append({
                "question":     q,
                "answer":       data["answer"],
                "contexts":     contexts if contexts else ["No context retrieved"],
                "ground_truth": gt,
                "confidence":   data["confidence"],
                "grade":        data["evidence_grade"],
                "faithfulness_internal": data["faithfulness"],
            })

        except Exception as e:
            print(f"    Error: {e} — skipping")

    return results


# ---------------------------------------------------------------------------
# Run RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(results: list[dict]) -> dict:
    """Run RAGAS metrics on collected results."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        print("RAGAS not installed. Run: pip install ragas")
        return {}

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    print("\nRunning RAGAS evaluation (this may take a few minutes)…")
    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return ragas_result


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list[dict], ragas_scores: dict):
    """Save raw results and RAGAS scores to CSV and JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw results JSON
    raw_path = RESULTS_DIR / f"raw_results_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved → {raw_path}")

    # RAGAS scores JSON
    if ragas_scores:
        ragas_path = RESULTS_DIR / f"ragas_scores_{ts}.json"
        with open(ragas_path, "w") as f:
            json.dump(dict(ragas_scores), f, indent=2)
        print(f"RAGAS scores saved → {ragas_path}")

    # CSV summary
    csv_path = RESULTS_DIR / f"evaluation_summary_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "ground_truth", "answer",
            "confidence", "grade", "faithfulness_internal"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question":              r["question"],
                "ground_truth":          r["ground_truth"],
                "answer":                r["answer"][:300],
                "confidence":            r["confidence"],
                "grade":                 r["grade"],
                "faithfulness_internal": r["faithfulness_internal"],
            })
    print(f"CSV summary saved → {csv_path}")

    # MLflow logging
    try:
        mlflow.set_experiment("diabetes-rag-evaluation")
        with mlflow.start_run(run_name=f"eval_{ts}"):
            if ragas_scores:
                for metric, value in ragas_scores.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric, value)
            avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0
            mlflow.log_metric("avg_confidence", avg_conf)
            mlflow.log_metric("num_questions", len(results))
        print("Metrics logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(results: list[dict], ragas_scores: dict):
    print("\n" + "=" * 60)
    print("EVALUATION REPORT — Evidence-Verified RAG Diabetes Care")
    print("=" * 60)
    print(f"Total questions evaluated : {len(results)}")

    if results:
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        avg_faith = sum(r["faithfulness_internal"] for r in results) / len(results)
        grade_dist = {}
        for r in results:
            grade_dist[r["grade"]] = grade_dist.get(r["grade"], 0) + 1

        print(f"Average confidence score  : {avg_conf:.3f}")
        print(f"Average faithfulness      : {avg_faith:.3f}")
        print(f"Evidence grade distribution: {grade_dist}")

    if ragas_scores:
        print("\nRAGAS Metrics:")
        for metric, value in ragas_scores.items():
            if isinstance(value, (int, float)):
                print(f"  {metric:<30}: {value:.4f}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Diabetes RAG system")
    parser.add_argument("--local", action="store_true", default=True,
                        help="Use local LLM (Ollama)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of questions to evaluate (max 50)")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip RAGAS evaluation (faster)")
    args = parser.parse_args()

    questions = GOLDEN_DATASET[: args.limit]

    print(f"Starting evaluation on {len(questions)} questions…")
    results = collect_rag_responses(questions, use_local=args.local)

    ragas_scores = {}
    if not args.skip_ragas and results:
        ragas_scores = run_ragas_evaluation(results)

    save_results(results, ragas_scores)
    print_report(results, ragas_scores)
