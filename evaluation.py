from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"
QUERIES_PATH = EVAL_DIR / "recruiter_queries.json"
RESULT_PATH = EVAL_DIR / "latest_evaluation.json"

NLP_JSON = BASE_DIR / "output" / "nlp" / "resumes.json"
LLM_JSON = BASE_DIR / "output" / "llm" / "resumes.json"
MANIFEST = BASE_DIR / "data" / "resume_manifest.json"
MD_DIR = BASE_DIR / "md_data"


def load_json(path: Path, default):
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#./@ -]", "", text)
    return text.strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        if value.strip():
            out.append(value.strip())
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_strings(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(flatten_strings(v))
    return out


def dedupe_norm(items: list[Any]) -> set[str]:
    result = set()
    for item in items:
        value = normalize_text(item)
        if value:
            result.add(value)
    return result


def get_application_no(record: dict[str, Any]) -> str:
    return str(record.get("application_no") or record.get("Application_No") or "").upper()


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def extract_years_from_text(text: str) -> float | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text.lower())
    if match:
        return float(match.group(1))
    return None


def extract_predicted_experience(record: dict[str, Any]) -> float | None:
    values: list[str] = []
    values.extend(flatten_strings(record.get("experience", [])))
    candidate = record.get("candidate", {}) or {}
    values.extend(flatten_strings(candidate.get("personal_summary", "")))
    values.extend(flatten_strings(record.get("summary", "")))
    for value in values:
        years = extract_years_from_text(value)
        if years is not None:
            return years
    return None


def extract_predicted_cgpa(record: dict[str, Any]) -> float | None:
    for education in as_list(record.get("education", [])):
        text = " ".join(flatten_strings(education))
        match = re.search(r"(?:cgpa|gpa)\s*[:=-]?\s*(\d+(?:\.\d+)?)", text, re.I)
        if match:
            return float(match.group(1))
    return None


def extract_predicted_college(record: dict[str, Any]) -> str:
    for education in as_list(record.get("education", [])):
        if isinstance(education, dict):
            for key in ("institution", "college", "university", "school", "College", "Institution"):
                if education.get(key):
                    return str(education[key])
    return ""


def extract_predicted_degree(record: dict[str, Any]) -> str:
    for education in as_list(record.get("education", [])):
        if isinstance(education, dict):
            for key in ("degree", "Degree", "qualification", "course", "program"):
                if education.get(key):
                    return str(education[key])
    return ""


def extract_predicted_record(record: dict[str, Any]) -> dict[str, Any]:
    candidate = record.get("candidate", {}) or {}
    contact = candidate.get("contact", {}) or {}
    education_text = " ".join(flatten_strings(record.get("education", [])))
    projects = record.get("projects", [])
    project_texts = flatten_strings(projects)
    certifications = flatten_strings(record.get("certifications", []))
    return {
        "application_no": get_application_no(record),
        "name": candidate.get("name") or record.get("name", ""),
        "email": contact.get("email") or record.get("email", ""),
        "phone": contact.get("phone") or record.get("phone", ""),
        "college": extract_predicted_college(record) or education_text,
        "degree": extract_predicted_degree(record) or education_text,
        "skills": as_list(record.get("skills", [])),
        "certifications": certifications,
        "projects": project_texts,
        "experience_years": extract_predicted_experience(record),
        "cgpa": extract_predicted_cgpa(record),
    }


def scalar_match(pred: Any, truth: Any) -> bool:
    p = normalize_text(pred)
    t = normalize_text(truth)
    return bool(p) and bool(t) and p == t


def set_prf(pred: set[str], truth: set[str]) -> dict[str, float]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_parser(prediction_path: Path, parser_name: str, ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_raw = load_json(prediction_path, [])
    if isinstance(predicted_raw, dict):
        for key in ("resumes", "records", "candidates"):
            if isinstance(predicted_raw.get(key), list):
                predicted_raw = predicted_raw[key]
                break
    pred_by_app = {get_application_no(r): extract_predicted_record(r) for r in predicted_raw if isinstance(r, dict)}
    truth_by_app = {str(r.get("application_no", "")).upper(): r for r in ground_truth}

    scalar_fields = ["name", "email", "phone"]
    scalar_correct = 0
    scalar_total = 0
    skill_tp = skill_fp = skill_fn = 0
    cert_tp = cert_fp = cert_fn = 0
    exp_correct = 0
    exp_total = 0
    cgpa_abs_error = []
    cgpa_within_01 = 0
    cgpa_total = 0
    record_coverage = len(set(pred_by_app) & set(truth_by_app)) / len(truth_by_app) if truth_by_app else 0.0

    per_field_accuracy: dict[str, float] = {}

    for app_no, truth in truth_by_app.items():
        pred = pred_by_app.get(app_no, {})
        for field in scalar_fields:
            scalar_total += 1
            if scalar_match(pred.get(field), truth.get(field)):
                scalar_correct += 1

        pskills = dedupe_norm(pred.get("skills", []))
        tskills = dedupe_norm(truth.get("skills", []))
        skill_tp += len(pskills & tskills)
        skill_fp += len(pskills - tskills)
        skill_fn += len(tskills - pskills)

        pcert = dedupe_norm(pred.get("certifications", []))
        tcert = dedupe_norm(truth.get("certifications", []))
        cert_tp += len(pcert & tcert)
        cert_fp += len(pcert - tcert)
        cert_fn += len(tcert - pcert)

        pexp = pred.get("experience_years")
        texp = parse_number(truth.get("experience_years"))
        if texp is not None:
            exp_total += 1
            if pexp is not None and abs(float(pexp) - float(texp)) <= 0.25:
                exp_correct += 1

        pcgpa = pred.get("cgpa")
        tcgpa = parse_number(truth.get("cgpa"))
        if tcgpa is not None:
            cgpa_total += 1
            if pcgpa is not None:
                error = abs(float(pcgpa) - float(tcgpa))
                cgpa_abs_error.append(error)
                if error <= 0.1:
                    cgpa_within_01 += 1

    def counts_prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    skills = counts_prf(skill_tp, skill_fp, skill_fn)
    certs = counts_prf(cert_tp, cert_fp, cert_fn)
    scalar_accuracy = scalar_correct / scalar_total if scalar_total else 0.0
    exp_accuracy = exp_correct / exp_total if exp_total else 0.0
    cgpa_mae = sum(cgpa_abs_error) / len(cgpa_abs_error) if cgpa_abs_error else None
    cgpa_accuracy = cgpa_within_01 / cgpa_total if cgpa_total else 0.0
    overall_f1 = (skills["f1"] + certs["f1"] + scalar_accuracy + exp_accuracy + cgpa_accuracy) / 5

    return {
        "parser": parser_name,
        "records_expected": len(truth_by_app),
        "records_predicted": len(pred_by_app),
        "record_coverage": record_coverage,
        "record_coverage_percent": round(record_coverage * 100, 2),
        "scalar_accuracy": round(scalar_accuracy, 4),
        "scalar_accuracy_percent": round(scalar_accuracy * 100, 2),
        "skills": {k: round(v, 4) if isinstance(v, float) else v for k, v in skills.items()},
        "certifications": {k: round(v, 4) if isinstance(v, float) else v for k, v in certs.items()},
        "experience_accuracy": round(exp_accuracy, 4),
        "experience_accuracy_percent": round(exp_accuracy * 100, 2),
        "cgpa_mae": round(cgpa_mae, 4) if cgpa_mae is not None else None,
        "cgpa_within_0_1_accuracy": round(cgpa_accuracy, 4),
        "cgpa_within_0_1_accuracy_percent": round(cgpa_accuracy * 100, 2),
        "overall_f1": round(overall_f1, 4),
        "overall_f1_percent": round(overall_f1 * 100, 2),
    }


def binary_ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    retrieved = retrieved[:k]
    dcg = 0.0
    for i, app_no in enumerate(retrieved, start=1):
        if app_no in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(predicted: list[str], expected: list[str], k: int = 10) -> dict[str, float]:
    predicted = [str(x).upper() for x in predicted]
    expected_set = {str(x).upper() for x in expected}
    top = predicted[:k]
    hits = [x for x in top if x in expected_set]
    precision = len(hits) / k if k else 0.0
    recall = len(hits) / len(expected_set) if expected_set else 0.0
    reciprocal_rank = 0.0
    for rank, app_no in enumerate(predicted, start=1):
        if app_no in expected_set:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": reciprocal_rank,
        "ndcg_at_k": binary_ndcg(predicted, expected_set, k),
        "relevant": len(expected_set),
        "returned": len(predicted),
        "hits_at_k": len(hits),
    }


def benchmark_retrieval(run_search) -> dict[str, Any]:
    queries = load_json(QUERIES_PATH, [])
    rows = []
    started_all = time.perf_counter()
    for item in queries:
        query = str(item.get("query", ""))
        expected = item.get("expected_application_nos", [])
        started = time.perf_counter()
        try:
            result = run_search(query)
            candidates = result.get("candidates", []) if isinstance(result, dict) else []
            predicted = [str(c.get("application_no", "")).upper() for c in candidates if isinstance(c, dict)]
            metrics = retrieval_metrics(predicted, expected, 10)
            metrics["latency_seconds"] = round(time.perf_counter() - started, 3)
            rows.append({"query": query, "status": "success", **metrics})
        except Exception as exc:
            rows.append({
                "query": query,
                "status": "failed",
                "error": str(exc),
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "ndcg_at_k": 0.0,
                "latency_seconds": round(time.perf_counter() - started, 3),
            })

    successful = [r for r in rows if r.get("status") == "success"]
    total = len(rows)
    aggregate = {
        "available": bool(successful),
        "precision_at_10": sum(r["precision_at_k"] for r in successful) / len(successful) if successful else 0.0,
        "recall_at_10": sum(r["recall_at_k"] for r in successful) / len(successful) if successful else 0.0,
        "mrr": sum(r["mrr"] for r in successful) / len(successful) if successful else 0.0,
        "ndcg_at_10": sum(r["ndcg_at_k"] for r in successful) / len(successful) if successful else 0.0,
        "avg_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows) if rows else 0.0,
        "success_rate": len(successful) / total if total else 0.0,
        "queries_total": total,
        "queries_successful": len(successful),
        "error_summary": rows[0].get("error", "") if rows and not successful else "",
        "elapsed_seconds": round(time.perf_counter() - started_all, 2),
    }
    return {"aggregate": aggregate, "queries": rows}


def conversion_metrics() -> dict[str, Any]:
    manifest = load_json(MANIFEST, [])
    total = len(manifest)
    converted = sum(1 for r in manifest if r.get("markdown_status") == "success")
    failed = sum(1 for r in manifest if r.get("markdown_status") == "failed")
    times = [float(r.get("markdown_conversion_seconds")) for r in manifest if r.get("markdown_conversion_seconds") is not None]
    avg_time = sum(times) / len(times) if times else None
    return {
        "total": total,
        "converted": converted,
        "failed": failed,
        "success_rate": converted / total if total else 0.0,
        "success_rate_percent": round((converted / total) * 100, 2) if total else 0.0,
        "average_seconds_per_resume": round(avg_time, 3) if avg_time is not None else None,
        "throughput_resumes_per_minute": round(60 / avg_time, 2) if avg_time and avg_time > 0 else None,
    }


def run_full_evaluation(run_search=None, include_retrieval: bool = True) -> dict[str, Any]:
    ground_truth = load_json(GROUND_TRUTH_PATH, [])
    if not ground_truth:
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_PATH}")

    result: dict[str, Any] = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": {
            "ground_truth_records": len(ground_truth),
            "recruiter_queries": len(load_json(QUERIES_PATH, [])),
        },
        "conversion": conversion_metrics(),
        "parsers": {},
        "retrieval": None,
    }

    if NLP_JSON.exists():
        result["parsers"]["nlp"] = evaluate_parser(NLP_JSON, "Traditional NLP", ground_truth)
    if LLM_JSON.exists():
        result["parsers"]["llm"] = evaluate_parser(LLM_JSON, "LLM", ground_truth)

    if include_retrieval and run_search is not None and (NLP_JSON.exists() or LLM_JSON.exists()):
        result["retrieval"] = benchmark_retrieval(run_search)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def load_latest_evaluation() -> dict[str, Any] | None:
    data = load_json(RESULT_PATH, None)
    return data if isinstance(data, dict) else None
