from __future__ import annotations

import hashlib
import importlib
import io
import json
import re
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    import fitz
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import pymupdf4llm
except Exception:
    pymupdf4llm = None

# Make the existing HireNex modules (nlp.py, llm.py, retrival.py) importable.
BASE_DIR = Path(__file__).resolve().parents[1]
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

APP_TITLE = "HireNex"
APP_TAGLINE = "Intelligent Candidate Screening & Retrieval Platform"
DATA_DIR = BASE_DIR / "data"
RESUME_DIR = DATA_DIR / "resumes"
OUTPUT_DIR = BASE_DIR / "output"
NLP_OUTPUT_DIR = OUTPUT_DIR / "nlp"
LLM_OUTPUT_DIR = OUTPUT_DIR / "llm"
MD_DIR = BASE_DIR / "md_data"
RAG_OUTPUT_DIR = BASE_DIR / "ragoutput"
RAG_DIR = BASE_DIR / "rag"
MANIFEST_PATH = DATA_DIR / "resume_manifest.json"
EVALUATION_DIR = BASE_DIR / "evaluation"
EVALUATION_RESULT = EVALUATION_DIR / "latest_evaluation.json"

for directory in [RESUME_DIR, NLP_OUTPUT_DIR, LLM_OUTPUT_DIR, MD_DIR, RAG_OUTPUT_DIR, RAG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hirenex-job")
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


class SearchRequest(BaseModel):
    query: str


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def create_job(kind: str, total: int = 0) -> str:
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "total": total,
            "completed": 0,
            "failed": 0,
            "percentage": 0,
            "phase": "queued",
            "current": "",
            "message": "Queued",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def update_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        return json_safe(job) if job else {}


def run_job(job_id: str, function, *args, **kwargs) -> None:
    update_job(
        job_id,
        status="running",
        phase="initializing",
        started_at=datetime.now(timezone.utc).isoformat(),
        message="Starting job…",
    )
    try:
        result = function(job_id, *args, **kwargs)
        final_job = get_job(job_id)
        failure_count = int(final_job.get("failed", 0) or 0)
        if isinstance(result, dict):
            if isinstance(result.get("errors"), list):
                failure_count = len(result["errors"])
            elif isinstance(result.get("failed"), list):
                failure_count = len(result["failed"])
        total = int(final_job.get("total", 0) or 0)
        update_job(
            job_id,
            status="completed",
            phase="completed",
            percentage=100 if total else final_job.get("percentage", 0),
            failed=failure_count,
            finished_at=datetime.now(timezone.utc).isoformat(),
            result=json_safe(result),
            message=(f"Completed with {failure_count} failure(s)" if failure_count else "Completed successfully"),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            phase="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
            message=f"Failed: {exc}",
        )


def load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_manifest(manifest: list[dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def next_application_number(manifest: list[dict[str, Any]]) -> str:
    numbers = []
    for item in manifest:
        match = re.fullmatch(r"APP(\d+)", str(item.get("application_no", "")))
        if match:
            numbers.append(int(match.group(1)))
    for path in RESUME_DIR.glob("APP*.pdf"):
        match = re.fullmatch(r"APP(\d+)", path.stem)
        if match:
            numbers.append(int(match.group(1)))
    for path in MD_DIR.glob("APP*.md"):
        match = re.fullmatch(r"APP(\d+)", path.stem)
        if match:
            numbers.append(int(match.group(1)))
    return f"APP{max(numbers, default=0) + 1:06d}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_pdf_date(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if text.startswith("D:"):
        text = text[2:]
    match = re.match(r"^(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", text)
    if not match:
        return str(value)
    year, month, day, hour, minute, second = match.groups()
    try:
        return datetime(
            int(year), int(month or 1), int(day or 1),
            int(hour or 0), int(minute or 0), int(second or 0)
        ).isoformat()
    except ValueError:
        return str(value)


def extract_pdf_metadata(pdf_bytes: bytes) -> dict[str, Any]:
    result = {
        "pdf_creation_date": "",
        "pdf_modification_date": "",
        "pdf_page_count": None,
        "pdf_metadata": {},
    }
    if PdfReader is None:
        return result
    try:
        reader = PdfReader(stream=pdf_bytes)
        result["pdf_page_count"] = len(reader.pages)
        metadata = reader.metadata
        if metadata:
            safe = {str(k): str(v) for k, v in metadata.items()}
            result["pdf_metadata"] = safe
            result["pdf_creation_date"] = parse_pdf_date(metadata.get("/CreationDate"))
            result["pdf_modification_date"] = parse_pdf_date(metadata.get("/ModDate"))
    except Exception:
        pass
    return result


def is_safe_zip_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    if ".." in Path(normalized).parts:
        return False
    return Path(normalized).suffix.lower() == ".pdf"


def collect_pdf_payloads(items: list[tuple[str, bytes]]) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_bytes = 0
    max_batch_bytes = 500 * 1024 * 1024

    for name, raw in items:
        suffix = Path(name).suffix.lower()
        if suffix == ".pdf":
            total_bytes += len(raw)
            if total_bytes > max_batch_bytes:
                raise ValueError("The selected batch exceeds 500 MB. Process smaller batches.")
            payloads.append({"source_name": name, "pdf_bytes": raw})
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        if not is_safe_zip_member(member.filename):
                            if Path(member.filename).suffix.lower() == ".pdf":
                                warnings.append(f"Skipped unsafe ZIP path: {member.filename}")
                            continue
                        pdf_bytes = archive.read(member)
                        total_bytes += len(pdf_bytes)
                        if total_bytes > max_batch_bytes:
                            raise ValueError("The extracted ZIP batch exceeds 500 MB. Process smaller batches.")
                        payloads.append({"source_name": member.filename, "pdf_bytes": pdf_bytes})
            except zipfile.BadZipFile:
                warnings.append(f"Invalid ZIP file skipped: {name}")
        else:
            warnings.append(f"Unsupported file skipped: {name}")

    return payloads, warnings


def ingest_resumes(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = load_manifest()
    by_hash = {str(x.get("sha256")): x for x in manifest if x.get("sha256")}
    created = []
    duplicates = []

    for payload in payloads:
        pdf_bytes = payload["pdf_bytes"]
        source_name = payload["source_name"]
        digest = sha256_bytes(pdf_bytes)
        if digest in by_hash:
            duplicates.append({"application_no": by_hash[digest]["application_no"], "source_name": source_name})
            continue

        app_no = next_application_number(manifest)
        pdf_path = RESUME_DIR / f"{app_no}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        record = {
            "application_no": app_no,
            "stored_pdf": pdf_path.name,
            "source_filename": Path(source_name).name,
            "source_path_in_upload": source_name.replace("\\", "/"),
            "sha256": digest,
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            **extract_pdf_metadata(pdf_bytes),
            "markdown_file": f"{app_no}.md",
            "markdown_status": "pending",
            "nlp_status": "pending",
            "llm_status": "pending",
        }
        manifest.append(record)
        by_hash[digest] = record
        created.append(record)

    save_manifest(manifest)
    return {"created": created, "duplicates": duplicates}


def convert_one_pdf_to_markdown(pdf_path: Path) -> str:
    if pymupdf4llm is None:
        raise RuntimeError("PyMuPDF4LLM is not installed. Run: python -m pip install pymupdf4llm")
    markdown = pymupdf4llm.to_markdown(
        str(pdf_path),
        show_progress=False,
        write_images=False,
        embed_images=False,
        ignore_images=False,
        ignore_graphics=False,
        table_strategy="lines_strict",
        use_ocr=True,
        force_ocr=False,
    )
    if not markdown or not markdown.strip():
        raise RuntimeError("PyMuPDF4LLM produced empty Markdown.")
    return markdown


def conversion_worker(job_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = load_manifest()
    by_app = {item["application_no"]: item for item in manifest if item.get("application_no")}
    total = len(records)
    converted = []
    failed = []
    started_all = time.perf_counter()

    update_job(job_id, total=total, completed=0, failed=0, percentage=0, phase="processing", current="", message="Starting PDF → Markdown conversion")

    for completed_count, record in enumerate(records, start=1):
        app_no = record["application_no"]
        pdf_path = RESUME_DIR / record["stored_pdf"]
        md_path = MD_DIR / f"{app_no}.md"
        update_job(
            job_id,
            current=record.get("source_filename", record.get("stored_pdf", app_no)),
            completed=completed_count - 1,
            percentage=round(((completed_count - 1) / total) * 100) if total else 0,
            phase="processing",
            message=f"Converting {app_no}",
        )
        started = time.perf_counter()
        try:
            markdown = convert_one_pdf_to_markdown(pdf_path)
            md_path.write_text(markdown, encoding="utf-8")
            elapsed = round(time.perf_counter() - started, 3)
            by_app[app_no].update({
                "markdown_status": "success",
                "markdown_characters": len(markdown),
                "markdown_file": md_path.name,
                "markdown_conversion_mode": "pymupdf4llm-hybrid-ocr",
                "markdown_error": "",
                "markdown_conversion_seconds": elapsed,
            })
            converted.append({"application_no": app_no, "markdown_file": md_path.name, "characters": len(markdown), "seconds": elapsed})
        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 3)
            by_app[app_no].update({
                "markdown_status": "failed",
                "markdown_conversion_mode": "pymupdf4llm-hybrid-ocr",
                "markdown_error": str(exc),
                "markdown_conversion_seconds": elapsed,
            })
            failed.append({"application_no": app_no, "error": str(exc), "seconds": elapsed})

        save_manifest(list(by_app.values()))
        update_job(
            job_id,
            completed=completed_count,
            failed=len(failed),
            percentage=round((completed_count / total) * 100) if total else 0,
            phase="processing",
            current=record.get("source_filename", record.get("stored_pdf", app_no)),
            message=f"Converted {completed_count} of {total}",
        )

    elapsed_all = round(time.perf_counter() - started_all, 2)
    return {"converted": converted, "failed": failed, "elapsed_seconds": elapsed_all}


def pending_conversion_records() -> list[dict[str, Any]]:
    manifest = load_manifest()
    records = []
    for record in manifest:
        app_no = record.get("application_no", "")
        md_path = MD_DIR / f"{app_no}.md"
        if record.get("markdown_status") == "success" and md_path.exists() and md_path.stat().st_size > 0:
            continue
        records.append(record)
    return records


def clear_all_hirenex_data() -> dict[str, Any]:
    removed = {"pdf": 0, "markdown": 0, "output_files": 0, "rag_output_files": 0, "manifest": 0, "rag_files": 0}

    def clear_dir(directory: Path, counter_key: str):
        directory.mkdir(parents=True, exist_ok=True)
        removed[counter_key] = sum(1 for x in directory.rglob("*") if x.is_file())
        for child in list(directory.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass

    clear_dir(RESUME_DIR, "pdf")
    clear_dir(MD_DIR, "markdown")
    clear_dir(OUTPUT_DIR, "output_files")
    clear_dir(RAG_OUTPUT_DIR, "rag_output_files")

    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
        removed["manifest"] = 1

    try:
        retrieval_module = importlib.import_module("retrival")
        clear_function = getattr(retrieval_module, "clear_retrieval_index", None)
        if callable(clear_function):
            removed["rag_files"] = int(clear_function() or 0)
    except Exception as exc:
        removed["rag_clear_warning"] = str(exc)

    for child in list(RAG_DIR.iterdir()):
        if child.name == "chroma_db":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass

    for directory in [RESUME_DIR, NLP_OUTPUT_DIR, LLM_OUTPUT_DIR, MD_DIR, RAG_OUTPUT_DIR, RAG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    return removed


def start_conversion(records: list[dict[str, Any]]) -> str:
    job_id = create_job("markdown_conversion", len(records))
    executor.submit(run_job, job_id, conversion_worker, records)
    return job_id


def parser_worker(job_id: str, module_name: str, output_dir: Path) -> Any:
    total = len(list(MD_DIR.glob("APP*.md")))
    update_job(
        job_id,
        total=total,
        completed=0,
        failed=0,
        percentage=0,
        phase="initializing",
        message=f"Preparing {module_name.upper()} parser…",
    )
    module = importlib.import_module(module_name)
    function = getattr(module, "process_resumes", None)
    if not callable(function):
        raise AttributeError(f"{module_name}.py must expose process_resumes(input_dir, output_dir)")

    def progress_callback(**progress: Any) -> None:
        progress_total = int(progress.get("total", total) or 0)
        completed = int(progress.get("completed", 0) or 0)
        updates = dict(progress)
        updates["percentage"] = round((completed / progress_total) * 100) if progress_total else 0
        update_job(job_id, **updates)

    return function(str(MD_DIR), str(output_dir), progress_callback=progress_callback)


def locate_output_file(folder: Path, extension: str) -> Path | None:
    preferred = [f"resumes.{extension}", f"resume.{extension}", f"parsed_resumes.{extension}"]
    for name in preferred:
        path = folder / name
        if path.exists():
            return path
    found = sorted(folder.rglob(f"*.{extension}"))
    return found[0] if found else None


def evaluation_worker(job_id: str, include_retrieval: bool = True) -> dict[str, Any]:
    update_job(job_id, message="Preparing evaluation benchmark...")
    evaluation = importlib.import_module("evaluation")

    def run_search(question: str):
        retrieval = importlib.import_module("retrival")
        search = getattr(retrieval, "search_candidates", None)
        if not callable(search):
            raise AttributeError("retrival.py must expose search_candidates().")
        return search(question)

    update_job(job_id, message="Evaluating conversion and parser quality...")
    result = evaluation.run_full_evaluation(
        run_search=run_search if include_retrieval else None,
        include_retrieval=include_retrieval,
    )
    return result


def conversion_stats() -> dict[str, Any]:
    manifest = load_manifest()
    total = len(manifest)
    converted = sum(1 for item in manifest if item.get("markdown_status") == "success")
    failed = sum(1 for item in manifest if item.get("markdown_status") == "failed")
    pending = max(total - converted - failed, 0)
    state = "empty" if total == 0 else "success" if converted == total else "incomplete" if failed else "pending"
    return {
        "total": total,
        "converted": converted,
        "failed": failed,
        "pending": pending,
        "not_converted": failed + pending,
        "state": state,
        "next_application": next_application_number(manifest),
        "failed_rows": [
            {
                "Application_No": item.get("application_no", ""),
                "Source_File": item.get("source_filename", ""),
                "Mode": item.get("markdown_conversion_mode", ""),
                "Error": item.get("markdown_error", ""),
            }
            for item in manifest if item.get("markdown_status") == "failed"
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "app": APP_TITLE}


@app.get("/api/status")
def status():
    stats = conversion_stats()
    nlp_path = NLP_OUTPUT_DIR / "resumes.json"
    llm_path = LLM_OUTPUT_DIR / "resumes.json"
    return {
        "app": APP_TITLE,
        "tagline": APP_TAGLINE,
        "conversion": stats,
        "nlp_records": count_json_records(nlp_path),
        "llm_records": count_json_records(llm_path),
        "nlp_excel": locate_output_file(NLP_OUTPUT_DIR, "xlsx") is not None,
        "llm_excel": locate_output_file(LLM_OUTPUT_DIR, "xlsx") is not None,
        "evaluation": {
            "benchmark_available": (EVALUATION_DIR / "ground_truth.json").exists(),
            "queries_available": (EVALUATION_DIR / "recruiter_queries.json").exists(),
            "latest_available": EVALUATION_RESULT.exists(),
        },
    }


def count_json_records(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("resumes", "candidates", "records"):
                if isinstance(data.get(key), list):
                    return len(data[key])
    except Exception:
        return 0
    return 0


@app.get("/api/evaluation/summary")
def evaluation_summary():
    if not EVALUATION_RESULT.exists():
        return {"available": False, "result": None}
    try:
        result = json.loads(EVALUATION_RESULT.read_text(encoding="utf-8"))
    except Exception:
        result = None
    return {"available": isinstance(result, dict), "result": result}


@app.post("/api/evaluation/run")
def run_evaluation():
    if not (EVALUATION_DIR / "ground_truth.json").exists():
        raise HTTPException(
            status_code=400,
            detail="Evaluation benchmark not found. Add evaluation/ground_truth.json and recruiter_queries.json.",
        )
    if not (NLP_OUTPUT_DIR / "resumes.json").exists() and not (LLM_OUTPUT_DIR / "resumes.json").exists():
        raise HTTPException(
            status_code=400,
            detail="Run NLP and/or LLM processing before evaluation.",
        )
    job_id = create_job("evaluation")
    executor.submit(run_job, job_id, evaluation_worker, True)
    return {"job_id": job_id}


@app.post("/api/upload")
async def upload_resumes(files: list[UploadFile] = File(...)):
    items: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename:
            continue
        raw = await upload.read()
        items.append((upload.filename, raw))

    payloads, warnings = collect_pdf_payloads(items)
    if not payloads:
        raise HTTPException(status_code=400, detail={"message": "No valid PDF resumes found.", "warnings": warnings})

    ingest_result = ingest_resumes(payloads)
    created = ingest_result["created"]
    current_manifest = load_manifest()

    records_to_convert = created if created else [
        item for item in current_manifest if item.get("markdown_status") != "success"
    ]

    job_id = start_conversion(records_to_convert) if records_to_convert else None

    return {
        "message": "Upload accepted.",
        "created_count": len(created),
        "duplicate_count": len(ingest_result["duplicates"]),
        "duplicates": ingest_result["duplicates"],
        "warnings": warnings,
        "job_id": job_id,
        "conversion": conversion_stats(),
    }


@app.post("/api/data/clear")
def clear_data():
    result = clear_all_hirenex_data()
    return {"message": "All current HireNex dataset data was removed.", "details": result, "conversion": conversion_stats()}


@app.post("/api/process/nlp")
def process_nlp():
    if not list(MD_DIR.glob("APP*.md")):
        raise HTTPException(status_code=400, detail="No Markdown resumes found. Upload and convert resumes first.")
    job_id = create_job("nlp")
    executor.submit(run_job, job_id, parser_worker, "nlp", NLP_OUTPUT_DIR)
    return {"job_id": job_id}


@app.post("/api/process/llm")
def process_llm():
    if not list(MD_DIR.glob("APP*.md")):
        raise HTTPException(status_code=400, detail="No Markdown resumes found. Upload and convert resumes first.")
    job_id = create_job("llm")
    executor.submit(run_job, job_id, parser_worker, "llm", LLM_OUTPUT_DIR)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/download/{parser}/{filetype}")
def download_parser_output(parser: str, filetype: str):
    mapping = {"nlp": NLP_OUTPUT_DIR, "llm": LLM_OUTPUT_DIR}
    if parser not in mapping or filetype not in {"json", "xlsx"}:
        raise HTTPException(status_code=404, detail="Output not found")
    path = locate_output_file(mapping[parser], filetype)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Output not generated yet")
    return FileResponse(path=path, filename=path.name)


@app.get("/api/download/rag/{filename}")
def download_rag_output(filename: str):
    safe = Path(filename).name
    path = RAG_OUTPUT_DIR / safe
    if not path.exists() or path.suffix.lower() != ".xlsx":
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path=path, filename=path.name)


@app.post("/api/search")
def search_candidates(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a recruiter query first.")
    if not (NLP_OUTPUT_DIR / "resumes.json").exists() and not (LLM_OUTPUT_DIR / "resumes.json").exists():
        raise HTTPException(status_code=400, detail="Run NLP and/or LLM processing first.")

    retrieval = importlib.import_module("retrival")
    function = getattr(retrieval, "search_candidates", None)
    if not callable(function):
        raise HTTPException(status_code=500, detail="retrival.py must expose search_candidates().")

    result = function(query)
    result = json_safe(result)
    excel_path = result.get("excel_path") if isinstance(result, dict) else None
    result["excel_url"] = None
    if excel_path:
        result["excel_url"] = f"/api/download/rag/{Path(excel_path).name}"
    result.pop("excel_path", None)
    return result
