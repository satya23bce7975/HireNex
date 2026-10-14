# HireNex — React + FastAPI

HireNex is an intelligent candidate screening and retrieval platform for resume ingestion, traditional NLP parsing, LLM extraction, hybrid RAG search, and measurable evaluation.

## Version 4 updates

- Interactive light-tech visual system with responsive hover states, animated ambient grid, orbit visuals, shimmer accents, and reduced-motion support.
- A shared live progress ticker for PDF → Markdown conversion, Traditional NLP parsing, Gemini LLM parsing, and benchmark jobs.
- Per-resume progress callbacks show the current application/file, phase, completed count, percentage, and failed count during parsing.
- LLM progress updates while a resume is being sent to Gemini and when each resume finishes; progress is not deferred until the entire batch ends.
- Completion and failure states are shown explicitly instead of displaying a spinner after processing finishes.
- This version is packaged separately from v3. It does not include a real `.env` file or API keys.

## Architecture

```text
React + Vite
      ↓
FastAPI
      ↓
┌───────────────┬──────────────┬──────────────┐
│ PDF → Markdown│ Traditional │ LLM Parsing  │
│ PyMuPDF4LLM   │ NLP + spaCy  │ Gemini       │
└───────────────┴──────────────┴──────────────┘
                       ↓
                 Hybrid RAG Search
                    ChromaDB
                       ↓
              Recruiter results + Excel
                       ↓
                 Evaluation Center
```

The Streamlit interface is removed from this version. Your existing Python intelligence modules remain reusable:

- `nlp.py` — traditional NLP only
- `llm.py` — Gemini structured extraction only
- `retrival.py` — retrieval/RAG only
- `evaluation.py` — benchmark metrics

## UI

The React UI includes:

- Colorful tech-oriented dashboard
- Drag-and-drop PDF/ZIP/folder upload
- Live PDF → Markdown, Traditional NLP, and LLM parsing progress tickers
- Per-resume progress messages, current file, completion count, percentage, and failure count
- Animated pipeline visualization and interactive hover states
- NLP and LLM processing cards with explicit success/failure states
- Natural-language recruiter search
- Candidate result table + Excel download
- Dedicated Evaluation Center
- Parser comparison charts
- Retrieval ranking metrics
- Runtime/System health dashboard

## Evaluation metrics shown

### PDF → Markdown
- Conversion success rate
- Average seconds per resume
- Throughput

### NLP / LLM extraction
- Record coverage
- Scalar field accuracy
- Skills precision / recall / F1
- Certification precision / recall / F1
- Experience accuracy
- CGPA MAE
- Overall extraction F1

### Retrieval / RAG
- Precision@10
- Recall@10
- MRR
- nDCG@10
- Average query latency
- Per-query benchmark results

The evaluation uses `evaluation/ground_truth.json` and `evaluation/recruiter_queries.json`. The included benchmark is a synthetic 90-resume CSE/ECE/MECH dataset for academic evaluation; replace it with a reviewed ground-truth set for final reporting.

## Backend

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --app-dir .
```

Backend runs at:

`http://127.0.0.1:8000`

## Frontend

Open a **second terminal** while the backend terminal stays running:

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at:

`http://localhost:5173`

## First test flow

1. Start FastAPI.
2. Start Vite.
3. Upload the 90 benchmark PDFs.
4. Wait for Markdown conversion.
5. Run Traditional NLP.
6. Run LLM Parsing.
7. Test recruiter queries.
8. Open **Evaluation** and click **Run full evaluation**.

Do not expose API keys in source files or the frontend.


## Version 4 progress behavior

When a batch parser starts, the Workspace shows a live ticker directly under its NLP or LLM action button. The LLM ticker updates before each Gemini request and again after each resume completes, so users can see the current application and progress while a batch is still running. These updates use FastAPI's existing `/api/jobs/{job_id}` polling, matching the PDF → Markdown conversion tracker.
