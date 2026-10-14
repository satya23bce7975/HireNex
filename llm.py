"""
HireNex - LLM Resume Parser

Responsibilities of this module:
    - Read ONLY Markdown resumes from md_data/
    - Use Gemini for structured resume extraction
    - Save results as JSON and Excel

No Streamlit UI is defined here. app.py owns the interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


load_dotenv()


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMA
# -----------------------------------------------------------------------------

class Contact(BaseModel):
    email: str = Field(default="")
    phone: str = Field(default="")
    linkedin: str = Field(default="")
    github: str = Field(default="")
    portfolio: str = Field(default="")


class Score(BaseModel):
    value: str = Field(default="")
    type: str = Field(default="")


class Education(BaseModel):
    degree: str = Field(default="")
    field: str = Field(default="")
    score: Score = Field(default_factory=Score)
    institute: str = Field(default="")
    start_year: str = Field(default="")
    end_year: str = Field(default="")


class Project(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    technologies: List[str] = Field(default_factory=list)
    github: str = Field(default="")
    year: str = Field(default="")


class Experience(BaseModel):
    title: str = Field(
        default="",
        description="Professional role/job title held"
    )
    Company: str = Field(
        default="",
        description="Name of the employer or organization"
    )
    duration: str = Field(
        default="",
        description="Employment period exactly as supported by the resume"
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key responsibilities supported by the resume"
    )


class Certification(BaseModel):
    name: str = Field(default="")
    issuer: str = Field(default="")
    year: str = Field(default="")
    credential_url: str = Field(default="")


class ResumeData(BaseModel):
    name: str = Field(
        default="",
        description="Full candidate name"
    )
    email: str = Field(
        default="",
        description="Primary email address"
    )
    phone: str = Field(default="")
    linkedin: str = Field(default="")
    github: str = Field(default="")
    portfolio: str = Field(default="")
    tag: str = Field(default="LLM")
    personal_summary: str = Field(default="")
    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )
    return key


def application_no_from_path(path: Path) -> str:
    return path.stem.upper()


def clean_result(record: ResumeData) -> dict[str, Any]:
    """Convert Pydantic output into the common HireNex JSON structure."""
    data = record.model_dump()

    experience = []
    for item in data.get("experience", []):
        experience.append({
            "company": item.get("Company", ""),
            "role": item.get("title", ""),
            "description": " ".join(item.get("responsibilities", [])),
            "start_year": "",
            "end_year": "",
            "duration": item.get("duration", ""),
        })

    return {
        "candidate": {
            "name": data.get("name", ""),
            "tag": data.get("tag", "LLM"),
            "contact": {
                "email": data.get("email", ""),
                "phone": data.get("phone", ""),
                "linkedin": data.get("linkedin", ""),
                "github": data.get("github", ""),
                "portfolio": data.get("portfolio", ""),
            },
            "personal_summary": data.get("personal_summary", ""),
        },
        "education": [item for item in data.get("education", [])],
        "skills": data.get("skills", []),
        "projects": [item for item in data.get("projects", [])],
        "experience": experience,
        "certifications": [item for item in data.get("certifications", [])],
        "achievements": data.get("achievements", []),
    }


def extract_resume_data(resume_text: str, api_key: str | None = None) -> dict[str, Any]:
    """
    Extract structured data from one Markdown resume using Gemini.

    The application number is deliberately NOT part of the Gemini schema.
    app.py assigns the unique application number, so the LLM cannot alter it.
    """
    key = api_key or get_api_key()
    client = genai.Client(api_key=key)

    prompt = """
You are the HireNex resume extraction engine.

Parse the supplied Markdown resume into the provided JSON schema.

Rules:
1. Extract only information supported by the resume.
2. Never invent a value.
3. When a scalar value is missing, return an empty string.
4. When a list has no data, return an empty list.
5. Preserve names, organizations, skills, project titles, dates and links faithfully.
6. Extract ALL clearly supported skills, education entries, projects, experience entries,
   certifications and achievements rather than returning only a few examples.
7. For experience, preserve the original duration text where present.
8. Do not infer years or responsibilities that the resume does not support.
9. Keep project descriptions complete and do not split one project into multiple projects merely
   because it contains several bullet points.
10. Return only data that belongs to this resume.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, resume_text],
        config={
            "response_mime_type": "application/json",
            "response_schema": ResumeData,
            "temperature": 0.1,
        },
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    try:
        parsed = ResumeData.model_validate_json(response.text)
    except Exception as exc:
        raise ValueError(
            f"Gemini returned invalid structured output: {exc}"
        ) from exc

    return clean_result(parsed)


# -----------------------------------------------------------------------------
# EXCEL OUTPUT
# -----------------------------------------------------------------------------


def flatten_for_excel(record: dict[str, Any]) -> dict[str, Any]:
    candidate = record.get("candidate", {})
    contact = candidate.get("contact", {})

    def as_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value or "")

    return {
        "Application_No": record.get("application_no", ""),
        "Name": candidate.get("name", ""),
        "Tag": candidate.get("tag", ""),
        "Email": contact.get("email", ""),
        "Phone": contact.get("phone", ""),
        "LinkedIn": contact.get("linkedin", ""),
        "GitHub": contact.get("github", ""),
        "Portfolio": contact.get("portfolio", ""),
        "Personal_Summary": candidate.get("personal_summary", ""),
        "Education": as_json(record.get("education", [])),
        "Skills": ", ".join(record.get("skills", [])),
        "Projects": as_json(record.get("projects", [])),
        "Experience": as_json(record.get("experience", [])),
        "Certifications": as_json(record.get("certifications", [])),
        "Achievements": as_json(record.get("achievements", [])),
    }


def save_outputs(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "resumes.json"
    excel_path = output_dir / "resumes.xlsx"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame([flatten_for_excel(record) for record in records])
    df.to_excel(excel_path, index=False)

    return {
        "records": len(records),
        "json_path": str(json_path),
        "excel_path": str(excel_path),
    }


# -----------------------------------------------------------------------------
# BATCH PROCESSOR USED BY app.py
# -----------------------------------------------------------------------------


def process_resumes(
    input_dir: str = "md_data",
    output_dir: str = "output/llm",
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Parse all APP*.md files and save LLM JSON + Excel.

    ``progress_callback`` reports which resume is currently being sent to Gemini,
    plus completed/failed counts, so the UI keeps updating during long API calls.
    It is optional to preserve compatibility with CLI and existing callers.
    """
    md_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")

    md_files = sorted(md_dir.glob("APP*.md"))
    if not md_files:
        raise FileNotFoundError(f"No APP*.md files found in {md_dir}")

    api_key = get_api_key()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total = len(md_files)

    def report(completed: int, current: str, message: str, phase: str = "processing") -> None:
        if progress_callback:
            progress_callback(
                total=total,
                completed=completed,
                failed=len(errors),
                current=current,
                message=message,
                phase=phase,
            )

    report(0, "", "Preparing Gemini LLM parser…", "initializing")
    for index, md_path in enumerate(md_files, start=1):
        app_no = application_no_from_path(md_path)
        report(index - 1, md_path.name, f"Sending {app_no} to Gemini for structured parsing…")
        try:
            resume_text = md_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            parsed = extract_resume_data(
                resume_text,
                api_key
            )

            parsed["application_no"] = app_no
            parsed["metadata"] = {
                "source_markdown": md_path.name,
                "processing_status": "success",
                "parser": "Gemini LLM",
                "model": MODEL_NAME,
            }

            records.append(parsed)

        except Exception as exc:
            errors.append({
                "application_no": app_no,
                "error": str(exc),
            })
        report(index, md_path.name, f"LLM parsed {index} of {total}")

    report(total, "", "Saving LLM JSON and Excel outputs…", "saving")
    output = save_outputs(records, out_dir)
    output["errors"] = errors
    return output


if __name__ == "__main__":
    result = process_resumes()
    print(json.dumps(result, indent=2, ensure_ascii=False))
