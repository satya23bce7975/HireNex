"""
HireNex - Traditional NLP Resume Parser

Responsibilities of this module:
    - Read ONLY Markdown resumes from md_data/
    - Use traditional NLP / rule-based techniques only
    - Extract structured resume information
    - Save results as JSON and Excel

No LLM/API calls are used in this file.
The Streamlit interface is intentionally kept in app.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import spacy


# -----------------------------------------------------------------------------
# NLP MODEL
# -----------------------------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Keep the module usable even if the small English model is not installed.
    # The sentencizer gives the blank pipeline sentence boundaries for summary fallback.
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

SKILL_DB = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql",
    "react", "react.js", "node.js", "express.js", "mongodb", "mysql",
    "postgresql", "oracle", "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp", "linux", "windows", "html", "css",
    "tailwind", "bootstrap", "flask", "django", "fastapi", "spring",
    "spring boot", "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "generative ai", "genai",
    "agentic ai", "tensorflow", "keras", "pytorch", "scikit-learn",
    "scikit learn", "numpy", "pandas", "matplotlib", "seaborn", "folium",
    "power bi", "tableau", "excel", "data science", "data analysis",
    "computer vision", "cnn", "u-net", "transformers", "bert", "rag",
    "chromadb", "langchain", "openai", "gemini", "groq", "streamlit",
    "vite", "npm", "rest api", "restful api", "postman", "java",
    "data structures", "data structures and algorithms", "dsa", "oops",
    "object oriented programming", "dbms", "gitlab", "jenkins",
]

SECTION_ALIASES = {
    "summary": {
        "professional summary", "profile", "summary", "objective", "career objective"
    },
    "education": {"education", "academic background", "academics", "educational qualification"},
    "skills": {"technical skills", "skills", "technical expertise", "core skills", "skill set"},
    "experience": {"experience", "work experience", "professional experience", "employment history", "internships"},
    "projects": {"projects", "academic projects", "personal projects", "key projects", "project experience"},
    "certifications": {"certifications", "certificates", "licenses & certifications", "licenses and certifications"},
    "achievements": {"achievements", "awards", "honors", "accomplishments"},
}

ALL_SECTION_NAMES = {name for group in SECTION_ALIASES.values() for name in group}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,12}(?!\d)")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s)]+", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[^\s)]+", re.I)
PORTFOLIO_RE = re.compile(r"(?:https?://)(?!github\.com|linkedin\.com)[^\s)]+", re.I)
YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current)\b", re.I)
MONTH_YEAR_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:19|20)\d{2}\b",
    re.I,
)


# -----------------------------------------------------------------------------
# GENERAL HELPERS
# -----------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip(" \t|•")


def normalize_heading(line: str) -> str:
    line = re.sub(r"^\s*#{1,6}\s*", "", line)
    line = re.sub(r"[*_`]+", "", line)
    line = re.sub(r"\s+", " ", line)
    return clean_text(line).rstrip(":").lower()


def canonical_section(line: str) -> str | None:
    heading = normalize_heading(line)
    if not heading or len(heading) > 80:
        return None
    for canonical, aliases in SECTION_ALIASES.items():
        if heading in aliases:
            return canonical
    return None


def lines_from_markdown(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-*+]\s+", "• ", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        lines.append(line)
    return lines


def build_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {key: [] for key in SECTION_ALIASES}
    current: str | None = None

    for line in lines_from_markdown(text):
        detected = canonical_section(line)
        if detected:
            current = detected
            continue
        if current:
            sections[current].append(line)

    return sections


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = clean_text(value)
        if not value:
            continue
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def application_no_from_path(path: Path) -> str:
    match = re.fullmatch(r"APP\d{6}", path.stem, flags=re.I)
    return path.stem.upper() if match else path.stem


# -----------------------------------------------------------------------------
# CONTACT / NAME
# -----------------------------------------------------------------------------

def extract_contact(text: str) -> dict[str, str]:
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)
    linkedin_match = LINKEDIN_RE.search(text)
    github_match = GITHUB_RE.search(text)

    portfolio = ""
    for match in PORTFOLIO_RE.findall(text):
        if "linkedin.com" not in match.lower() and "github.com" not in match.lower():
            portfolio = match.rstrip(".,;")
            break

    return {
        "email": email_match.group(0) if email_match else "",
        "phone": clean_text(phone_match.group(0)) if phone_match else "",
        "linkedin": linkedin_match.group(0).rstrip(".,;") if linkedin_match else "",
        "github": github_match.group(0).rstrip(".,;") if github_match else "",
        "portfolio": portfolio,
    }


def extract_name(text: str) -> str:
    # Prefer the first substantial line before the first known section.
    for line in lines_from_markdown(text)[:15]:
        candidate = re.sub(r"^#+\s*", "", line)
        candidate = EMAIL_RE.sub("", candidate)
        candidate = LINKEDIN_RE.sub("", candidate)
        candidate = GITHUB_RE.sub("", candidate)
        candidate = re.sub(r"https?://\S+", "", candidate)
        candidate = re.sub(r"[|•]+", " ", candidate)
        candidate = clean_text(candidate)

        if not candidate:
            continue
        if canonical_section(candidate):
            break
        if "+" in candidate and len(candidate.split()) <= 3:
            continue
        if "@" in candidate:
            continue
        if len(candidate.split()) > 7:
            continue
        if re.search(r"\b(B\.Tech|BTech|M\.Tech|resume|curriculum vitae|linkedin|github)\b", candidate, re.I):
            continue
        if re.search(r"\d", candidate):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", candidate)
        if 2 <= len(words) <= 5:
            return " ".join(words)

    return ""


# -----------------------------------------------------------------------------
# SKILLS
# -----------------------------------------------------------------------------

def extract_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found: list[str] = []

    # Longer phrases first to avoid premature partial matches.
    for skill in sorted(SKILL_DB, key=len, reverse=True):
        pattern = re.escape(skill)
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text_lower):
            found.append(skill)

    # Prefer resume spelling for common skills while preserving uniqueness.
    display = []
    mapping = {
        "python": "Python", "java": "Java", "c++": "C++", "c#": "C#",
        "javascript": "JavaScript", "typescript": "TypeScript", "sql": "SQL",
        "react": "React", "react.js": "React.js", "node.js": "Node.js",
        "mongodb": "MongoDB", "mysql": "MySQL", "postgresql": "PostgreSQL",
        "aws": "AWS", "azure": "Azure", "gcp": "GCP", "git": "Git", "github": "GitHub",
        "docker": "Docker", "kubernetes": "Kubernetes", "machine learning": "Machine Learning",
        "deep learning": "Deep Learning", "artificial intelligence": "Artificial Intelligence",
        "natural language processing": "Natural Language Processing", "nlp": "NLP",
        "generative ai": "Generative AI", "genai": "GenAI", "agentic ai": "Agentic AI",
        "tensorflow": "TensorFlow", "keras": "Keras", "pytorch": "PyTorch",
        "scikit-learn": "Scikit-learn", "scikit learn": "Scikit-learn", "numpy": "NumPy",
        "pandas": "Pandas", "matplotlib": "Matplotlib", "seaborn": "Seaborn",
        "folium": "Folium", "power bi": "Power BI", "tableau": "Tableau", "excel": "Excel",
        "computer vision": "Computer Vision", "cnn": "CNN", "u-net": "U-Net", "transformers": "Transformers",
        "bert": "BERT", "rag": "RAG", "chromadb": "ChromaDB", "langchain": "LangChain",
        "streamlit": "Streamlit", "vite": "Vite", "npm": "npm", "rest api": "REST API",
        "postman": "Postman", "data structures and algorithms": "Data Structures and Algorithms",
        "dsa": "DSA", "oops": "OOPs", "dbms": "DBMS",
    }
    for skill in found:
        display.append(mapping.get(skill, skill.title()).rstrip(".,;:"))

    return dedupe(display)


# -----------------------------------------------------------------------------
# EDUCATION
# -----------------------------------------------------------------------------

def parse_score(text: str) -> dict[str, str]:
    patterns = [
        (r"\bCGPA\s*[:=-]?\s*(\d+(?:\.\d+)?)", "CGPA"),
        (r"\b(?:percentage|percent|percentage score)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*%?", "Percentage"),
        (r"\b(\d+(?:\.\d+)?)\s*%", "Percentage"),
    ]
    for pattern, score_type in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return {"value": match.group(1), "type": score_type}
    return {"value": "", "type": ""}


def extract_years(text: str) -> tuple[str, str]:
    match = YEAR_RANGE_RE.search(text)
    if match:
        return match.group(1), match.group(2)

    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    if len(years) >= 2:
        return years[0], years[1]
    if len(years) == 1:
        return years[0], ""
    return "", ""


def extract_education(text: str, sections: dict[str, list[str]]) -> list[dict[str, Any]]:
    lines = sections.get("education", [])
    if not lines:
        # Fallback over full document for resumes where headings are missing.
        lines = lines_from_markdown(text)

    records: list[dict[str, Any]] = []
    degree_terms = (
        "b.tech", "btech", "b.e", "bachelor", "m.tech", "mtech", "m.e", "master",
        "b.sc", "bsc", "m.sc", "msc", "phd", "diploma", "degree", "intermediate", "12th", "10th",
    )

    for index, line in enumerate(lines):
        lower = line.lower()
        if not any(term in lower for term in degree_terms):
            continue

        degree = line
        field = ""
        score = parse_score(line)
        start_year, end_year = extract_years(line)

        # Combine the next one/two lines when they likely contain institute/year info.
        context = " ".join(lines[index:index + 3])
        if index + 1 < len(lines):
            context += " " + lines[index + 1]
        if index + 2 < len(lines):
            context += " " + lines[index + 2]

        start2, end2 = extract_years(context)
        start_year = start_year or start2
        end_year = end_year or end2
        score_line = parse_score(context)
        if not score["value"]:
            score = score_line

        # Extract the degree field from common forms: B.Tech CSE (AI and ML), etc.
        degree_match = re.search(
            r"((?:B\.?Tech|BTech|M\.?Tech|MTech|B\.?E|M\.?E|B\.?Sc|M\.?Sc|Bachelor|Master|PhD|Diploma)[^\n|]*)",
            line,
            re.I,
        )
        if degree_match:
            degree = clean_text(degree_match.group(1))

        paren = re.search(r"\(([^)]+)\)", degree)
        if paren:
            field = clean_text(paren.group(1))

        if not field:
            field_match = re.search(
                r"\b(?:in|of)\s+([A-Za-z][A-Za-z &/.-]{2,60})",
                degree,
                re.I,
            )
            if field_match:
                field = clean_text(field_match.group(1))

        # Institute is usually on the next line in Markdown produced from a resume.
        institute = ""
        for next_line in lines[index + 1:index + 4]:
            candidate = clean_text(YEAR_RANGE_RE.sub("", next_line))
            candidate = clean_text(re.sub(r"\b(?:CGPA|Percentage|Percent)\b.*", "", candidate, flags=re.I))
            if not candidate:
                continue
            if any(term in candidate.lower() for term in degree_terms):
                continue
            if re.search(r"\b(board of|school|college|university|institute|academy|institute of)\b", candidate, re.I):
                institute = candidate
                break

        records.append({
            "degree": degree,
            "field": field,
            "score": score,
            "institute": institute,
            "start_year": start_year,
            "end_year": end_year,
        })

    return records


# -----------------------------------------------------------------------------
# PROJECTS / EXPERIENCE / CERTIFICATIONS
# -----------------------------------------------------------------------------

def looks_like_project_title(line: str) -> bool:
    line = clean_text(line).lstrip("• ")
    if len(line) < 5 or len(line) > 140:
        return False
    if re.match(r"^(built|developed|designed|created|implemented|performed|used|applied|achieved|worked|preprocessed|tuned|responsible)\b", line, re.I):
        return False
    if line.endswith((".", ":")):
        return False
    # Project title patterns commonly contain separators, technologies, or title-case words.
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#./&-]*", line)
    if len(words) < 2:
        return False
    return True


def extract_projects(sections: dict[str, list[str]]) -> list[dict[str, Any]]:
    lines = sections.get("projects", [])
    if not lines:
        return []

    projects: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        clean = clean_text(line)
        if not clean:
            continue

        title_part = clean
        date_text = ""
        if "|" in clean:
            title_part, date_text = [clean_text(x) for x in clean.split("|", 1)]
        elif re.search(r"\b(?:19|20)\d{2}\b", clean) and not clean.startswith("•"):
            date_match = re.search(r"\b(?:19|20)\d{2}\b", clean)
            if date_match and len(clean[:date_match.start()].strip()) >= 5:
                title_part = clean[:date_match.start()].strip(" -–—")
                date_text = clean[date_match.start():]

        if not clean.startswith("•") and looks_like_project_title(title_part):
            current = {
                "name": title_part,
                "description": "",
                "technologies": [],
                "github": "",
                "year": date_text,
            }
            projects.append(current)
            continue

        if current is None:
            continue

        bullet = clean.lstrip("• ")
        if re.search(r"tools?\s*(?:and|&)\s*technologies|technologies\s*[:=-]", bullet, re.I):
            tech_text = re.split(r"tools?\s*(?:and|&)\s*technologies\s*[:=-]?|technologies\s*[:=-]?", bullet, maxsplit=1, flags=re.I)[-1]
            current["technologies"] = dedupe([clean_text(x).rstrip(".,;:") for x in re.split(r",|\||;", tech_text)])
        elif "github.com/" in bullet.lower():
            current["github"] = bullet
            current["description"] = clean_text((current["description"] + " " + re.sub(r"https?://\S+", "", bullet)).strip())
        else:
            current["description"] = clean_text((current["description"] + " " + bullet).strip())

    return projects


def extract_experience(sections: dict[str, list[str]]) -> list[dict[str, Any]]:
    lines = sections.get("experience", [])
    if not lines:
        return []

    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in lines:
        clean = clean_text(line)
        if not clean:
            continue

        date_match = YEAR_RANGE_RE.search(clean) or MONTH_YEAR_RE.search(clean)
        role_match = re.search(r"\b(intern|engineer|developer|analyst|manager|scientist|designer|consultant|associate|lead|trainee|architect)\b", clean, re.I)

        new_entry = (
            not clean.startswith("•")
            and (date_match is not None or (role_match is not None and len(clean.split()) <= 18))
            and not clean.lower().startswith(("responsibilities", "description"))
        )

        if new_entry:
            current = {
                "company": "",
                "role": clean,
                "description": "",
                "start_year": "",
                "end_year": "",
            }
            start, end = extract_years(clean)
            current["start_year"] = start
            current["end_year"] = end

            # Common "Role | Company | Date" or "Role - Company" layouts.
            parts = [clean_text(x) for x in re.split(r"\||\s+[-–—]\s+", clean) if clean_text(x)]
            if len(parts) >= 2:
                current["role"] = parts[0]
                current["company"] = parts[1]
            entries.append(current)
            continue

        if current is not None:
            bullet = clean.lstrip("• ")
            current["description"] = clean_text((current["description"] + " " + bullet).strip())

    return entries


def extract_certifications(sections: dict[str, list[str]]) -> list[dict[str, Any]]:
    lines = sections.get("certifications", [])
    records: list[dict[str, Any]] = []
    for line in lines:
        clean = clean_text(line).lstrip("• ")
        if not clean:
            continue
        years = re.findall(r"\b(?:19|20)\d{2}\b", clean)
        records.append({
            "name": clean,
            "issuer": "",
            "year": years[-1] if years else "",
            "credential_url": "",
        })
    return records


def extract_achievements(sections: dict[str, list[str]]) -> list[str]:
    return dedupe([
        clean_text(line).lstrip("• ")
        for line in sections.get("achievements", [])
        if clean_text(line).lstrip("• ")
    ])


def extract_personal_summary(text: str, sections: dict[str, list[str]]) -> str:
    lines = sections.get("summary", [])
    if lines:
        return clean_text(" ".join(lines))

    # Small NER-based fallback: sentence beginning near the top before first section.
    doc = nlp(text[:4000])
    sentences = [clean_text(sent.text) for sent in doc.sents if clean_text(sent.text)]
    for sentence in sentences:
        if 10 <= len(sentence.split()) <= 60 and not EMAIL_RE.search(sentence):
            return sentence
    return ""


# -----------------------------------------------------------------------------
# MAIN PARSER
# -----------------------------------------------------------------------------

def parse_markdown_resume(md_path: Path) -> dict[str, Any]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    sections = build_sections(text)
    contact = extract_contact(text)

    result = {
        "application_no": application_no_from_path(md_path),
        "candidate": {
            "name": extract_name(text),
            "tag": "Traditional NLP",
            "contact": contact,
            "personal_summary": extract_personal_summary(text, sections),
        },
        "education": extract_education(text, sections),
        "skills": extract_skills(text),
        "projects": extract_projects(sections),
        "experience": extract_experience(sections),
        "certifications": extract_certifications(sections),
        "achievements": extract_achievements(sections),
    }

    return result


# -----------------------------------------------------------------------------
# JSON / EXCEL OUTPUT
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


def process_resumes(
    input_dir: str = "md_data",
    output_dir: str = "output/nlp",
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Parse all APP*.md files and save NLP JSON + Excel.

    ``progress_callback`` is optional so CLI and older callers remain compatible.
    It receives per-resume progress updates for the FastAPI UI ticker.
    """
    md_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not md_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")

    md_files = sorted(md_dir.glob("APP*.md"))
    if not md_files:
        raise FileNotFoundError(f"No APP*.md files found in {md_dir}")

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

    report(0, "", "Preparing Traditional NLP parser…", "initializing")
    for index, md_path in enumerate(md_files, start=1):
        app_no = application_no_from_path(md_path)
        report(index - 1, md_path.name, f"Extracting {app_no} with Traditional NLP…")
        try:
            records.append(parse_markdown_resume(md_path))
        except Exception as exc:
            errors.append({
                "application_no": app_no,
                "error": str(exc),
            })
        report(index, md_path.name, f"NLP processed {index} of {total}")

    report(total, "", "Saving NLP JSON and Excel outputs…", "saving")
    output = save_outputs(records, out_dir)
    output["errors"] = errors
    return output


if __name__ == "__main__":
    result = process_resumes()
    print(json.dumps(result, indent=2, ensure_ascii=False))
