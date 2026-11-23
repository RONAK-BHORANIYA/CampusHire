"""Input / output guardrails."""
from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now", r"act as (an? )?(unfiltered|jailbroken|dan)",
    r"reveal (your|the) (system )?prompt", r"developer mode",
]
PII_PATTERNS = {
    "phone": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
}
DOMAIN_TERMS = [
    "placement", "company", "companies", "drive", "jd", "job", "role", "intern", "ctc", "package",
    "stipend", "eligib", "cgpa", "backlog", "deadline", "register", "policy", "offer", "interview",
    "test", "branch", "batch", "notice", "apply", "salary", "lpa", "criteria", "dream", "tnp", "t&p",
    "recruit", "hiring", "location", "bond", "skills",
]
MAX_QUERY_CHARS = 1000


@dataclass
class GuardResult:
    ok: bool
    text: str
    reason: str = ""


def check_input(query: str, known_companies: set[str] | None = None) -> GuardResult:
    q = query.strip()
    if not q:
        return GuardResult(False, q, "Empty question.")
    if len(q) > MAX_QUERY_CHARS:
        return GuardResult(False, q, f"Question too long (max {MAX_QUERY_CHARS} characters).")
    low = q.lower()
    if any(re.search(p, low) for p in INJECTION_PATTERNS):
        return GuardResult(False, q, "This looks like a prompt-injection attempt and was blocked.")
    companies = {c.lower() for c in (known_companies or set())}
    if not any(t in low for t in DOMAIN_TERMS) and not any(c in low for c in companies):
        return GuardResult(False, q, "I can only answer questions about placements, companies, "
                                     "eligibility, policies and deadlines.")
    redacted = q
    for name, pat in PII_PATTERNS.items():
        redacted = re.sub(pat, f"[{name.upper()} REDACTED]", redacted)
    return GuardResult(True, redacted)


def check_output(answer: str, num_sources: int) -> GuardResult:
    """Ensure every citation points at a real source and the answer is actually cited."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    invalid = sorted(n for n in cited if n < 1 or n > num_sources)
    if invalid:
        answer = re.sub(r"\[(\d+)\]",
                        lambda m: "" if int(m.group(1)) in invalid else m.group(0), answer)
    refusal = "don't have enough information" in answer.lower()
    if not cited and not refusal:
        return GuardResult(False, answer, "Answer had no citations; flagged as possibly ungrounded.")
    for name, pat in PII_PATTERNS.items():
        answer = re.sub(pat, f"[{name.upper()} REDACTED]", answer)
    return GuardResult(True, answer, "Removed invalid citations." if invalid else "")
