"""Deterministic, rule-based eligibility evaluation (no LLM involved)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from . import config


@dataclass
class StudentProfile:
    name: str
    branch: str
    degree: str
    cgpa: float
    tenth_pct: float
    twelfth_pct: float
    active_backlogs: int
    graduation_year: int
    has_offer: bool = False
    offer_ctc_lpa: float = 0.0


@dataclass
class RuleCheck:
    rule: str
    passed: bool
    detail: str


@dataclass
class EligibilityResult:
    company: str
    role: str
    eligible: bool
    checks: list[RuleCheck] = field(default_factory=list)


def load_companies() -> list[dict]:
    if not config.COMPANIES_FILE.exists():
        return []
    return json.loads(config.COMPANIES_FILE.read_text(encoding="utf-8"))


def evaluate(student: StudentProfile, company: dict, policy: dict | None = None,
             today: date | None = None) -> EligibilityResult:
    c = company.get("criteria", {})
    policy = policy or {}
    today = today or date.today()
    checks: list[RuleCheck] = []

    def add(rule, passed, detail):
        checks.append(RuleCheck(rule, bool(passed), detail))

    if "degrees" in c:
        add("Degree", student.degree in c["degrees"], f"{student.degree} vs allowed {c['degrees']}")
    if "branches" in c:
        add("Branch", student.branch in c["branches"], f"{student.branch} vs allowed {c['branches']}")
    if "min_cgpa" in c:
        add("CGPA", student.cgpa >= c["min_cgpa"], f"{student.cgpa} ≥ {c['min_cgpa']}")
    if "min_10th" in c:
        add("10th %", student.tenth_pct >= c["min_10th"], f"{student.tenth_pct} ≥ {c['min_10th']}")
    if "min_12th" in c:
        add("12th %", student.twelfth_pct >= c["min_12th"], f"{student.twelfth_pct} ≥ {c['min_12th']}")
    if "max_active_backlogs" in c:
        add("Active backlogs", student.active_backlogs <= c["max_active_backlogs"],
            f"{student.active_backlogs} ≤ {c['max_active_backlogs']}")
    if "graduation_years" in c:
        add("Batch", student.graduation_year in c["graduation_years"],
            f"{student.graduation_year} in {c['graduation_years']}")

    if company.get("deadline"):
        dl = date.fromisoformat(company["deadline"])
        add("Registration open", today <= dl, f"deadline {dl.isoformat()}")

    # Institute policy: placed students may only apply to "dream" offers above a CTC multiple
    if student.has_offer:
        mult = policy.get("dream_offer_multiplier", 1.5)
        need = student.offer_ctc_lpa * mult
        ctc = company.get("ctc_lpa", 0)
        add("Placement policy (dream offer)", ctc >= need,
            f"already placed at {student.offer_ctc_lpa} LPA; need ≥ {need:.1f} LPA, offer is {ctc} LPA")

    return EligibilityResult(company["company"], company.get("role", ""),
                             all(ch.passed for ch in checks), checks)


def evaluate_all(student: StudentProfile, policy: dict | None = None,
                 today: date | None = None) -> list[EligibilityResult]:
    return [evaluate(student, c, policy, today) for c in load_companies()]
