"""Plain-language names, explanations and code tables for EVERY field in the system.

The raw datasets use terse names ("Curricular units 2nd sem (approved)", "Daytime/evening attendance").
Everything the student/teacher sees - dashboard forms, charts, reports - goes through this file, so a
field is described in exactly one place.

Context (be upfront about it): the university data are real students of a Portuguese polytechnic
(Realinho et al., 2022). The fields are generic to any college: course, age, fees, scholarship and
semester results. Grades are on a 0-20 scale where 10 is the pass mark (about 50 % in Indian terms).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ code tables
COURSES = {
    33: "Biofuel Production Technologies", 171: "Animation & Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising & Marketing Management",
    9773: "Journalism & Communication", 9853: "Basic Education", 9991: "Management (evening)",
}
ADMISSION_ROUTES = {
    1: "General admission - 1st round", 17: "General admission - 2nd round", 18: "General admission - 3rd round",
    39: "Mature student (over 23)", 42: "Transfer from another institution", 43: "Change of course",
    51: "Change of institution / course", 57: "Change of institution / course (international)",
    7: "Already holds another higher-education course", 44: "Technological specialisation diploma holder",
    53: "Short-cycle diploma holder", 15: "International student",
    2: "Special-regime admission (ordinance 612/93)", 10: "Special-regime admission (ordinance 854-B/99)",
    26: "Special-regime admission (ordinance 533-A/99, new plan)",
    27: "Special-regime admission (ordinance 533-A/99, other institution)",
    5: "Special quota - Azores islands", 16: "Special quota - Madeira islands",
}
PREVIOUS_QUALIFICATION = {
    1: "Secondary school (12th year)", 2: "Bachelor's degree", 3: "Degree", 4: "Master's degree",
    5: "Doctorate", 6: "Started a higher-education course", 9: "12th year, not completed",
    10: "11th year, not completed", 12: "Other 11th-year qualification", 14: "10th year",
    15: "10th year, not completed", 19: "Basic education, 3rd cycle (9th-11th year)",
    38: "Basic education, 2nd cycle (6th-8th year)", 39: "Technological specialisation course",
    40: "Higher-education degree (1st cycle)", 42: "Professional higher technical course",
    43: "Master's (2nd cycle)",
}
YES_NO = {0: "No", 1: "Yes"}
CODE_MAPS = {
    "Course": COURSES,
    "Application mode": ADMISSION_ROUTES,
    "Previous qualification": PREVIOUS_QUALIFICATION,
    "Daytime/evening attendance": {0: "Evening class", 1: "Day class"},
    "Displaced": YES_NO, "Debtor": YES_NO, "Tuition fees up to date": YES_NO,
    "Scholarship holder": YES_NO,
    "works_part_time": YES_NO, "hostel": YES_NO, "scholarship": YES_NO,
}

# The public copy of the dataset that this project downloads stores Course / Application mode /
# Previous qualification as RANKS (1, 2, 3 ...) instead of the official codes (33, 171, ...).
# The ranks follow the sorted official codes (verified against the published class sizes).
_RANKED = {"Course": COURSES, "Application mode": ADMISSION_ROUTES,
           "Previous qualification": PREVIOUS_QUALIFICATION}


def normalize_university(df: pd.DataFrame) -> pd.DataFrame:
    """Convert rank-coded columns to the official codes. Official files pass through unchanged."""
    df = df.copy()
    for col, table in _RANKED.items():
        if col not in df.columns:
            continue
        codes = sorted(table)
        v = pd.to_numeric(df[col], errors="coerce")
        if v.notna().all() and v.max() <= len(codes) and v.min() >= 1:      # rank-coded copy
            df[col] = v.astype(int).map(lambda r, c=codes: c[r - 1])
    return df


# ------------------------------------------------------------------ field dictionary
def _sem(n: str, key: str, label: str, help_: str):
    return f"Curricular units {n} sem ({key})", {"label": f"Sem {n[0]} · {label}", "help": help_, "group": f"Semester {n[0]} record"}


FIELDS: dict[str, dict] = {
    # ---- background
    "Course": {"label": "Course / programme", "group": "Background",
               "help": "The degree programme the student is enrolled in."},
    "Age at enrollment": {"label": "Age when joining", "group": "Background", "unit": "years",
                          "help": "Age in years on the day of enrolment. Older (mature) students often balance work and study."},
    "Daytime/evening attendance": {"label": "Class timing", "group": "Background",
                                   "help": "Whether the student attends day classes or evening classes."},
    "Displaced": {"label": "Living away from home", "group": "Background",
                  "help": "Yes if the student moved away from their home town to study (hostel / rented room)."},
    "Application order": {"label": "Course preference rank", "group": "Background",
                          "help": "0 = this course was the student's first choice, 1 = second choice, and so on. "
                                  "Students in a lower-choice course may be less motivated."},
    "Application mode": {"label": "Admission route", "group": "Background",
                         "help": "How the student got in (regular admission round, transfer, mature-student route ...)."},
    "Previous qualification": {"label": "Qualification before joining", "group": "Background",
                               "help": "Highest education completed before this course."},
    # ---- money
    "Tuition fees up to date": {"label": "Tuition fees paid up to date", "group": "Money matters",
                                "help": "Yes if there are no overdue fees."},
    "Debtor": {"label": "Has unpaid dues", "group": "Money matters",
               "help": "Yes if the student owes the institution money (fees, library, hostel ...)."},
    "Scholarship holder": {"label": "Receives a scholarship", "group": "Money matters",
                           "help": "Yes if the student holds a scholarship / fee waiver."},
    # ---- semester records
    **dict([
        _sem(n, "credited", "subjects credited",
             "Subjects already recognised from earlier study (exempted or transferred in) - no exam needed.")
        for n in ("1st", "2nd")]),
    **dict([
        _sem(n, "enrolled", "subjects registered", "How many subjects the student registered for this semester.")
        for n in ("1st", "2nd")]),
    **dict([
        _sem(n, "evaluations", "exams taken",
             "Total exams / assessments the student actually sat. Re-attempts count again, so this can exceed the subjects registered.")
        for n in ("1st", "2nd")]),
    **dict([
        _sem(n, "approved", "subjects cleared", "How many registered subjects the student passed.")
        for n in ("1st", "2nd")]),
    **dict([
        _sem(n, "grade", "average grade (0-20)",
             "Average mark over the subjects passed, on a 0-20 scale (10 = pass mark). 0 if nothing was passed.")
        for n in ("1st", "2nd")]),
    **dict([
        _sem(n, "without evaluations", "never assessed",
             "Registered subjects where the student did not attend any assessment at all (a strong warning sign).")
        for n in ("1st", "2nd")]),
    # ---- fields used only by the dropout/performance models' target and the class view
    "Target": {"label": "Final outcome", "group": "Outcome",
               "help": "Dropout = left without finishing; Enrolled = still studying; Graduate = finished the course."},
    # ---- attendance module (simulated)
    "age": {"label": "Age", "group": "Student", "help": "Age in years."},
    "commute_km": {"label": "Distance home to campus", "group": "Student", "unit": "km",
                   "help": "One-way distance from home to campus in kilometres."},
    "works_part_time": {"label": "Works part-time", "group": "Student", "help": "Has a job beside studies."},
    "hostel": {"label": "Lives in hostel", "group": "Student", "help": "Stays in the college hostel."},
    "scholarship": {"label": "Has a scholarship", "group": "Student", "help": "Receives a scholarship."},
    "semester": {"label": "Current semester (1-8)", "group": "Student", "help": "Which semester the student is in."},
    "prev_sem_attendance": {"label": "Last semester's attendance", "group": "Student", "unit": "%",
                            "help": "Overall attendance percentage in the previous semester."},
    "attendance_pct": {"label": "Attendance %", "group": "Weekly record",
                       "help": "Share of classes attended that week (0-100)."},
    "assignment_rate": {"label": "Assignments submitted", "group": "Weekly record",
                        "help": "Share of that week's assignments submitted on time (0 = none, 1 = all)."},
    "lms_logins": {"label": "Learning-portal logins", "group": "Weekly record",
                   "help": "Times the student logged in to the college learning portal (LMS) that week."},
}

# Macro-economic columns exist in the raw file but are deliberately not used.
EXCLUDED_AND_WHY = {
    "Gender, Marital status, Nationality, International, Special needs":
        "Sensitive personal attributes - a risk tool should not judge students by them. Tested: adding them back changes dropout ROC-AUC by only 0.000 to +0.004 (tested at semester 1 and 2, Logistic Regression and XGBoost).",
    "Mother's / Father's qualification and occupation":
        "Dozens of obscure codes, unfair to hold against a student, and adding them back moves ROC-AUC by at most 0.007 in either direction.",
    "Unemployment rate, Inflation rate, GDP":
        "National economy figures, identical for everyone who joined in the same year - not about the student.",
}


def label(feature: str) -> str:
    return FIELDS.get(feature, {}).get("label", feature)


def help_text(feature: str) -> str:
    h = FIELDS.get(feature, {}).get("help", "")
    u = FIELDS.get(feature, {}).get("unit")
    return h


def group(feature: str) -> str:
    return FIELDS.get(feature, {}).get("group", "Other")


def format_value(feature: str, value) -> str:
    """Human-readable value: codes -> names, 0/1 -> No/Yes, numbers trimmed."""
    m = CODE_MAPS.get(feature)
    try:
        if m is not None:
            return m.get(int(float(value)), str(value))
        v = float(value)
        unit = FIELDS.get(feature, {}).get("unit", "")
        txt = f"{v:.0f}" if v == int(v) else f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{txt} {unit}".strip()
    except (TypeError, ValueError):
        return str(value)


def pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: label(c) for c in df.columns})


def glossary_table(features=None) -> pd.DataFrame:
    rows = []
    for f, d in FIELDS.items():
        if features is not None and f not in features:
            continue
        rows.append({"Where it appears": d.get("group", ""), "Field": d["label"],
                     "What it means": d.get("help", ""), "Raw column name": f})
    return pd.DataFrame(rows)


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Friendly view of raw records: plain-English column names, codes replaced by names (Course 9119 ->
    'Informatics Engineering', 1 -> 'Yes')."""
    out = df.copy()
    for col, table in CODE_MAPS.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda v, t=table: t.get(int(v), v) if pd.notna(v) else v)
    return pretty_columns(out)


def friendly_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse of ``display_frame`` for column NAMES and Yes/No + names -> codes (accepts raw or friendly headers)."""
    rename = {label(c): c for c in FIELDS}
    out = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    for col, table in CODE_MAPS.items():
        if col in out.columns and not pd.api.types.is_numeric_dtype(out[col]):
            rev = {v.lower(): k for k, v in table.items()}
            out[col] = out[col].astype(str).str.strip().str.lower().map(rev)
    return out
