"""Synthetic data generators.

IMPORTANT
---------
* The performance, dropout and feedback generators mimic the *schema* of the real
  public datasets so the whole pipeline can be tested without downloading
  anything. Numbers obtained on them are NOT research results.
* The attendance generator is the project's *simulated attendance dataset*
  (no clean public attendance dataset exists). Its generating rules are
  documented below so that the report can describe them honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================ B. dropout
_COURSES = [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500,
            9556, 9670, 9773, 9853, 9991]
_APP_MODES = [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57]
_PREV_Q = [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43]
_QUAL = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
_OCC = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194]


def make_dropout(n: int = 4424, seed: int = C.SEED) -> pd.DataFrame:
    """Schema of 'Predict Students' Dropout and Academic Success' (UCI id 697)."""
    r = np.random.default_rng(seed)
    s = r.normal(0, 1, n)                                   # latent academic strength
    gender = (r.random(n) < 0.35).astype(int)
    age = np.clip(np.round(r.gamma(3.0, 2.2, n) + 17.5 - 0.5 * s), 17, 70).astype(int)
    debtor = (r.random(n) < _sigmoid(-2.2 - 0.5 * s)).astype(int)
    tuition = (r.random(n) < _sigmoid(2.4 + 0.5 * s - 1.2 * debtor)).astype(int)
    scholar = (r.random(n) < _sigmoid(-1.3 + 0.9 * s)).astype(int)

    def sem(prefix_enrolled_mean):
        enrolled = np.clip(np.round(r.normal(prefix_enrolled_mean, 1.8, n)), 0, 26).astype(int)
        p_pass = _sigmoid(0.2 + 1.3 * s - 0.4 * debtor - 0.02 * (age - 20))
        approved = r.binomial(enrolled, np.clip(p_pass, 0.01, 0.99))
        evals = np.clip(enrolled + r.poisson(1.5, n) + (r.random(n) < 0.3) * r.poisson(3, n), 0, 45)
        grade = np.where(approved > 0, np.clip(r.normal(12 + 1.5 * s, 1.6, n), 0, 18.5), 0.0)
        credited = np.where(r.random(n) < 0.12, r.integers(1, 12, n), 0)
        noeval = np.where(r.random(n) < 0.08, r.integers(1, 8, n), 0)
        return enrolled, approved, evals, grade, credited, noeval

    e1, a1, ev1, g1, cr1, ne1 = sem(6.2)
    e2, a2, ev2, g2, cr2, ne2 = sem(6.1)
    ratio = (a1 + a2) / np.maximum(e1 + e2, 1)
    z_drop = -0.35 + 2.0 * (0.5 - ratio) * 2 + 0.9 * debtor - 1.4 * tuition + 0.03 * (age - 22) - 0.6 * scholar
    z_grad = -0.4 + 3.2 * (ratio - 0.5) * 2 + 0.7 * tuition + 0.5 * scholar
    z_enr = -1.0 + 0.0 * ratio + r.normal(0, 0.3, n)
    logits = np.stack([z_drop, z_enr, z_grad], axis=1) + r.gumbel(0, 0.9, (n, 3))
    target = np.array(C.DROPOUT_LABELS)[logits.argmax(1)]
    df = pd.DataFrame({
        "Marital status": r.choice([1, 2, 3, 4, 5, 6], n, p=[0.886, 0.08, 0.002, 0.01, 0.02, 0.002]),
        "Application mode": r.choice(_APP_MODES, n),
        "Application order": r.choice(range(0, 10), n, p=[0.32, 0.38, 0.1, 0.07, 0.05, 0.04, 0.02, 0.01, 0.005, 0.005]),
        "Course": r.choice(_COURSES, n),
        "Daytime/evening attendance": (r.random(n) < 0.89).astype(int),
        "Previous qualification": r.choice(_PREV_Q, n, p=np.r_[[0.84], np.full(len(_PREV_Q) - 1, 0.16 / (len(_PREV_Q) - 1))]),
        "Previous qualification (grade)": np.clip(r.normal(132 + 4 * s, 13, n), 95, 190).round(1),
        "Nacionality": r.choice([1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26], n, p=np.r_[[0.97], np.full(11, 0.03 / 11)]),
        "Mother's qualification": r.choice(_QUAL, n),
        "Father's qualification": r.choice(_QUAL, n),
        "Mother's occupation": r.choice(_OCC, n),
        "Father's occupation": r.choice(_OCC, n),
        "Admission grade": np.clip(r.normal(127 + 5 * s, 14, n), 95, 190).round(1),
        "Displaced": (r.random(n) < 0.55).astype(int),
        "Educational special needs": (r.random(n) < 0.01).astype(int),
        "Debtor": debtor, "Tuition fees up to date": tuition, "Gender": gender,
        "Scholarship holder": scholar, "Age at enrollment": age,
        "International": (r.random(n) < 0.025).astype(int),
        "Curricular units 1st sem (credited)": cr1,
        "Curricular units 1st sem (enrolled)": e1,
        "Curricular units 1st sem (evaluations)": ev1,
        "Curricular units 1st sem (approved)": a1,
        "Curricular units 1st sem (grade)": g1.round(2),
        "Curricular units 1st sem (without evaluations)": ne1,
        "Curricular units 2nd sem (credited)": cr2,
        "Curricular units 2nd sem (enrolled)": e2,
        "Curricular units 2nd sem (evaluations)": ev2,
        "Curricular units 2nd sem (approved)": a2,
        "Curricular units 2nd sem (grade)": g2.round(2),
        "Curricular units 2nd sem (without evaluations)": ne2,
        "Unemployment rate": r.choice([7.6, 8.9, 9.4, 10.8, 11.1, 12.4, 12.7, 13.9, 15.5, 16.2], n),
        "Inflation rate": r.choice([-0.8, 0.3, 0.6, 1.4, 2.6, 2.8, 3.7], n),
        "GDP": r.choice([-4.06, -3.12, -1.7, 0.32, 0.79, 1.74, 2.02, 3.51], n),
        "Target": target,
    })
    return df


# ============================================================ C. attendance
STATIC_ATT_COLS = ["age", "commute_km", "works_part_time", "hostel",
                   "scholarship", "semester", "prev_sem_attendance"]
SEQ_CHANNELS = ["attendance_pct", "assignment_rate", "lms_logins"]


def make_attendance(n: int = 3000, seed: int = C.SEED,
                    weeks: int = C.TOTAL_WEEKS, observed: int = C.OBSERVED_WEEKS):
    """Simulated weekly attendance records (documented generating rules).

    Rules
    -----
    1. Every student has a latent engagement score e ~ N(0,1).
    2. Baseline attendance = 82 + 8e - 0.35*commute_km - 4*works_part_time
       + 2*hostel + noise.
    3. 20% of students are "fading": attendance drifts down ~1.8 points per week.
       Others drift slightly (N(-0.2, 0.5) per week).
    4. Weekly attendance = baseline + drift*week + AR(1) noise, clipped to [0,100].
       25% of students additionally suffer an unpredictable late shock (-15 points
       from a random week after the observed window), so early prediction is imperfect.
    5. Assignment completion and LMS logins are correlated with e.
    6. Label: At-Risk (1) if the mean attendance over ALL ``weeks`` is below
       ``ATTENDANCE_THRESHOLD`` (75%). The model only sees the first
       ``observed`` weeks -> early-warning prediction.

    Returns (static_df, seq[n, observed, 3], full_attendance[n, weeks], y[n]).
    """
    r = np.random.default_rng(seed)
    e = r.normal(0, 1, n)
    commute = np.clip(r.gamma(2.0, 4.0, n), 0.3, 60)
    works = (r.random(n) < _sigmoid(-1.0 - 0.3 * e)).astype(int)
    hostel = (r.random(n) < 0.35).astype(int)
    prev = np.clip(r.normal(80 + 8 * e, 9, n), 30, 100)
    static = pd.DataFrame({
        "age": np.clip(np.round(r.normal(21, 2.2, n)), 17, 32).astype(int),
        "commute_km": commute.round(1),
        "works_part_time": works,
        "hostel": hostel,
        "scholarship": (r.random(n) < 0.25).astype(int),
        "semester": r.integers(1, 9, n),
        "prev_sem_attendance": prev.round(1),
    })
    base = 88 + 8 * e - 0.35 * commute - 4 * works + 2 * hostel + r.normal(0, 3, n)
    fading = r.random(n) < 0.20
    drift = np.where(fading, r.normal(-1.8, 0.5, n), r.normal(-0.2, 0.5, n))
    # unpredictable late-semester shock (illness, job, family): 25% of students lose
    # ~15 points from a random week AFTER the observed window -> early warning is imperfect
    shock = r.random(n) < 0.25
    shock_week = r.integers(observed, weeks, n)
    shock_size = r.normal(15, 5, n)
    att = np.zeros((n, weeks))
    noise = np.zeros(n)
    for t in range(weeks):
        noise = 0.6 * noise + r.normal(0, 4.0, n)
        hit = shock & (t >= shock_week)
        att[:, t] = np.clip(base + drift * t + noise - hit * shock_size, 0, 100)
    assign = np.clip(0.55 + 0.17 * e[:, None] + 0.1 * (att[:, :observed] - 75) / 25
                     + r.normal(0, 0.12, (n, observed)), 0, 1)
    logins = r.poisson(np.exp(1.3 + 0.3 * e[:, None] + 0.004 * (att[:, :observed] - 75)), (n, observed))
    seq = np.stack([att[:, :observed], assign, logins.astype(float)], axis=2)
    y = (att.mean(axis=1) < C.ATTENDANCE_THRESHOLD).astype(int)
    return static, seq, att, y


# ============================================================ D. feedback text
_ASPECTS = ["lectures", "the instructor", "assignments", "the lab sessions", "course material",
            "the exams", "the pace of the course", "doubt-clearing sessions", "the syllabus",
            "placement support", "the campus facilities", "the online platform"]
_POS = ["excellent", "very helpful", "clear and engaging", "well organised", "inspiring",
        "great", "super useful", "outstanding", "really good", "easy to follow", "amazing"]
_NEG = ["terrible", "confusing", "boring", "poorly organised", "a waste of time", "too rushed",
        "very disappointing", "useless", "unclear", "frustrating", "way too difficult", "bad"]
_NEU = ["okay", "average", "fine", "acceptable", "neither good nor bad", "as expected",
        "decent", "standard", "mixed", "alright"]
_OPEN_POS = ["I really enjoyed this course.", "Overall a great experience.", "Happy with the semester.", ""]
_OPEN_NEG = ["I am not happy with this course.", "Overall a poor experience.", "Honestly disappointed.", ""]
_OPEN_NEU = ["The course was what I expected.", "Nothing special to report.", "It was an ordinary semester.", ""]


def make_feedback(n: int = 3000, seed: int = C.SEED) -> pd.DataFrame:
    """Template-based student feedback with ~8% mixed and ~4% flipped (noisy) labels."""
    r = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        lab = r.choice(["Positive", "Negative", "Neutral"], p=[0.42, 0.33, 0.25])
        bank, opener = {"Positive": (_POS, _OPEN_POS), "Negative": (_NEG, _OPEN_NEG),
                        "Neutral": (_NEU, _OPEN_NEU)}[lab]
        parts = [r.choice(opener)]
        for _ in range(r.integers(1, 3)):
            parts.append(f"{str(r.choice(_ASPECTS)).capitalize()} {r.choice(['was', 'were', 'felt'])} {r.choice(bank)}.")
        if r.random() < 0.08:        # mixed sentiment, e.g. positive text with one negative aside
            others = [b for b in (_POS, _NEG, _NEU) if b is not bank]
            other = others[r.integers(0, len(others))]
            parts.append(f"But {r.choice(_ASPECTS)} was {r.choice(other)}.")
        text = " ".join(p for p in parts if p)
        if r.random() < 0.04:
            lab = r.choice([x for x in ("Positive", "Negative", "Neutral") if x != lab])
        rows.append((text, lab))
    return pd.DataFrame(rows, columns=["text", "label"])
