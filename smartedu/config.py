"""Central configuration: paths, seeds, thresholds and label definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, MODELS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# 70 / 15 / 15 split (train / validation / test)
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---- University records (shared by modules A and B) ---------------------------
# One real dataset of university students feeds BOTH the "subjects cleared" forecast (A) and the
# dropout model (B). Columns that are sensitive (gender, nationality, marital status, parents'
# jobs ...) or not about the student (national economy figures) are deliberately NOT used.
def _sem(n, k):
    return f"Curricular units {n} sem ({k})"

BACKGROUND_COLS = ["Course", "Age at enrollment", "Daytime/evening attendance", "Displaced",
                   "Application order", "Application mode", "Previous qualification",
                   "Tuition fees up to date", "Debtor", "Scholarship holder"]
SEM1_COLS = [_sem("1st", k) for k in ("credited", "enrolled", "evaluations", "approved", "grade",
                                      "without evaluations")]
SEM2_REGISTRATION_COLS = [_sem("2nd", "credited"), _sem("2nd", "enrolled")]       # known when sem 2 starts
SEM2_RESULT_COLS = [_sem("2nd", k) for k in ("evaluations", "approved", "grade", "without evaluations")]
UNIVERSITY_CATEGORICAL = ["Course", "Application mode", "Previous qualification"]

# "How much do we know about the student?"  stage -> input columns
STAGES = {
    "start": {"label": "Before any exam results", "cols": BACKGROUND_COLS + SEM2_REGISTRATION_COLS},
    "sem1": {"label": "After semester 1", "cols": BACKGROUND_COLS + SEM1_COLS + SEM2_REGISTRATION_COLS},
    "sem2": {"label": "After semester 2", "cols": BACKGROUND_COLS + SEM1_COLS + SEM2_REGISTRATION_COLS + SEM2_RESULT_COLS},
}

# ---- Module A: subjects-cleared forecast --------------------------------------
PERFORMANCE_TARGET = _sem("2nd", "approved")      # subjects cleared in semester 2
PERFORMANCE_REGISTERED = _sem("2nd", "enrolled")  # subjects registered in semester 2 (known at registration)
PERFORMANCE_STAGES = ["start", "sem1"]

# ---- Module B: dropout --------------------------------------------------------
DROPOUT_TARGET = "Target"
DROPOUT_LABELS = ["Dropout", "Enrolled", "Graduate"]
DROPOUT_STAGES = ["start", "sem1", "sem2"]
DROPOUT_CATEGORICAL = UNIVERSITY_CATEGORICAL

# ---- Module C: attendance ----------------------------------------------------
ATTENDANCE_THRESHOLD = 75.0   # final attendance % below this = At-Risk
TOTAL_WEEKS = 16              # length of a semester (simulated data)
OBSERVED_WEEKS = 8            # weeks the model is allowed to see (early warning)
ATTENDANCE_LABELS = ["Safe", "At-Risk"]

# ---- Module D: sentiment -----------------------------------------------------
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# ---- Combined risk card ------------------------------------------------------
DEFAULT_RISK_WEIGHTS = {
    "dropout": 0.35,
    "performance": 0.30,
    "attendance": 0.25,
    "sentiment": 0.10,
}
RISK_BANDS = [(0.33, "Low"), (0.66, "Medium"), (1.01, "High")]
