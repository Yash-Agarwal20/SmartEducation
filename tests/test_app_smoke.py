"""Headless smoke test of every dashboard page (Streamlit AppTest).

Run after training:   python -m pytest tests -q      (or: python tests/test_app_smoke.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["app/Home.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "app" / "pages").glob("*.py"))


def run_page(path, click=None):
    """Start the app from Home.py (like `streamlit run app/Home.py`) and open the page the way a user does,
    so links between pages resolve exactly as they do in the real app."""
    at = AppTest.from_file(str(ROOT / "app/Home.py"), default_timeout=120).run()
    if path != "app/Home.py":
        at.switch_page(path.replace("app/", "", 1)).run()
    if click:
        for b in at.button:
            if click in (b.label or ""):
                b.click()
        at.run()
    return at


def test_all_pages_render():
    for p in PAGES:
        at = run_page(p)
        assert not at.exception, f"{p}: {[e.value for e in at.exception]}"


if __name__ == "__main__":
    for p in PAGES:
        at = run_page(p)
        print(("OK   " if not at.exception else "FAIL ") + p, [e.value[:200] for e in at.exception])
