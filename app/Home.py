"""Smart Education & Student Intelligence System - home page.   Run:  streamlit run app/Home.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import about  # noqa: E402
from smartedu import config as C  # noqa: E402
from smartedu import engine as E  # noqa: E402
from smartedu import glossary as G  # noqa: E402
from smartedu import ui  # noqa: E402

ui.page_setup("Smart Education & Student Intelligence", "🎓",
              "An early-warning assistant for college advisers. It reads a student's record and shows who may need help, "
              "why, and what to try - while there is still time to act.",
              chips=["For colleges & universities", "4 deep-learning models", "Explains every number"])
ui.require("dropout_sem1", "performance_sem1")


def describe(p: dict) -> str:
    s1 = C.SEM1_COLS
    fees = "fees paid" if int(p["Tuition fees up to date"]) else "fees overdue"
    sch = ", on a scholarship" if int(p["Scholarship holder"]) else ""
    return (f"{int(p['Age at enrollment'])}-year-old {G.format_value('Course', p['Course'])} student, {fees}{sch}. "
            f"In semester 1 they cleared {int(p[s1[3]])} of {int(p[s1[1]])} subjects "
            f"(average grade {float(p[s1[4]]):.1f} out of 20) and never sat an exam in {int(p[s1[5]])}.")


# ------------------------------------------------------------------ live example
st.markdown("## See it work")
left, right = st.columns([1, 1.5], gap="large")
with left:
    name = st.radio("Pick an example student", list(E.PRESETS), key="home_pick",
                    help="Four ready-made student records. Open 'Student 360' to edit every field yourself.")
    prof = {**E.default_profile(), **E.PRESETS[name]}
    st.markdown(f"<div class='callout'>{describe(prof)}</div>", unsafe_allow_html=True)
    st.caption("Prediction made at the end of semester 1 - early enough to help.")

subj = E.forecast_subjects(prof, "sem1")
drop = E.dropout_outcome(prof, "sem1")
from smartedu.risk import combine  # noqa: E402

res = combine({"dropout": drop["risk"], "performance": subj["risk"]})
with right:
    k = st.columns(3)
    k[0].markdown(ui.kpi("Subjects expected to clear", f"{subj['cleared']:.1f} of {subj['registered']:.0f}",
                         f"next semester{ui.subject_dots(subj['cleared'], subj['registered'])}",
                         {"Low": "good", "Medium": "warn", "High": "bad"}[ui.band_of(subj["risk"])]),
                  unsafe_allow_html=True)
    p = drop["probs"]
    k[1].markdown(ui.kpi("Most likely outcome",
                         {"Dropout": "Drops out", "Enrolled": "Still studying", "Graduate": "Graduates"}[drop["most_likely"]],
                         f"drop out {p[0]:.0%} &middot; still studying {p[1]:.0%} &middot; graduate {p[2]:.0%}",
                         {"Dropout": "bad", "Enrolled": "warn", "Graduate": "good"}[drop["most_likely"]]),
                  unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Overall risk", f"{res['score']:.0%}", f"{ui.pill(res['band'])}", "info"),
                  unsafe_allow_html=True)
    with st.spinner("Working out the reasons ..."):
        names, vals, _ = ui.explain(E.DROP_KEY["sem1"], prof, class_index=C.DROPOUT_LABELS.index("Dropout"))
    st.markdown("**Why the dropout chance is what it is**")
    for r in ui.reasons(np.array(names), np.array(vals), 100, "percentage points", top=3, subject="the dropout chance"):
        st.markdown(f"- {r}")
    st.page_link("pages/2_Student_360.py", label="Open the full Student 360 view and edit this student", icon="🧭")

# ------------------------------------------------------------------ who is it for
st.markdown("## Who is it for?")
ui.callout(f"<b>{about.AUDIENCE}</b> {about.REAL_DATA_NOTE}")
c = st.columns(3)
c[0].markdown(ui.card("Class advisers and mentors",
                      "See which students in a class need a conversation this week, and the facts behind each flag.",
                      "Class overview"), unsafe_allow_html=True)
c[1].markdown(ui.card("Heads of department and administration",
                      "Spot patterns by course, fee status or scholarship, and test which support (fee help, tutoring) could matter.",
                      "What-if tool"), unsafe_allow_html=True)
c[2].markdown(ui.card("Students and project reviewers",
                      "A transparent demonstration of how deep learning is used on real education data - every step is explained.",
                      "How it works"), unsafe_allow_html=True)

# ------------------------------------------------------------------ four models
st.markdown("## Four AI models, four questions")
mods = about.modules()
cols = st.columns(4)
for col, m in zip(cols, mods):
    data_tag = "real data" if m["real"] else "simulated data"
    col.markdown(ui.card(f"{m['icon']} {m['name']}", m["question"], f"{m['kind']} &middot; {data_tag}"),
                 unsafe_allow_html=True)
st.caption("Open 'How it works' in the sidebar to see exactly which data trains which model.")

# ------------------------------------------------------------------ honest accuracy
st.markdown("## How accurate are they, honestly?")
cards = ui.best_metric_cards()
cc = st.columns(len(cards))
for col, (t, v, s, tone) in zip(cc, cards):
    col.markdown(ui.kpi(t, v, s, tone), unsafe_allow_html=True)
st.caption("All numbers are measured on students the models never saw while learning. "
           "The 'Accuracy' page shows every comparison, including where simpler models do as well as deep learning.")

st.markdown("## Where to go next")
n = st.columns(4)
n[0].page_link("pages/1_How_It_Works.py", label="How it works", icon="📖")
n[1].page_link("pages/2_Student_360.py", label="Student 360", icon="🧭")
n[2].page_link("pages/7_Class_Overview.py", label="Class overview", icon="👥")
n[3].page_link("pages/8_Inside_the_Network.py", label="Inside the network", icon="🧠")
st.caption("Decision support only: predictions are estimates based on patterns in past students, not verdicts about a person.")
