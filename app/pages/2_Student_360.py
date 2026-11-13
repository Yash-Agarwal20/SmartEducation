import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import config as C  # noqa: E402
from smartedu import engine as E  # noqa: E402
from smartedu import glossary as G  # noqa: E402
from smartedu import ui  # noqa: E402
from smartedu.risk import combine, recommendations  # noqa: E402
from smartedu.sentiment import LABELS as SENT_LABELS  # noqa: E402

PAGE = "s360"
ui.page_setup("Student 360", "🧭",
              "One student, four AI models, one clear picture - with reasons, what-ifs and next steps.",
              chips=["Real records of 4,424 students", "Explains every number", "Try changes live"])
ui.data_source_banner(("attendance",))
ui.require("dropout_sem1")

# ---------------------------------------------------------------- how much do we know?
top = st.columns([1.1, 1.4])
with top[0]:
    ui.preset_selector(PAGE)
with top[1]:
    stage = ui.stage_selector(PAGE)
prof = ui.profile()

with st.expander("📝 Student record - edit any field", expanded=True):
    ui.student_form(stage, PAGE)

# -------------------------------------------------------------- optional extra evidence
extra = st.columns(2)
use_att = extra[0].toggle("➕ Add weekly attendance record", value=False,
                          help="Adds the attendance early-warning model (trained on simulated data).")
use_fb = extra[1].toggle("➕ Add what the student said about their courses", value=False,
                         help="Adds the feedback-mood model (trained on real course reviews).")

components, details = {}, {}
subj = E.forecast_subjects(prof, stage)
drop = E.dropout_outcome(prof, stage)
if subj:
    components["performance"] = subj["risk"]
    details["performance"] = f"clears {subj['cleared']:.1f} of {subj['registered']:.0f} subjects"
components["dropout"] = drop["risk"]
details["dropout"] = f"{drop['risk']:.0%} chance of dropping out"

if use_att and E.available("attendance"):
    with st.expander("🗓️ Weekly attendance record", expanded=True):
        ab = E.load_bundle("attendance")
        sv = ui.attendance_static_form(ab, PAGE)
        weekly = ui.weekly_editor(ab, PAGE)
        p_att = float(ab.predict_proba(weekly.to_numpy(float)[None], pd.DataFrame([sv]))[0])
    components["attendance"] = p_att
    details["attendance"] = f"{p_att:.0%} chance attendance falls below 75%"
if use_fb and E.available("sentiment"):
    with st.expander("💬 Student feedback", expanded=True):
        txt = st.text_area("Comments (one per line)", "The lectures were clear, but the exams felt too rushed.",
                           key=f"{PAGE}:fb")
        comments = [c.strip() for c in txt.splitlines() if c.strip()]
    if comments:
        pr = ui.get_sentiment().predict_proba(comments)
        components["sentiment"] = float(pr[:, SENT_LABELS.index("Negative")].mean())
        details["sentiment"] = f"{components['sentiment']:.0%} negative on average"

with st.sidebar:
    with st.expander("⚖️ How much each signal counts", expanded=False):
        st.caption("The overall score is a weighted average. Change the weights to match your college's priorities.")
        weights = {k: st.slider({"dropout": "Dropout", "performance": "Backlog", "attendance": "Attendance",
                                 "sentiment": "Feedback mood"}[k], 0.0, 1.0, float(v), 0.05, key=f"{PAGE}:w_{k}")
                   for k, v in C.DEFAULT_RISK_WEIGHTS.items()}

res = combine(components, weights)
st.divider()

# ----------------------------------------------------------------------- the card
c1, c2, c3 = st.columns([1, 1.15, 1.25])
with c1:
    st.plotly_chart(ui.gauge(res["score"], "Overall risk"), key=f"{PAGE}_g")
    st.markdown(ui.pill(res["band"]), unsafe_allow_html=True)
    msg = {"Low": "No strong warning signs.", "Medium": "Some warning signs - worth a check-in.",
           "High": "Several warning signs - act soon."}[res["band"]]
    st.markdown(f"<div class='small' style='margin-top:6px'>{msg}</div>", unsafe_allow_html=True)
with c2:
    names = {"performance": "Backlog", "dropout": "Dropout", "attendance": "Attendance", "sentiment": "Feedback mood"}
    mods = list(res["module_risks"])
    cats = [names[m] for m in mods]
    vals = [100 * res["module_risks"][m] for m in mods]
    if len(mods) >= 3:
        fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself",
                                        line_color=ui.PAL["primary"], fillcolor="rgba(14,124,134,.25)"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")), showlegend=False,
                          title="Risk by signal")
    else:
        fig = go.Figure(go.Bar(x=vals, y=cats, orientation="h", marker_color=ui.PAL["primary"],
                               text=[f"{v:.0f}%" for v in vals], textposition="outside"))
        fig.update_layout(title="Risk by signal", xaxis=dict(range=[0, 118]), yaxis=dict(autorange="reversed"))
    st.plotly_chart(ui.style(fig, 300), key=f"{PAGE}_r")
with c3:
    st.markdown("#### Suggested next steps")
    for t in recommendations(components):
        st.markdown(f"- {t}")
    tbl = pd.DataFrame({"Signal": [names[m] for m in mods], "Risk": [f"{res['module_risks'][m]:.0%}" for m in mods],
                        "Weight": [f"{res['weights_used'][m]:.0%}" for m in mods],
                        "What it says": [details.get(m, "") for m in mods]})
    st.dataframe(tbl, hide_index=True, width="stretch")

# ------------------------------------------------------------------ module summary
k = st.columns(4)
if subj:
    k[0].markdown(ui.kpi("Next-semester forecast", f"{subj['cleared']:.1f} / {subj['registered']:.0f}",
                         "subjects expected to be cleared" + ui.subject_dots(subj["cleared"], subj["registered"]),
                         {"Low": "good", "Medium": "warn", "High": "bad"}[ui.band_of(subj["risk"])]), unsafe_allow_html=True)
else:
    k[0].markdown(ui.kpi("Next-semester forecast", "n/a", "Semester-2 results are already in the record", "info"),
                  unsafe_allow_html=True)
p = drop["probs"]
k[1].markdown(ui.kpi("Most likely outcome", {"Dropout": "Drops out", "Enrolled": "Still studying", "Graduate": "Graduates"}[drop["most_likely"]],
                     f"drop out {p[0]:.0%} · still studying {p[1]:.0%} · graduate {p[2]:.0%}",
                     {"Dropout": "bad", "Enrolled": "warn", "Graduate": "good"}[drop["most_likely"]]), unsafe_allow_html=True)
k[2].markdown(ui.kpi("Attendance", f"{components['attendance']:.0%}" if "attendance" in components else "not added",
                     "chance of falling below 75%" if "attendance" in components else "switch on above to include",
                     "info"), unsafe_allow_html=True)
k[3].markdown(ui.kpi("Feedback mood", f"{components['sentiment']:.0%}" if "sentiment" in components else "not added",
                     "average negative feedback" if "sentiment" in components else "switch on above to include",
                     "info"), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- tabs
t1, t2, t3 = st.tabs(["🔍 Why this result?", "🎛️ What if ...?", "👥 Compared with other students"])
with t1:
    cc = st.columns(2)
    with cc[0]:
        with st.spinner("Working out the reasons ..."):
            names_d, vals_d, _ = ui.explain(E.DROP_KEY[stage], prof, class_index=C.DROPOUT_LABELS.index("Dropout"))
        st.markdown("**Dropout risk - what pushes it up or down**")
        for r in ui.reasons(np.array(names_d), np.array(vals_d), 100, "percentage points", subject="the dropout risk"):
            st.markdown(f"- {r}")
        st.plotly_chart(ui.attribution_chart(names_d, np.array(vals_d), "", 100, "pts"), key=f"{PAGE}_sd")
    with cc[1]:
        if subj:
            names_s, vals_s, _ = ui.explain(E.PERF_KEY[stage], prof)
            st.markdown("**Subjects cleared - what pushes it up or down**")
            for r in ui.reasons(np.array(names_s), np.array(vals_s), 1, "subjects", subject="the number of subjects cleared"):
                st.markdown(f"- {r}")
            st.plotly_chart(ui.attribution_chart(names_s, np.array(vals_s), "", 1, "subj", bad_is_positive=False),
                            key=f"{PAGE}_ss")
        else:
            st.info("The subject forecast is not needed after semester 2 - the results are already known.")
    ui.callout("<b>How to read this:</b> each bar is one fact about the student. Red pushes the risk <i>up</i>, green pushes it "
               "<i>down</i>, compared with an average student. The method is called SHAP.")
with t2:
    ui.what_if_panel(prof, stage, PAGE)
with t3:
    ui.cohort_position(drop["risk"], stage, PAGE)

card = {"student": "", "stage": stage, "overall_risk": res["score"], "band": res["band"],
        "module_risks": res["module_risks"], "details": details}
import json  # noqa: E402
st.download_button("⬇️ Download this risk card (JSON)", json.dumps(card, indent=2, default=float).encode(),
                   "risk_card.json", "application/json")
st.caption("Decision support only: the overall score is a weighted average of the signals, not a trained model, "
           "and predictions are estimates about groups of similar students - never a verdict on one person.")
