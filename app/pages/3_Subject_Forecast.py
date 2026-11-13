import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import about  # noqa: E402
from smartedu import config as C  # noqa: E402
from smartedu import engine as E  # noqa: E402
from smartedu import ui  # noqa: E402
from smartedu.utils import load_json  # noqa: E402

PAGE = "subj"
M = about.by_key("performance")
ui.page_setup("Subject forecast", "📘", M["question"],
              chips=["Model A", "Neural network (MLP)", "Real university records"])
ui.require("performance_start", "performance_sem1")
with st.expander("What this model is, and what data it learned from", expanded=False):
    ui.module_strip(M)

top = st.columns([1.1, 1.4])
with top[0]:
    ui.preset_selector(PAGE)
with top[1]:
    stage = ui.stage_selector(PAGE, allowed=("start", "sem1"),
                              help_="'Before any exam results' uses background only; 'After semester 1' also uses what the student achieved.")
st.caption(about.STAGE_EXPLAINER[stage])
prof = ui.profile()
with st.expander("📝 Student record - edit any field", expanded=True):
    ui.student_form(stage, PAGE)

f = E.forecast_subjects(prof, stage)
st.divider()
metrics = (load_json(C.REPORTS_DIR / "metrics_performance.json") or {}).get("settings", {}).get(stage, {})
mlp = metrics.get("test", {}).get("MLP (deep learning)", {})
naive = metrics.get("test", {}).get("Typical pass rate (no model)", {})
err = f["typical_error"] or mlp.get("rmse", 1.0)

c = st.columns([1.2, 1, 1])
tone = {"Low": "good", "Medium": "warn", "High": "bad"}[ui.band_of(f["risk"])]
c[0].markdown(ui.kpi("Subjects expected to clear", f"{f['cleared']:.1f} of {f['registered']:.0f}",
                     "registered for semester 2" + ui.subject_dots(f["cleared"], f["registered"]), tone),
              unsafe_allow_html=True)
c[1].markdown(ui.kpi("Expected backlog", f"{f['backlog']:.1f} subjects",
                     "subjects likely to be left uncleared", tone), unsafe_allow_html=True)
lo, hi = max(0.0, f["cleared"] - err), min(f["registered"], f["cleared"] + err)
c[2].markdown(ui.kpi("Likely range", f"{lo:.1f} to {hi:.1f}",
                     f"typical error of the model is about &plusmn;{err:.1f} subjects on unseen students", "info"),
              unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Bar(y=["Semester 2"], x=[f["cleared"]], orientation="h", name="Expected to clear",
                     marker_color=ui.PAL["good"], text=[f"{f['cleared']:.1f}"], textposition="inside"))
fig.add_trace(go.Bar(y=["Semester 2"], x=[f["backlog"]], orientation="h", name="Expected backlog",
                     marker_color=ui.PAL["bad"], text=[f"{f['backlog']:.1f}"], textposition="inside"))
fig.update_layout(barmode="stack", title="Registered subjects: expected result", legend=dict(orientation="h", y=-0.4),
                  xaxis=dict(range=[0, max(f["registered"], 1)], title="subjects"))
st.plotly_chart(ui.style(fig, 210), key=f"{PAGE}_bar")

t1, t2, t3 = st.tabs(["🔍 Why this forecast?", "🎛️ What if ...?", "🎯 How accurate is it?"])
with t1:
    with st.spinner("Working out the reasons ..."):
        names, vals, base = ui.explain(E.PERF_KEY[stage], prof)
    for r in ui.reasons(np.array(names), np.array(vals), 1, "subjects", subject="the number of subjects cleared"):
        st.markdown(f"- {r}")
    st.plotly_chart(ui.attribution_chart(names, np.array(vals), "Effect on subjects cleared", 1, "subj",
                                         bad_is_positive=False), key=f"{PAGE}_shap")
    ui.callout("<b>How to read this:</b> each bar is one fact about the student. Green pushes the number of cleared subjects "
               "<i>up</i>, red pushes it <i>down</i>, compared with an average student. The method is called SHAP.")
with t2:
    ui.what_if_panel(prof, stage, PAGE)
with t3:
    if mlp:
        a = st.columns(3)
        a[0].markdown(ui.kpi("Typical error of the model", f"&plusmn;{mlp['rmse']:.2f} subjects",
                             f"on {metrics['n_test']} students it never saw", "good"), unsafe_allow_html=True)
        a[1].markdown(ui.kpi("Guessing the average pass rate", f"&plusmn;{naive['rmse']:.2f} subjects",
                             "what you get with no model at all", "warn"), unsafe_allow_html=True)
        a[2].markdown(ui.kpi("Within 1 subject", f"{mlp['within_1_subject']:.0%}",
                             "of students: forecast off by at most one subject", "info"), unsafe_allow_html=True)
        rows = {k: v["rmse"] for k, v in metrics["test"].items()}
        order = sorted(rows, key=rows.get)
        fig = go.Figure(go.Bar(x=[rows[k] for k in order], y=order, orientation="h",
                               marker_color=[ui.PAL["primary"] if "MLP" in k else ui.PAL["line"] for k in order],
                               text=[f"{rows[k]:.2f}" for k in order], textposition="outside", cliponaxis=False))
        fig.update_layout(title=f"Typical error (subjects) by model - {C.STAGES[stage]['label'].lower()}  (shorter is better)",
                          yaxis=dict(autorange="reversed"), xaxis=dict(title="typical error in subjects"))
        st.plotly_chart(ui.style(fig, 330), key=f"{PAGE}_acc")
        ui.callout("The neural network (teal) is about as good as the simpler models here - on table-like data that is normal. "
                   "What matters: every model beats guessing, and knowing the semester-1 result roughly halves the error.")
