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

PAGE = "drop"
M = about.by_key("dropout")
ui.page_setup("Dropout warning", "🚪", M["question"],
              chips=["Model B", "Neural network (MLP)", "Real university records"])
ui.require("dropout_start", "dropout_sem1", "dropout_sem2")
with st.expander("What this model is, and what data it learned from", expanded=False):
    ui.module_strip(M)

top = st.columns([1.1, 1.4])
with top[0]:
    ui.preset_selector(PAGE)
with top[1]:
    stage = ui.stage_selector(PAGE)
st.caption(about.STAGE_EXPLAINER[stage])
prof = ui.profile()
with st.expander("📝 Student record - edit any field", expanded=True):
    ui.student_form(stage, PAGE)

d = E.dropout_outcome(prof, stage)
p = d["probs"]
st.divider()
c = st.columns([1, 1.4])
with c[0]:
    st.plotly_chart(ui.gauge(d["risk"], "Chance of dropping out"), key=f"{PAGE}_g")
    band = ui.band_of(d["risk"])
    st.markdown(ui.pill(band), unsafe_allow_html=True)
with c[1]:
    st.plotly_chart(ui.prob_bar(["Drops out", "Still studying", "Graduates"], p,
                                "The three possible outcomes (they add up to 100%)"), key=f"{PAGE}_p")
    st.markdown(f"Most likely outcome: **{ {'Dropout': 'drops out', 'Enrolled': 'still studying', 'Graduate': 'graduates'}[d['most_likely']] }** "
                f"({100 * max(p):.0f}%).")

t1, t2, t3, t4 = st.tabs(["🔍 Why this result?", "🎛️ What if ...?", "👥 Compared with other students",
                          "🎯 How accurate is it?"])
with t1:
    with st.spinner("Working out the reasons ..."):
        names, vals, _ = ui.explain(E.DROP_KEY[stage], prof, class_index=C.DROPOUT_LABELS.index("Dropout"))
    for r in ui.reasons(np.array(names), np.array(vals), 100, "percentage points", subject="the dropout chance"):
        st.markdown(f"- {r}")
    st.plotly_chart(ui.attribution_chart(names, np.array(vals), "Effect on the dropout chance", 100, "pts"), key=f"{PAGE}_shap")
    ui.callout("<b>How to read this:</b> red bars push the dropout chance <i>up</i>, green bars push it <i>down</i>, "
               "compared with an average student. Percentage points are added to or taken from the average chance.")
with t2:
    ui.what_if_panel(prof, stage, PAGE)
with t3:
    ui.cohort_position(d["risk"], stage, PAGE)
with t4:
    m = (load_json(C.REPORTS_DIR / "metrics_dropout.json") or {}).get("settings", {})
    if m:
        cur = m[stage]["test"]["MLP (deep learning)"]
        a = st.columns(3)
        a[0].markdown(ui.kpi("Real dropouts caught", f"{cur['dropout_recall']:.0%}",
                             f"of students who really dropped out were flagged (stage: {C.STAGES[stage]['label'].lower()})", "good"),
                      unsafe_allow_html=True)
        a[1].markdown(ui.kpi("Flags that were right", f"{cur['dropout_precision']:.0%}",
                             "of flagged students really did drop out", "info"), unsafe_allow_html=True)
        a[2].markdown(ui.kpi("Overall ranking quality", f"{cur['roc_auc']:.2f}",
                             "ROC-AUC: 0.5 = coin toss, 1.0 = perfect", "info"), unsafe_allow_html=True)
        stages = list(m)
        fig = go.Figure(go.Bar(x=[C.STAGES[s]["label"] for s in stages],
                               y=[m[s]["test"]["MLP (deep learning)"]["roc_auc"] for s in stages],
                               marker_color=[ui.PAL["primary"] if s == stage else ui.PAL["line"] for s in stages],
                               text=[f"{m[s]['test']['MLP (deep learning)']['roc_auc']:.2f}" for s in stages],
                               textposition="outside"))
        fig.update_layout(title="The more we know, the better the warning (ROC-AUC of the neural network)",
                          yaxis=dict(range=[0.5, 1.0], title="ROC-AUC"))
        st.plotly_chart(ui.style(fig, 320), key=f"{PAGE}_stages")
        ui.callout("<b>Honest note:</b> the simpler models (Logistic Regression, Random Forest, XGBoost) score about the same as the "
                   "neural network on this kind of table - see the 'Accuracy' page. The big gain comes from <i>knowing more</i> "
                   "about the student, not from a fancier model.")
