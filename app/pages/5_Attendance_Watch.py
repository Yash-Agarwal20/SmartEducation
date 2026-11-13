import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import about  # noqa: E402
from smartedu import config as C  # noqa: E402
from smartedu import engine as E  # noqa: E402
from smartedu import glossary as G  # noqa: E402
from smartedu import ui  # noqa: E402
from smartedu.utils import load_json  # noqa: E402

PAGE = "att"
M = about.by_key("attendance")
ui.page_setup("Attendance watch", "🗓️", M["question"],
              chips=["Model C", "BiLSTM (reads week by week)", "Simulated data"])
ui.data_source_banner(("attendance",))
ui.require("attendance")
with st.expander("What this model is, and what data it learned from", expanded=False):
    ui.module_strip(M)

b = E.load_bundle("attendance")
st.markdown("### 1. About the student")
sv = ui.attendance_static_form(b, PAGE)
st.markdown("### 2. The first 8 weeks")
weekly = ui.weekly_editor(b, PAGE)
seq = weekly.to_numpy(float)[None]
static = pd.DataFrame([sv])
p = float(b.predict_proba(seq, static)[0])
flag = p >= b.threshold
band = ui.band_of(p)

st.divider()
c = st.columns([1, 1.5])
with c[0]:
    st.plotly_chart(ui.gauge(p, "Chance attendance ends below 75%"), key=f"{PAGE}_g")
    st.markdown(ui.pill(band, "Flag for a check-in" if flag else "On track"), unsafe_allow_html=True)
    st.caption(f"The model raises a flag from {100 * b.threshold:.0f}% upwards - the line is set low on purpose, because "
               "missing a student who is slipping is worse than a needless chat.")
with c[1]:
    ch = b.channels
    weeks = list(range(1, b.seq_len + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=weekly[ch[0]], mode="lines+markers", name="Attendance %",
                             line=dict(color=ui.PAL["primary"], width=3), marker=dict(size=9)))
    fig.add_hline(y=C.ATTENDANCE_THRESHOLD, line_dash="dash", line_color=ui.PAL["bad"],
                  annotation_text="75% minimum", annotation_position="bottom right")
    fig.update_layout(title="Attendance in the first 8 weeks", xaxis=dict(title="week", dtick=1),
                      yaxis=dict(title="attendance (%)", range=[0, 100]), showlegend=False)
    st.plotly_chart(ui.style(fig, 290), key=f"{PAGE}_line")

t1, t2 = st.tabs(["🔍 Why this result?", "🎯 How accurate is it?"])
with t1:
    occ = b.occlusion_groups(seq, static)

    def nice(f):
        m = re.match(r"(\w+) \(all weeks\)", f)
        if m:
            return f"{G.label(m.group(1))}, all {b.seq_len} weeks"
        return f"{G.label(f)} = {G.format_value(f, sv.get(f, ''))}"

    names = [nice(f) for f in occ["feature"]]
    vals = occ["effect"].to_numpy(float)
    for r in ui.reasons(np.array(names), vals, 100, "percentage points", subject="the risk"):
        st.markdown(f"- {r}")
    st.plotly_chart(ui.attribution_chart(names, vals, "Effect on the chance of falling below 75%", 100, "pts", top=8),
                    key=f"{PAGE}_occ")
    ui.callout("<b>How to read this:</b> we replace one whole signal at a time (for example all 8 weeks of attendance) by the average student's "
               "values and see how much the prediction changes. Red = this value pushes the risk up; green = it lowers it. "
               "")
with t2:
    ui.callout("<b>Read with care:</b> these scores come from <b>simulated</b> students. They show that the method works "
               "when attendance really does predict later attendance - not how accurate it would be at a real college.", "warn")
    m = load_json(C.REPORTS_DIR / "metrics_attendance.json")
    if m:
        t = m["test"]["BiLSTM + static (deep learning)"]
        a = st.columns(3)
        a[0].markdown(ui.kpi("At-risk students caught", f"{t['recall']:.0%}", "of the simulated students who ended below 75%",
                             "warn"), unsafe_allow_html=True)
        a[1].markdown(ui.kpi("Flags that were right", f"{t['precision']:.0%}", "of flagged students really ended below 75%",
                             "warn"), unsafe_allow_html=True)
        a[2].markdown(ui.kpi("Ranking quality", f"{t['roc_auc']:.2f}", "ROC-AUC (0.5 coin toss, 1.0 perfect)", "warn"),
                      unsafe_allow_html=True)
        rows = {k: v["roc_auc"] for k, v in m["test"].items()}
        order = sorted(rows, key=rows.get, reverse=True)
        fig = go.Figure(go.Bar(x=[rows[k] for k in order], y=order, orientation="h",
                               marker_color=[ui.PAL["primary"] if "BiLSTM" in k else ui.PAL["line"] for k in order],
                               text=[f"{rows[k]:.3f}" for k in order], textposition="outside", cliponaxis=False))
        fig.update_layout(title="Ranking quality (ROC-AUC) by model - simulated data", yaxis=dict(autorange="reversed"),
                          xaxis=dict(range=[0.5, 1.0]))
        st.plotly_chart(ui.style(fig, 320), key=f"{PAGE}_acc")
        st.caption("All models are close: in the simulation, early attendance predicts final attendance directly. "
                   "The BiLSTM becomes more valuable with real, irregular data.")
