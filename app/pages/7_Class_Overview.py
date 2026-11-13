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
from smartedu.risk import band  # noqa: E402

PAGE = "cls"
ui.page_setup("Class overview", "👥",
              "Score a whole class at once: who needs a conversation first, and which groups of students are most at risk.",
              chips=["Many students at once", "Compare groups", "Upload your own class"])
ui.require("dropout_start", "dropout_sem1", "dropout_sem2", "performance_start", "performance_sem1")

GROUPS = {"Course / programme": "Course", "Tuition fees paid up to date": "Tuition fees up to date",
          "Has unpaid dues": "Debtor", "Receives a scholarship": "Scholarship holder",
          "Class timing": "Daytime/evening attendance", "Living away from home": "Displaced",
          "Admission route": "Application mode"}
STAGE_LABELS = {k: C.STAGES[k]["label"] for k in ("start", "sem1", "sem2")}


@st.cache_data(show_spinner="Scoring students ...")
def scored(stage: str) -> pd.DataFrame:
    t = E.test_cohort()
    return pd.concat([t.reset_index(drop=True), E.predict_students(t, stage).reset_index(drop=True)], axis=1)


def kpis(d: pd.DataFrame, has_backlog: bool):
    k = st.columns(5)
    high = int((d["overall_risk"] >= 0.66).sum())
    k[0].markdown(ui.kpi("Students", f"{len(d)}", "in this selection", "info"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("High overall risk", f"{high}", f"{high / max(len(d), 1):.0%} of the selection",
                         "bad" if high else "good"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Average dropout chance", f"{d['p_drop'].mean():.0%}", "predicted by the model", "warn"),
                  unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Likely to drop out", f"{int((d['p_drop'] >= 0.5).sum())}", "students with a chance of 50% or more",
                         "bad"), unsafe_allow_html=True)
    k[4].markdown(ui.kpi("Average backlog", f"{d['backlog'].mean():.1f}" if has_backlog else "n/a",
                         "subjects expected uncleared" if has_backlog else "semester-2 results already known",
                         "warn"), unsafe_allow_html=True)


tab1, tab2 = st.tabs(["🎓 Sample class (real students)", "📤 Upload your own class"])

# ----------------------------------------------------------------------------------- sample class
with tab1:
    ui.callout("These are <b>664 real students the model never saw while learning</b>, so what you see here is an honest test. "
               "We also know what really happened to each of them, so you can check the predictions against reality.")
    stage = st.segmented_control("What do we know about these students?", list(STAGE_LABELS), default="sem1",
                                 format_func=lambda k: STAGE_LABELS[k], key=f"{PAGE}:stage") or "sem1"
    data = scored(stage)
    f = st.columns(4)
    courses = sorted(data["Course"].unique(), key=lambda c: G.COURSES.get(int(c), str(c)))
    sel_c = f[0].multiselect("Course", courses, format_func=lambda c: G.format_value("Course", c),
                             placeholder="All courses", key=f"{PAGE}:course")
    fees = f[1].selectbox("Fees", ["All", "Paid up to date", "Overdue"], key=f"{PAGE}:fees")
    sch = f[2].selectbox("Scholarship", ["All", "Has scholarship", "No scholarship"], key=f"{PAGE}:sch")
    tm = f[3].selectbox("Class timing", ["All", "Day", "Evening"], key=f"{PAGE}:tm")
    d = data
    if sel_c:
        d = d[d["Course"].isin(sel_c)]
    if fees != "All":
        d = d[d["Tuition fees up to date"] == (1 if fees.startswith("Paid") else 0)]
    if sch != "All":
        d = d[d["Scholarship holder"] == (1 if sch.startswith("Has") else 0)]
    if tm != "All":
        d = d[d["Daytime/evening attendance"] == (1 if tm == "Day" else 0)]
    if d.empty:
        st.warning("No students match these filters. Clear a filter to see results.")
        st.stop()
    has_backlog = "backlog" in d
    kpis(d, has_backlog)

    actual = float((d[C.DROPOUT_TARGET] == "Dropout").mean())
    st.markdown(
        f"<div class='callout good'><b>Reality check:</b> the model expected <b>{d['p_drop'].mean():.0%}</b> of this selection to drop out; "
        f"in reality <b>{actual:.0%}</b> did.</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2, gap="large")
    with g1:
        bands = d["overall_risk"].map(band).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        fig = go.Figure(go.Bar(x=list(bands.index), y=list(bands.values),
                               marker_color=[ui.BAND_COLOR[b] for b in bands.index],
                               text=list(bands.values), textposition="outside"))
        fig.update_layout(title="Students by overall risk", yaxis_title="students")
        st.plotly_chart(ui.style(fig, 320), key=f"{PAGE}_bands")
    with g2:
        fig = go.Figure(go.Histogram(x=100 * d["p_drop"], nbinsx=20, marker_color=ui.PAL["tint"],
                                     marker_line=dict(color=ui.PAL["primary"], width=1)))
        fig.update_layout(title="Spread of predicted dropout chance", xaxis_title="chance of dropping out (%)",
                          yaxis_title="students")
        st.plotly_chart(ui.style(fig, 320), key=f"{PAGE}_hist")

    st.markdown("#### Compare groups")
    gname = st.selectbox("Compare students by ...", list(GROUPS), key=f"{PAGE}:grp")
    col = GROUPS[gname]
    gdf = (d.assign(_d=(d[C.DROPOUT_TARGET] == "Dropout").astype(float))
           .groupby(col).agg(n=("p_drop", "size"), predicted=("p_drop", "mean"), real=("_d", "mean")).reset_index())
    gdf = gdf[gdf["n"] >= 5].sort_values("predicted", ascending=False)
    if gdf.empty:
        st.info("Groups here are too small to compare (fewer than 5 students). Widen the filters.")
    else:
        names = [G.format_value(col, v) for v in gdf[col]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=100 * gdf["predicted"], name="Predicted by the model", marker_color=ui.PAL["primary"]))
        fig.add_trace(go.Bar(x=names, y=100 * gdf["real"], name="What really happened", marker_color=ui.PAL["accent"],
                             text=[f"n={n}" for n in gdf["n"]], textposition="outside"))
        fig.update_layout(barmode="group", title=f"Dropout rate by {gname.lower()} (groups of 5+ students)",
                          yaxis=dict(title="dropout rate (%)", range=[0, 100]), legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(ui.style(fig, 380), key=f"{PAGE}_cmp")
        st.caption("Where the teal and yellow bars are close, the model is well calibrated for that group. "
                   "Association is not cause: a group can look riskier because of other things it has in common.")

    st.markdown("#### Students to talk to first")
    show = d.sort_values("overall_risk", ascending=False).head(15).copy()
    show.insert(0, "Student #", show.index + 1)
    cols = ["Student #", "Course", "Age at enrollment", "Tuition fees up to date", "Scholarship holder"]
    if has_backlog:
        cols += [C.PERFORMANCE_REGISTERED]
    table = G.display_frame(show[cols])
    table["Chance of dropping out"] = (100 * show["p_drop"]).round(0).astype(int).astype(str) + "%"
    if has_backlog:
        table["Expected backlog"] = show["backlog"].round(1)
    table["Overall risk"] = (100 * show["overall_risk"]).round(0).astype(int).astype(str) + "%"
    table["What really happened"] = show[C.DROPOUT_TARGET].map({"Dropout": "Dropped out", "Enrolled": "Still studying",
                                                                "Graduate": "Graduated"}).to_numpy()
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("⬇️ Download this selection (CSV)", table.to_csv(index=False).encode(), "class_overview.csv")

# ----------------------------------------------------------------------------------- upload
with tab2:
    st.markdown("#### 1. Get the template")
    stage_u = st.segmented_control("Prediction stage for your file", list(STAGE_LABELS), default="sem1",
                                   format_func=lambda k: STAGE_LABELS[k], key=f"{PAGE}:ustage") or "sem1"
    cols_needed = [c for c in C.STAGES[stage_u]["cols"]]
    tmpl = G.display_frame(E.test_cohort()[cols_needed].head(5))
    st.dataframe(tmpl, hide_index=True, width="stretch")
    st.download_button("⬇️ Download this template (CSV)", tmpl.to_csv(index=False).encode(), f"class_template_{stage_u}.csv")
    st.caption("Use plain-English headings and values exactly as in the template (for example 'Informatics Engineering', "
               "'Yes' / 'No'). Technical column names from the original dataset also work. Missing columns are filled with typical values.")
    st.markdown("#### 2. Upload your class")
    up = st.file_uploader("CSV file", type=["csv"], key=f"{PAGE}:upload")
    if up is not None:
        raw = pd.read_csv(up, sep=None, engine="python", encoding="utf-8-sig")
        raw.columns = [str(c).strip() for c in raw.columns]
        df = G.friendly_to_raw(raw)
        have = [c for c in cols_needed if c in df.columns]
        miss = [c for c in cols_needed if c not in df.columns]
        st.markdown(f"Found **{len(have)} of {len(cols_needed)}** fields for this stage"
                    + (f"; filled with typical values: {', '.join(G.label(c) for c in miss)}." if miss else "."))
        bad = [G.label(c) for c in ("Course", "Application mode", "Previous qualification") if c in df.columns
               and df[c].isna().any()]
        if bad:
            st.warning("Some values in " + ", ".join(bad) + " were not recognised and are treated as unknown. "
                       "Check the spelling against the template.")
        pr = E.predict_students(df, stage_u)
        res = pd.concat([raw.reset_index(drop=True), pr.reset_index(drop=True)], axis=1)
        kpis(pr.assign(overall_risk=pr["overall_risk"]), "backlog" in pr)
        out = raw.copy()
        out["Chance of dropping out"] = (100 * pr["p_drop"]).round(0).astype(int).astype(str) + "%"
        if "backlog" in pr:
            out["Expected backlog"] = pr["backlog"].round(1)
        out["Overall risk"] = (100 * pr["overall_risk"]).round(0).astype(int).astype(str) + "%"
        out["Risk band"] = pr["overall_risk"].map(band)
        st.dataframe(out.assign(_s=pr["overall_risk"]).sort_values("_s", ascending=False).drop(columns="_s"),
                     hide_index=True, width="stretch")
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(), "class_predictions.csv",
                           type="primary")
