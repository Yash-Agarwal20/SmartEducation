import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import about  # noqa: E402
from smartedu import config as C  # noqa: E402
from smartedu import engine as E  # noqa: E402
from smartedu import glossary as G  # noqa: E402
from smartedu import synthetic as S  # noqa: E402
from smartedu import ui  # noqa: E402

ui.page_setup("How it works", "📖",
              "What goes in, what the models learned from, and what comes out - in plain words, with the real data to look at.",
              chips=["No jargon", "Real data samples", "Every field explained"])

tabs = st.tabs(["🗺️ The big picture", "🗃️ Which data trains which model", "🔤 What every field means",
                "🚫 What we left out", "⚠️ Honest limits"])

# ------------------------------------------------------------------------------------ big picture
with tabs[0]:
    st.markdown("### From a student record to a warning")
    st.markdown(ui.flow(about.JOURNEY), unsafe_allow_html=True)
    c = st.columns(2, gap="large")
    with c[0]:
        st.markdown("#### What is a neural network, in one minute?")
        st.markdown(
            "- Think of a **panel of tiny calculators** arranged in layers. Each one takes some numbers, multiplies each by a "
            "weight (how much it cares), adds them up and passes the result on.\n"
            "- At the start the weights are random, so the answers are nonsense.\n"
            "- We show it **thousands of past students whose outcome we already know**. After each batch it nudges every weight "
            "a little so that its answers get closer to the truth. This is called *training*.\n"
            "- After many rounds, the weights capture patterns such as *'students who clear few subjects in semester 1 and "
            "have overdue fees often leave'*.\n"
            "- To predict a **new** student, we only run the numbers forward through the layers - no learning happens.")
        st.page_link("pages/8_Inside_the_Network.py", label="See the real network, layer by layer", icon="🧠")
    with c[1]:
        st.markdown("#### Why four models instead of one?")
        st.markdown(
            "Each question needs different data and a different shape of network:\n"
            "- **Table of facts about a student** (course, fees, marks) → an *MLP*, the standard network for tables.\n"
            "- **Something that unfolds week by week** (attendance) → a *BiLSTM*, a network that reads in order and remembers.\n"
            "- **Free text** (comments) → an *embedding + BiLSTM*, which first turns words into numbers.\n\n"
            "The **Student 360** page then combines the answers into one risk card, with weights you can change.")
    st.markdown("#### What does *stage* mean on the prediction pages?")
    st.markdown("A forecast depends on **how much we already know**. The same student gets a rough guess before any exam and a much sharper one after semester 1:")
    sc = st.columns(3)
    for col, k in zip(sc, ["start", "sem1", "sem2"]):
        col.markdown(ui.card(C.STAGES[k]["label"], about.STAGE_EXPLAINER[k]), unsafe_allow_html=True)

# ------------------------------------------------------------------------------------ data per model
with tabs[1]:
    st.markdown("### One card per model: the question, the data, the input, the output")
    ui.callout("Two models (A and B) learn from the <b>same real university file</b>, with different questions. "
               "The attendance model learns from <b>simulated</b> students because no public attendance data exists. "
               "The feedback model learns from <b>real course reviews</b>.")
    for m in about.modules():
        with st.expander(f"{m['icon']}  Model {m['letter']} - {m['name']}:  {m['question']}", expanded=(m["letter"] == "A")):
            ui.module_strip(m)
            st.page_link(f"pages/{ {'performance': '3_Subject_Forecast', 'dropout': '4_Dropout_Warning', 'attendance': '5_Attendance_Watch', 'sentiment': '6_Feedback_Mood'}[m['key']] }.py",
                         label=f"Try model {m['letter']}", icon=m["icon"])

    st.markdown("### A peek at the actual training data")
    pk = st.radio("Which data?", ["University records (models A and B)", "Weekly attendance (model C, simulated)",
                                  "Course reviews (model D)"], horizontal=True)
    if pk.startswith("University"):
        df = E.cohort().sample(8, random_state=3)
        keep = ["Course", "Age at enrollment", "Tuition fees up to date", "Scholarship holder", C.SEM1_COLS[1],
                C.SEM1_COLS[3], C.SEM1_COLS[4], C.SEM2_REGISTRATION_COLS[1], C.SEM2_RESULT_COLS[1], C.DROPOUT_TARGET]
        st.dataframe(G.display_frame(df[keep]), hide_index=True, width="stretch")
        st.caption("Each row is one real student. The last column is what actually happened to them - the answer the "
                   "dropout model learns to predict. 'Subjects cleared in semester 2' is what model A learns to predict.")
    elif pk.startswith("Weekly"):
        static, seq, att, y = S.make_attendance(300, seed=3)
        idx = list(range(6))
        rows = []
        for i in idx:
            r = {"Distance to campus (km)": static["commute_km"].iloc[i], "Part-time job": "Yes" if static["works_part_time"].iloc[i] else "No"}
            r.update({f"Wk {w + 1}": round(float(seq[i, w, 0])) for w in range(C.OBSERVED_WEEKS)})
            r["Final attendance"] = f"{att[i].mean():.0f}%"
            r["Outcome"] = "At-Risk (<75%)" if y[i] else "Safe"
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("SIMULATED. Weekly columns show attendance % in the first 8 weeks - all the model may look at. "
                   "'Final attendance' (all 16 weeks) decides the outcome it learns to predict early.")
    else:
        fb = pd.read_csv(C.RAW_DIR / "feedback.csv").sample(6, random_state=5) if (C.RAW_DIR / "feedback.csv").exists() \
            else S.make_feedback(6)
        show = fb.rename(columns={"text": "Review", "label": "Mood label", "stars": "Stars given"})
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("Real Coursera reviews. The mood label comes from the star rating; the model reads only the text.")

# ------------------------------------------------------------------------------------ glossary
with tabs[2]:
    st.markdown("### Every field, in plain language")
    q = st.text_input("Search a field or word", placeholder="e.g. fees, grade, attendance")
    gt = G.glossary_table()
    if q:
        mask = gt.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)
        gt = gt[mask]
    st.dataframe(gt, hide_index=True, width="stretch", height=460,
                 column_config={"What it means": st.column_config.TextColumn(width="large"),
                                "Raw column name": st.column_config.TextColumn("Name in the original file")})
    st.caption("Grades are on a 0-20 scale where 10 is the pass mark; multiply by 5 for a rough percentage. "
               "The original dataset is Portuguese, which is why the scale is 0-20.")

# ------------------------------------------------------------------------------------ excluded
with tabs[3]:
    st.markdown("### Columns in the raw data that the system deliberately does not use")
    st.dataframe(pd.DataFrame([{"Columns": k, "Why left out": v} for k, v in G.EXCLUDED_AND_WHY.items()]),
                 hide_index=True, width="stretch",
                 column_config={"Why left out": st.column_config.TextColumn(width="large")})
    ui.callout("A risk tool should not judge a student by gender, nationality or their parents' jobs. We tested it (5-fold cross-validation, "
               "after-semester-1 stage): adding these columns back changed accuracy by between -0.6 and +0.2 percentage points, "
               "so nothing useful is lost by leaving them out.", "good")

# ------------------------------------------------------------------------------------ limits
with tabs[4]:
    st.markdown("### Read this before trusting any number")
    for lim in about.LIMITS:
        st.markdown(f"- {lim}")
