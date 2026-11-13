import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import about  # noqa: E402
from smartedu import config as C  # noqa: E402
from smartedu import engine as E  # noqa: E402
from smartedu import ui  # noqa: E402
from smartedu.sentiment import LABELS  # noqa: E402
from smartedu.utils import load_json  # noqa: E402

PAGE = "mood"
M = about.by_key("sentiment")
ui.page_setup("Feedback mood", "💬", M["question"],
              chips=["Model D", "Neural network for text (BiLSTM)", "Real course reviews"])
ui.require("sentiment")
with st.expander("What this model is, and what data it learned from", expanded=False):
    ui.module_strip(M)

pred = ui.get_sentiment()
st.caption(f"Model in use: **{pred.name}**.")

EXAMPLES = {
    "Happy": "The professor explained every topic clearly and the assignments were really helpful. Loved this course!",
    "Mixed": "The lectures were good but the exams felt too rushed and the labs were not well organised.",
    "Unhappy": "Boring lectures, no feedback on assignments, and the syllabus is outdated. A waste of my fees.",
}
MOOD_WORD = {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}


def _set(txt):
    st.session_state[f"{PAGE}:text"] = txt


tab1, tab2 = st.tabs(["✍️ One comment", "📋 Many comments"])
with tab1:
    st.markdown("Try an example, or type your own:")
    ec = st.columns(len(EXAMPLES) + 1)
    for col, (k, v) in zip(ec, EXAMPLES.items()):
        col.button(k, key=f"{PAGE}:ex:{k}", on_click=_set, args=(v,), width="stretch")
    if f"{PAGE}:text" not in st.session_state:
        st.session_state[f"{PAGE}:text"] = EXAMPLES["Mixed"]
    text = st.text_area("What the student wrote", key=f"{PAGE}:text", height=110,
                        help="Any feedback in English: about a course, a teacher, the labs, the exams ...")
    if text.strip():
        pr = pred.predict_proba([text])[0]
        label, toks, eff = pred.explain_tokens(text)
        c = st.columns([1, 1.4], gap="large")
        with c[0]:
            tone = {"Negative": "bad", "Neutral": "warn", "Positive": "good"}[label]
            st.markdown(ui.kpi("The comment sounds", MOOD_WORD[label], f"{100 * pr.max():.0f}% sure", tone),
                        unsafe_allow_html=True)
            st.plotly_chart(ui.prob_bar(LABELS, pr, "", height=190), key=f"{PAGE}_p")
        with c[1]:
            st.markdown("**Which words decided it?**")
            st.markdown(ui.highlight_words(toks, eff), unsafe_allow_html=True)
            st.caption(f"Green words push towards '{MOOD_WORD[label]}', red words push against it. "
                       "We remove each word in turn and see how the model's confidence changes.")
        ui.callout("A <b>mixed</b> comment often lands on 'neutral' or splits its probability - that is the model "
                   "being honest, not broken. The mood model reads only the words, not the student's marks.")
with tab2:
    st.markdown("Paste several comments (one per line) or upload a CSV with a column of comments.")
    many = st.text_area("Comments, one per line", height=150, key=f"{PAGE}:many",
                        value="\n".join(EXAMPLES.values()) + "\nGreat teaching and friendly staff.\nNot enough practical sessions.")
    up = st.file_uploader("...or upload a CSV", type=["csv"], key=f"{PAGE}:up")
    lines = [x.strip() for x in many.splitlines() if x.strip()]
    if up is not None:
        df_up = pd.read_csv(up, sep=None, engine="python", encoding="utf-8-sig")
        col = st.selectbox("Which column holds the comments?", list(df_up.columns))
        lines = df_up[col].dropna().astype(str).tolist()
    if lines:
        P = pred.predict_proba(lines)
        out = pd.DataFrame({"Comment": lines, "Mood": [MOOD_WORD[LABELS[i]] for i in P.argmax(1)],
                            "Negative %": (100 * P[:, 0]).round(0), "Neutral %": (100 * P[:, 1]).round(0),
                            "Positive %": (100 * P[:, 2]).round(0)})
        counts = out["Mood"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
        k = st.columns(4)
        k[0].markdown(ui.kpi("Comments", f"{len(out)}", "analysed", "info"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Negative", f"{counts['negative']}", f"{counts['negative'] / len(out):.0%} of comments", "bad"),
                      unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Neutral", f"{counts['neutral']}", f"{counts['neutral'] / len(out):.0%} of comments", "warn"),
                      unsafe_allow_html=True)
        k[3].markdown(ui.kpi("Positive", f"{counts['positive']}", f"{counts['positive'] / len(out):.0%} of comments", "good"),
                      unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=list(counts.index), y=list(counts.values),
                               marker_color=[ui.PAL["bad"], ui.PAL["warn"], ui.PAL["good"]],
                               text=list(counts.values), textposition="outside"))
        fig.update_layout(title="How many comments in each mood", yaxis_title="comments")
        st.plotly_chart(ui.style(fig, 290), key=f"{PAGE}_counts")
        st.markdown("**Most negative comments first** - a good list to read first:")
        st.dataframe(out.sort_values("Negative %", ascending=False), hide_index=True, width="stretch")
        st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode(), "feedback_mood.csv")

with st.expander("🎯 How accurate is it?"):
    m = load_json(C.REPORTS_DIR / "metrics_sentiment.json")
    bert = load_json(C.REPORTS_DIR / "metrics_sentiment_distilbert.json")
    if m:
        rows = {k: v["macro_f1"] for k, v in m["test"].items()}
        if bert:
            rows.update({k: v["macro_f1"] for k, v in bert["test"].items()})
        a = st.columns(len(rows))
        for col, (k, v) in zip(a, rows.items()):
            col.markdown(ui.kpi(k, f"{v:.2f}", "balanced score (macro-F1) on 1,800 unseen reviews", "info"),
                         unsafe_allow_html=True)
        per = m["test"]["Embedding + BiLSTM (deep learning)"]["per_class"]
        st.markdown(f"Per mood (F1): negative **{per['Negative']['f1']:.2f}**, neutral **{per['Neutral']['f1']:.2f}**, "
                    f"positive **{per['Positive']['f1']:.2f}**.")
        ui.callout("'Neutral' is hardest: a 3-star review usually contains both praise and complaints. The simple TF-IDF "
                   "baseline ties the BiLSTM with only 12,000 reviews; a pretrained model (DistilBERT, see the Colab notebook) "
                   "is the usual way to do better.")
        st.caption("Labels come from star ratings (1-2 negative, 3 neutral, 4-5 positive) - a rough stand-in for real sentiment. "
                   "Test reviews are balanced (equal numbers per mood).")
