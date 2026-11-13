import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import config as C  # noqa: E402
from smartedu import ui  # noqa: E402
from smartedu.utils import load_json  # noqa: E402

PAGE = "acc"
ui.page_setup("Accuracy", "🎯",
              "How good is each model - measured only on students it never saw while learning - and where simpler models "
              "do just as well.",
              chips=["Every model compared", "Honest results", "Deep learning vs simpler models"])

R = C.REPORTS_DIR
perf, drop = load_json(R / "metrics_performance.json"), load_json(R / "metrics_dropout.json")
att, sen = load_json(R / "metrics_attendance.json"), load_json(R / "metrics_sentiment.json")
bert = load_json(R / "metrics_sentiment_distilbert.json")
DEEP = ("MLP (deep learning)", "BiLSTM + static (deep learning)", "Embedding + BiLSTM (deep learning)", "DistilBERT (fine-tuned)")


def bars(models, values, title, axis, lower_is_better=False, fmt="{:.2f}", rng=None, key=""):
    """Horizontal bars; deep-learning models in teal, the rest in grey."""
    order = sorted(range(len(models)), key=lambda i: values[i], reverse=not lower_is_better)
    m = [models[i] for i in order][::-1]
    v = [values[i] for i in order][::-1]
    fig = go.Figure(go.Bar(x=v, y=[x.replace(" (deep learning)", " (deep learning)") for x in m], orientation="h",
                           marker_color=[ui.PAL["primary"] if x in DEEP else "#B8C0D0" for x in m],
                           text=[fmt.format(a) for a in v], textposition="outside", cliponaxis=False))
    fig.update_layout(title=title, xaxis=dict(title=axis, range=rng))
    st.plotly_chart(ui.style(fig, 62 * len(m) + 100), key=f"{PAGE}_{key}")


def image(name: str, caption: str):
    p = R / "figures" / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


st.markdown(
    "**How to read this page.** Every model is trained on one group of students and then tested on a separate group it has "
    "never seen. <span style='color:#0E7C86'><b>Teal bars = deep learning</b></span>; grey bars = simpler classical models used "
    "as a fair comparison.", unsafe_allow_html=True)
tabs = st.tabs(["📘 A. Subject forecast", "🚪 B. Dropout warning", "🗓️ C. Attendance watch", "💬 D. Feedback mood"])

# ---------------------------------------------------------------------------------------- A
with tabs[0]:
    if not perf:
        st.info("Not trained yet.")
    else:
        stage = st.radio("How much is known?", list(perf["settings"]), horizontal=True, key=f"{PAGE}_a_s",
                         format_func=lambda s: perf["settings"][s]["label"])
        d = perf["settings"][stage]
        t = d["test"]
        deep, naive = t["MLP (deep learning)"], t["Typical pass rate (no model)"]
        k = st.columns(4)
        k[0].markdown(ui.kpi("Typical error (neural network)", f"±{deep['rmse']:.2f} subjects", f"measured on {d['n_test']} unseen students", "good"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Guessing the average", f"±{naive['rmse']:.2f} subjects", "what you get with no model at all", "warn"), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Within 1 subject", f"{deep['within_1_subject']:.0%}", "of forecasts off by at most one subject", "info"), unsafe_allow_html=True)
        k[3].markdown(ui.kpi("Explained", f"{deep['r2']:.0%}", "R² - share of the differences between students captured", "info"), unsafe_allow_html=True)
        c = st.columns([1, 1.15])
        with c[0]:
            bars(list(t), [v["rmse"] for v in t.values()], "Typical error in subjects (lower is better)", "error (subjects)",
                 lower_is_better=True, key="a_bar")
        with c[1]:
            df = pd.DataFrame([{"Model": m, "Error": round(v["rmse"], 2), "Within ±1": f"{v['within_1_subject']:.0%}",
                                "5-fold check": f"{d['cv'][m]['rmse_mean']:.2f} ± {d['cv'][m]['rmse_std']:.2f}"}
                               for m, v in t.items()])
            st.dataframe(df, hide_index=True, width="stretch")
            st.caption(f"Tested on {d['n_test']} unseen students. The 5-fold check repeats the test five times on different groups to show it is not luck.")
        ui.callout("<b>What this says:</b> before any exam, background alone only improves a little on guessing the average. "
                   "After semester 1, the error roughly halves, because the best predictor of next semester is this semester. "
                   "The neural network and the simpler models land almost together - <i>knowing more about the student</i> helps far more than a fancier model.")
        c2 = st.columns(2)
        with c2[0]:
            image(f"performance_{stage}_pred_vs_actual.png", "Forecast vs reality (closer to the diagonal is better)")
        with c2[1]:
            image(f"performance_{stage}_shap_global.png", "Which facts matter most (average effect on the forecast)")

# ---------------------------------------------------------------------------------------- B
with tabs[1]:
    if not drop:
        st.info("Not trained yet.")
    else:
        stage = st.radio("How much is known?", list(drop["settings"]), horizontal=True, key=f"{PAGE}_b_s",
                         format_func=lambda s: drop["settings"][s]["label"])
        d = drop["settings"][stage]
        t = d["test"]
        deep = t["MLP (deep learning)"]
        k = st.columns(4)
        k[0].markdown(ui.kpi("Real dropouts caught", f"{deep['dropout_recall']:.0%}", "of students who really dropped out were flagged", "good"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Flags that were right", f"{deep['dropout_precision']:.0%}", "of flagged students really dropped out", "info"), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Ranking quality", f"{deep['roc_auc']:.2f}", "ROC-AUC: 0.5 = coin toss, 1.0 = perfect", "info"), unsafe_allow_html=True)
        k[3].markdown(ui.kpi("Right overall", f"{deep['accuracy']:.0%}", "of all 3-way outcomes (drop / studying / graduate)", "info"), unsafe_allow_html=True)
        c = st.columns(2)
        with c[0]:
            bars(list(t), [v["roc_auc"] for v in t.values()], "Ranking quality - ROC-AUC (higher is better)", "ROC-AUC",
                 fmt="{:.3f}", rng=[0.5, 1.0], key="b_auc")
        with c[1]:
            bars(list(t), [v["dropout_recall"] for v in t.values()], "Share of real dropouts caught (higher is better)", "recall",
                 fmt="{:.0%}", rng=[0, 1], key="b_rec")
        df = pd.DataFrame([{"Model": m, "Right overall": f"{v['accuracy']:.0%}", "Dropouts caught": f"{v['dropout_recall']:.0%}",
                            "Flags that were right": f"{v['dropout_precision']:.0%}", "ROC-AUC": round(v["roc_auc"], 3)} for m, v in t.items()])
        st.dataframe(df, hide_index=True, width="stretch")
        pc = d["full_mlp_metrics"]["per_class"]
        st.caption("Per outcome (neural network), F1 score: " + ", ".join(
            f"{ {'Dropout': 'drops out', 'Enrolled': 'still studying', 'Graduate': 'graduates'}[k_] } {v['f1']:.2f}" for k_, v in pc.items())
                   + ". 'Still studying' is the hardest group because those students could still go either way.")
        ui.callout("<b>What this says:</b> the same story as module A. Each extra semester of results sharpens the warning; the neural network ties "
                   "the simpler models. This matches published research that tree models and neural networks perform similarly on table-like data. "
                   "The warning after semester 1 is the useful one - there is still time to help.")
        c2 = st.columns(2)
        with c2[0]:
            image(f"dropout_{stage}_confusion_mlp.png", "Confusion matrix: rows = what really happened, columns = what the network said")
        with c2[1]:
            image(f"dropout_{stage}_shap_global.png", "Which facts push the dropout chance up or down the most")

# ---------------------------------------------------------------------------------------- C
with tabs[2]:
    if not att:
        st.info("Not trained yet.")
    else:
        ui.callout("<b>These numbers are on SIMULATED students.</b> No public attendance data exists, so the model learned from students generated by "
                   "written rules (see 'How it works'). The scores prove the method and the code work - they say nothing certain about your college.", "warn")
        t = att["test"]
        bars(list(t), [v["roc_auc"] for v in t.values()], "Ranking quality - ROC-AUC on simulated data (higher is better)", "ROC-AUC",
             fmt="{:.3f}", rng=[0.8, 1.0], key="c_auc")
        df = pd.DataFrame([{"Model": m, "At-risk caught": f"{v['recall']:.0%}", "Flags that were right": f"{v['precision']:.0%}",
                            "F1": round(v["f1"], 3), "ROC-AUC": round(v["roc_auc"], 3)} for m, v in t.items()])
        st.dataframe(df, hide_index=True, width="stretch")
        ui.callout("All models are equal here because in the simulation early attendance predicts final attendance by construction. "
                   "A BiLSTM is the right <i>kind</i> of network for weekly data, but only real attendance logs could show whether it beats simpler models.")
        c2 = st.columns(2)
        with c2[0]:
            image("attendance_confusion_bilstm.png", "Confusion matrix (simulated test students)")
        with c2[1]:
            image("attendance_occlusion_importance.png", "Which weeks and signals matter most")

# ---------------------------------------------------------------------------------------- D
with tabs[3]:
    if not sen:
        st.info("Not trained yet.")
    else:
        t = dict(sen["test"])
        if bert:
            t.update(bert["test"])
        bars(list(t), [v["macro_f1"] for v in t.values()], "Macro-F1 on 1,800 unseen reviews (higher is better)", "macro-F1",
             fmt="{:.3f}", rng=[0.4, 1.0], key="d_f1")
        df = pd.DataFrame([{"Model": m, "Right overall": f"{v['accuracy']:.1%}", "Macro-F1": round(v["macro_f1"], 3)} for m, v in t.items()])
        st.dataframe(df, hide_index=True, width="stretch")
        pc = sen["test"]["Embedding + BiLSTM (deep learning)"]["per_class"]
        st.caption("BiLSTM F1 per mood: " + ", ".join(f"{k_.lower()} {v['f1']:.2f}" for k_, v in pc.items())
                   + ". Neutral (3-star) reviews are hardest: they often mix praise and complaint.")
        if not bert:
            ui.callout("<b>DistilBERT result not added yet.</b> A pretrained language model is the one place where deep learning is expected to clearly "
                       "beat the simple baseline. Run <code>colab_distilbert.ipynb</code> on a free Colab GPU, unzip the result into this folder and the "
                       "row appears here automatically.", "warn")
        ui.callout("<b>What this says:</b> a BiLSTM trained from scratch on 12,000 short reviews only ties a strong word-counting baseline (TF-IDF). "
                   "That is expected - small text data is exactly why pretrained models exist.")
        tw = sen.get("top_words_baseline", {})
        if tw:
            st.markdown("**Words the baseline relies on most**")
            cc = st.columns(3)
            for col, (mood, words) in zip(cc, tw.items()):
                col.markdown(ui.card(mood, ", ".join(words[:10])), unsafe_allow_html=True)
        c2 = st.columns(2)
        with c2[0]:
            image("sentiment_confusion_bilstm.png", "Confusion matrix (BiLSTM)")
        with c2[1]:
            image("sentiment_training.png", "How the BiLSTM learned")
