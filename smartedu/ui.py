"""Streamlit toolkit shared by every dashboard page: theme, cards, charts and the student form.

Everything a person reads goes through ``glossary.py`` (plain-language names), and everything that
predicts goes through ``engine.py`` (no Streamlit code), so the pages stay short.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config as C
from . import engine as E
from . import glossary as G
from .utils import load_json

PAL = dict(primary="#0E7C86", dark="#14213D", accent="#FFD93D", good="#16A34A", warn="#D97706",
           bad="#DC2626", ink="#14213D", muted="#5B6579", soft="#F4F6FA", line="#DDE2EC", tint="#D6ECEE")
BAND_COLOR = {"Low": PAL["good"], "Medium": PAL["warn"], "High": PAL["bad"], "n/a": PAL["muted"]}
FONT = "Figtree, 'Segoe UI', system-ui, sans-serif"

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="st-"] {font-family:'Figtree','Segoe UI',system-ui,sans-serif}
[data-testid="stIconMaterial"], [class*="material-icons"], [class*="material-symbols"], span[translate="no"]{
font-family:'Material Symbols Rounded','Material Icons' !important}
h1,h2,h3,h4,.hero h1,.kpi .v,.big{font-family:'Bricolage Grotesque','Figtree','Segoe UI',sans-serif;letter-spacing:-.015em;color:#14213D}
.block-container{padding-top:1.4rem;max-width:1240px}
[data-testid="stSidebar"]{border-right:1px solid #DDE2EC}
.stAppDeployButton{display:none}
.hero{display:flex;gap:20px;align-items:center;padding:24px 30px;border-radius:6px 14px 14px 6px;margin-bottom:22px;
background:#14213D;border-left:10px solid #FFD93D}
.hero .ico{font-size:2.6rem;line-height:1}
.hero h1{margin:0;padding:0;font-size:2rem;color:#fff}
.hero p{margin:.3rem 0 .7rem 0;font-size:1.05rem;color:#CBD3E6;max-width:62rem;line-height:1.5}
.chip{display:inline-block;margin:2px 8px 2px 0;padding:3px 12px;border-radius:99px;border:1px solid #5B6B93;color:#E8ECF7;font-size:.82rem;font-weight:600}
.card{border:1px solid #DDE2EC;border-radius:10px;padding:16px 18px;background:#fff;height:100%}
.card h4{margin:0 0 4px 0;font-size:1.05rem}
.card p{margin:0;color:#44506A;font-size:.93rem;line-height:1.5}
.card .tag{display:inline-block;margin-top:9px;font-size:.78rem;font-weight:600;color:#0B5F67;background:#D6ECEE;padding:2px 9px;border-radius:5px}
.kpi{border-radius:10px;padding:14px 16px;border:1px solid #DDE2EC;background:#fff;height:100%}
.kpi .l{font-size:.9rem;font-weight:600;color:#5B6579}
.kpi .v{font-size:clamp(1.25rem,2.1vw,2rem);font-weight:800;line-height:1.15;margin:2px 0;overflow-wrap:normal;word-break:keep-all}
.kpi .s{font-size:.88rem;color:#44506A;line-height:1.45}
.kpi.good{border-left:6px solid #16A34A}.kpi.warn{border-left:6px solid #D97706}.kpi.bad{border-left:6px solid #DC2626}.kpi.info{border-left:6px solid #0E7C86}
.callout{border-radius:8px;padding:12px 16px;background:#E8F4F5;border-left:5px solid #0E7C86;color:#173B40;font-size:.95rem;line-height:1.55;margin:6px 0 12px 0}
.callout.warn{background:#FFF6DB;border-left-color:#D9A400;color:#5E4700}
.callout.good{background:#E6F6EC;border-left-color:#16A34A;color:#0F4A27}
.pill{display:inline-block;padding:5px 14px;border-radius:99px;color:#fff;font-weight:700;font-size:.95rem}
.dots span{display:inline-block;width:24px;height:24px;border-radius:50%;margin:3px 5px 2px 0;border:3px solid #16A34A}
.dots span.on{background:#16A34A}.dots span.off{border-color:#DC2626;background:#FDECEC}
.mark{background:linear-gradient(transparent 58%,#FFD93D 58%);padding:0 3px}
.step{display:flex;gap:12px;align-items:flex-start;margin:8px 0}
.step .n{flex:0 0 34px;height:34px;border-radius:6px;background:#14213D;color:#fff;font-weight:800;display:flex;align-items:center;justify-content:center}
.step b{display:block}
.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:8px 0 14px 0}
.flow .box{flex:1 1 150px;border:1px solid #DDE2EC;background:#fff;border-radius:10px;padding:10px 12px;font-size:.9rem;line-height:1.4}
.flow .box b{display:block;margin-bottom:2px}
.flow .arrow{align-self:center;color:#5B6579;font-size:1.3rem}
.flow .box.data{border-top:5px solid #FFD93D}.flow .box.model{border-top:5px solid #0E7C86}.flow .box.out{border-top:5px solid #14213D}
.small{color:#5B6579;font-size:.86rem}
div[data-testid="stTabs"] button[role="tab"]{font-weight:600}
div[data-testid="stMetric"]{background:#fff;border:1px solid #DDE2EC;border-radius:10px;padding:10px 14px}
</style>"""


# ====================================================================== page frame
def page_setup(title: str, icon: str, subtitle: str = "", chips=()):
    st.set_page_config(page_title=f"{title} - Smart Education", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🎓 Smart Education\n<span class='small'>Early-warning assistant for college advisers</span>",
                    unsafe_allow_html=True)
    chips_html = "".join(f"<span class='chip'>{html.escape(c)}</span>" for c in chips)
    st.markdown(f"<div class='hero'><div class='ico'>{icon}</div><div><h1>{html.escape(title)}</h1>"
                f"<p>{html.escape(subtitle)}</p>{chips_html}</div></div>", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "info") -> str:
    return (f"<div class='kpi {tone}'><div class='l'>{html.escape(label)}</div>"
            f"<div class='v'>{html.escape(str(value))}</div><div class='s'>{sub}</div></div>")


def card(title: str, text: str, tag: str = "") -> str:
    t = f"<span class='tag'>{tag}</span>" if tag else ""
    return f"<div class='card'><h4>{title}</h4><p>{text}</p>{t}</div>"


def callout(text: str, kind: str = "info"):
    st.markdown(f"<div class='callout {kind if kind != 'info' else ''}'>{text}</div>", unsafe_allow_html=True)


def pill(band: str, text: str | None = None) -> str:
    return f"<span class='pill' style='background:{BAND_COLOR.get(band, PAL['muted'])}'>{text or band + ' risk'}</span>"


def subject_dots(cleared: float, registered: float) -> str:
    n, k = int(round(registered)), int(round(cleared))
    return "<div class='dots'>" + "".join(f"<span class='{'on' if i < k else 'off'}'></span>" for i in range(n)) + "</div>"


def band_of(score: float) -> str:
    from .risk import band
    return band(score)


# ==================================================================== data banner
def data_source_banner(modules=("performance", "dropout", "attendance", "sentiment")):
    srcs = {m: (load_json(C.REPORTS_DIR / f"metrics_{m}.json") or {}).get("source") for m in modules}
    synth = [k for k, v in srcs.items() if v == "synthetic"]
    if synth:
        st.warning("Models for **" + ", ".join(synth) + "** were trained on SYNTHETIC test data - demo only. "
                   "Retrain with `python train_all.py --source real`.")
    if srcs.get("attendance") == "simulated":
        callout("<b>Heads-up:</b> no public dataset of real attendance exists, so the attendance model learned from "
                "<b>simulated</b> students generated by documented rules. It shows how the method works, "
                "not how accurate it would be at your college.", "warn")


def require(*keys):
    miss = [k for k in keys if not E.available(k)]
    if miss:
        st.error(f"Trained model(s) not found: {', '.join(miss)}.")
        st.code("python train_all.py --source real", language="bash")
        st.stop()


# =========================================================================== charts
def style(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(font=dict(family=FONT, color=PAL["ink"]), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=44, b=10))
    if height:
        fig.update_layout(height=height)
    return fig


def gauge(value: float, title: str, height: int = 230) -> go.Figure:
    v = 100 * float(value)
    band = band_of(float(value))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={"suffix": "%", "valueformat": ".0f", "font": {"size": 40}},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0}, "bar": {"color": BAND_COLOR[band], "thickness": 0.32},
               "bgcolor": "#EEF1F6", "borderwidth": 0,
               "steps": [{"range": [0, 33], "color": "#E3F4EA"}, {"range": [33, 66], "color": "#FFF1CF"},
                         {"range": [66, 100], "color": "#FBE0E0"}]}))
    return style(fig, height)


def prob_bar(labels, probs, title="", colors=None, height=None) -> go.Figure:
    colors = colors or [PAL["bad"], PAL["warn"], PAL["good"]][:len(labels)]
    fig = go.Figure(go.Bar(x=[100 * float(p) for p in probs], y=list(labels), orientation="h", marker_color=colors,
                           text=[f"{100 * float(p):.0f}%" for p in probs], textposition="outside", cliponaxis=False))
    fig.update_layout(title=title, xaxis=dict(range=[0, 118], title="chance (%)", showgrid=False),
                      yaxis=dict(autorange="reversed"))
    return style(fig, height or 62 * len(labels) + 90)


def attribution_chart(names, values, title="", unit_scale=1.0, unit="", top=7, bad_is_positive=True) -> go.Figure:
    order = np.argsort(-np.abs(values))[:top][::-1]
    vals = [float(values[i]) * unit_scale for i in order]
    up, down = (PAL["bad"], PAL["good"]) if bad_is_positive else (PAL["good"], PAL["bad"])
    fig = go.Figure(go.Bar(x=vals, y=[names[i] for i in order], orientation="h",
                           marker_color=[up if v > 0 else down for v in vals],
                           text=[f"{v:+.1f} {unit}".strip() for v in vals], textposition="outside", cliponaxis=False))
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = 0.28 * max(hi - lo, 1e-9)
    fig.update_layout(title=title, xaxis=dict(zeroline=True, zerolinecolor="#9CA3AF", showgrid=False, range=[lo - pad, hi + pad],
                                              title=f"effect ({unit})" if unit else "effect"),
                      yaxis=dict(automargin=True))
    return style(fig, 38 * len(order) + 110)


@st.cache_data(show_spinner=False, max_entries=128)
def explain_cached(bundle_key: str, row_json: str, class_index):
    """SHAP attribution of one student, cached so sliders stay snappy."""
    from .tabular import shap_attributions
    b = E.load_bundle(bundle_key)
    row = pd.DataFrame([json.loads(row_json)])
    vals, base, cols = shap_attributions(b, row, class_index=class_index, n_background=25)
    fr = b.frame(row).iloc[0]
    names = [f"{G.label(c)} = {G.format_value(c, fr[c])}" for c in cols]
    return names, [float(v) for v in vals[0]], float(base)


def explain(bundle_key: str, profile: dict, class_index=None):
    b = E.load_bundle(bundle_key)
    row = {c: profile.get(c, b.defaults[c]) for c in b.feature_cols}
    return explain_cached(bundle_key, json.dumps(row, sort_keys=True, default=float), class_index)


def reasons(names, values, scale=100.0, unit="points", top=3, subject="the risk"):
    """Plain-English top reasons from attribution values."""
    out = []
    for i in np.argsort(-np.abs(values))[:top]:
        v = values[i] * scale
        if abs(v) < 0.5:
            continue
        verb = "raises" if v > 0 else "lowers"
        out.append(f"**{names[i]}** {verb} {subject} by about {abs(v):.0f} {unit}")
    return out


# ===================================================================== student form
COUNT_CAPS = {"credited": 12, "enrolled": 12, "evaluations": 16, "approved": 12, "without evaluations": 8}


def profile() -> dict:
    if "profile" not in st.session_state:
        st.session_state["profile"] = E.default_profile()
    return st.session_state["profile"]


def current_stage(default="sem1") -> str:
    return st.session_state.get("stage", default)


def stage_selector(page: str, allowed=("start", "sem1", "sem2"), help_=None) -> str:
    cur = current_stage()
    if cur not in allowed:
        cur = allowed[0]
    labels = {k: C.STAGES[k]["label"] for k in allowed}
    pick = st.segmented_control("What do we know about this student?", list(allowed), default=cur,
                                format_func=lambda k: labels[k], key=f"{page}:stage",
                                help=help_ or "Forecasts get better as more results come in - but later. "
                                              "Choose how far into the course this student is.")
    stage = pick or cur
    st.session_state["stage"] = stage
    return stage


def _sync(feature: str, key: str):
    st.session_state["profile"][feature] = st.session_state[key]


def _apply_preset(page: str):
    name = st.session_state[f"{page}:preset"]
    base = E.default_profile()
    st.session_state["profile"] = {**base, **E.PRESETS.get(name, {})}
    for k in [k for k in st.session_state if k.startswith(f"{page}:") and not k.endswith((":preset", ":stage"))]:
        del st.session_state[k]


def preset_selector(page: str):
    names = ["Typical student"] + list(E.PRESETS)
    st.selectbox("Start from an example student", names, key=f"{page}:preset", on_change=_apply_preset, args=(page,),
                 help="Loads a ready-made student record so you can see how the system reacts. "
                      "Then change anything below.")


def _widget(f: str, spec: dict, page: str, prof: dict):
    key, label, hlp = f"{page}:{f}", G.label(f), G.help_text(f)
    cur = prof[f]
    fmt = lambda v, f=f: G.format_value(f, v)   # noqa: E731
    if spec["type"] == "cat":
        opts = sorted(int(float(o)) for o in spec["options"])
        idx = opts.index(int(cur)) if int(cur) in opts else 0
        st.selectbox(label, opts, index=idx, key=key, help=hlp, format_func=fmt, on_change=_sync, args=(f, key))
    elif spec.get("int") and spec["min"] == 0 and spec["max"] == 1:
        names = G.CODE_MAPS.get(f, G.YES_NO)
        st.radio(label, [0, 1], index=int(cur), horizontal=True, key=key, help=hlp,
                 format_func=lambda v, n=names: n[v], on_change=_sync, args=(f, key))
    else:
        if f == "Age at enrollment":
            lo, hi, step, val = 17, 60, 1, int(np.clip(cur, 17, 60))
        elif "(grade)" in f:
            lo, hi, step, val = 0.0, 20.0, 0.1, float(np.clip(cur, 0, 20))
        elif f == "Application order":
            lo, hi, step, val = 0, 6, 1, int(np.clip(cur, 0, 6))
        else:
            cap = next((v for k, v in COUNT_CAPS.items() if k in f), 12)
            lo, hi, step, val = 0, cap, 1, int(np.clip(cur, 0, cap))
        st.slider(label, lo, hi, val, step, key=key, help=hlp, on_change=_sync, args=(f, key))


def student_form(stage: str, page: str, columns: int = 3):
    """Plain-language form for the fields relevant to ``stage``. Returns the profile dict."""
    prof = profile()
    spec = E.load_bundle("dropout_sem2").ui_spec
    shown = [c for c in C.STAGES[stage]["cols"] if c in spec]
    groups: dict[str, list] = {}
    for c in shown:
        groups.setdefault(G.group(c), []).append(c)
    icons = {"Background": "🎓", "Money matters": "💰", "Semester 1 record": "📘", "Semester 2 record": "📗"}
    tabs = st.tabs([f"{icons.get(g, '•')} {g}" for g in groups])
    for tab, (g, feats) in zip(tabs, groups.items()):
        with tab:
            if g == "Semester 2 record" and stage != "sem2":
                st.caption("Only the registration is known so far: subjects the student has signed up for in semester 2.")
            cols = st.columns(columns)
            for i, f in enumerate(feats):
                with cols[i % columns]:
                    _widget(f, spec[f], page, prof)
    return prof


# ============================================================ scoreboard helpers
def best_metric_cards():
    """Honest, plain-English accuracy cards read from the saved metrics (None-safe)."""
    perf = load_json(C.REPORTS_DIR / "metrics_performance.json")
    drop = load_json(C.REPORTS_DIR / "metrics_dropout.json")
    att = load_json(C.REPORTS_DIR / "metrics_attendance.json")
    sen = load_json(C.REPORTS_DIR / "metrics_sentiment.json")
    bert = load_json(C.REPORTS_DIR / "metrics_sentiment_distilbert.json")
    cards = []
    if perf:
        d = perf["settings"]["sem1"]["test"]["MLP (deep learning)"]
        naive = perf["settings"]["sem1"]["test"]["Typical pass rate (no model)"]
        cards.append(("Subject forecast (after sem 1)", f"±{d['rmse']:.1f} subjects",
                      f"typical error on unseen students (guessing the average pass rate: ±{naive['rmse']:.1f}); "
                      f"{d['within_1_subject']:.0%} are within 1 subject", "good"))
    if drop:
        d = drop["settings"]["sem1"]["test"]["MLP (deep learning)"]
        cards.append(("Dropout warning (after sem 1)", f"{d['dropout_recall']:.0%} caught",
                      f"of students who really dropped out were flagged; ROC-AUC {d['roc_auc']:.2f}", "good"))
    if att:
        d = att["test"]["BiLSTM + static (deep learning)"]
        cards.append(("Attendance risk (simulated data)", f"{d['recall']:.0%} caught",
                      "of at-risk students flagged - on SIMULATED data, so it proves the method, not real accuracy", "warn"))
    if sen:
        m = sen["test"]["Embedding + BiLSTM (deep learning)"]
        txt = f"macro-F1 {m['macro_f1']:.2f} with the BiLSTM"
        if bert:
            txt += f"; DistilBERT {bert['test']['DistilBERT (fine-tuned)']['macro_f1']:.2f}"
        cards.append(("Feedback mood (3 classes)", f"{m['accuracy']:.0%} right", txt, "info"))
    return cards


# ================================================================== what-if + cohort
LEVERS = ["Tuition fees up to date", "Debtor", "Scholarship holder",
          C.SEM1_COLS[3], C.SEM1_COLS[4], C.SEM1_COLS[5], C.SEM1_COLS[2],
          C.SEM2_REGISTRATION_COLS[1], C.SEM2_RESULT_COLS[1], C.SEM2_RESULT_COLS[2], C.SEM2_RESULT_COLS[3]]


def _range_for(f: str):
    if f == "Age at enrollment":
        return 17, 60, 1
    if "(grade)" in f:
        return 0.0, 20.0, 0.5
    cap = next((v for k, v in COUNT_CAPS.items() if k in f), 12)
    return 0, cap, 1


def what_if_panel(prof: dict, stage: str, page: str):
    """'Change one thing' sandbox: before/after forecasts plus a curve as the value moves."""
    import plotly.express as px
    options = [f for f in LEVERS if f in C.STAGES[stage]["cols"]]
    c1, c2 = st.columns([1.2, 1])
    f = c1.selectbox("Change one thing about this student ...", options, format_func=G.label, key=f"{page}:wi_f",
                     help="Pick a factor and move it to see how the forecasts react. This is how an adviser can "
                          "test which intervention might help.")
    cur = prof[f]
    binary = f in G.CODE_MAPS and set(G.CODE_MAPS[f]) == {0, 1}
    if binary:
        new = c2.radio(f"... to", [0, 1], index=1 - int(cur), horizontal=True, key=f"{page}:wi_v:{f}",
                       format_func=lambda v, n=G.CODE_MAPS[f]: n[v])
    else:
        lo, hi, step = _range_for(f)
        new = c2.slider("... to", lo, hi, type(step)(np.clip(cur, lo, hi)), step, key=f"{page}:wi_v:{f}")
    r = E.what_if(prof, stage, {f: new})
    b, a = r["before"], r["after"]
    cols = st.columns(2)
    if b["dropout"]:
        d = a["dropout"]["risk"] - b["dropout"]["risk"]
        cols[0].markdown(kpi("Dropout risk", f"{100 * a['dropout']['risk']:.0f}%",
                             f"was {100 * b['dropout']['risk']:.0f}% &rarr; <b>{100 * d:+.0f} points</b>",
                             "good" if d < -0.005 else ("bad" if d > 0.005 else "info")), unsafe_allow_html=True)
    if b["subjects"] and a["subjects"]:
        d = a["subjects"]["backlog"] - b["subjects"]["backlog"]
        cols[1].markdown(kpi("Expected backlog", f"{a['subjects']['backlog']:.1f} subjects",
                             f"was {b['subjects']['backlog']:.1f} &rarr; <b>{d:+.1f}</b>",
                             "good" if d < -0.05 else ("bad" if d > 0.05 else "info")), unsafe_allow_html=True)
    if not binary:
        vals = np.round(np.arange(lo, hi + 1e-9, step), 2)
        df = E.sweep(prof, stage, f, vals)
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                            subplot_titles=("Chance of dropping out (%)", "Expected backlog (subjects not cleared)"))
        if df["dropout_risk"].notna().any():
            fig.add_trace(go.Scatter(x=df["value"], y=100 * df["dropout_risk"], mode="lines+markers", showlegend=False,
                                     line=dict(color=PAL["bad"], width=2), marker=dict(size=8),
                                     hovertemplate="%{x}: %{y:.0f}%<extra></extra>"), row=1, col=1)
            fig.update_yaxes(range=[0, 100], row=1, col=1)
        if df["backlog"].notna().any():
            fig.add_trace(go.Scatter(x=df["value"], y=df["backlog"], mode="lines+markers", showlegend=False,
                                     line=dict(color=PAL["primary"], width=2), marker=dict(size=8),
                                     hovertemplate="%{x}: %{y:.1f} subjects<extra></extra>"), row=1, col=2)
        for c in (1, 2):
            fig.add_vline(x=float(cur), line_dash="dot", line_color=PAL["muted"], row=1, col=c)
            fig.update_xaxes(title_text=G.label(f), row=1, col=c)
        st.plotly_chart(style(fig, 340), key=f"{page}_wi_curve")
        st.caption("Other fields are held at this student's values. The curve shows what the model has learned from "
                   "past students - it is a statistical association, not a guarantee that changing the factor "
                   "will cause the change.")


def cohort_position(risk: float, stage: str, page: str):
    """Where does this student sit among the 4,424 real students? (histogram + marker)"""
    r = 100 * E.cohort_dropout_risk(stage)
    pct = float((r < 100 * risk).mean() * 100)
    fig = go.Figure(go.Histogram(x=r, nbinsx=30, marker_color=PAL["tint"]))
    fig.add_vline(x=100 * risk, line_color=PAL["bad"], line_width=3, annotation_text="this student")
    fig.update_layout(title=f"Dropout risk of all {len(r):,} real students at this stage",
                      xaxis_title="Dropout risk (%)", yaxis_title="students")
    st.plotly_chart(style(fig, 300), key=f"{page}_cohort")
    st.caption(f"This student's dropout risk is higher than that of **{pct:.0f}%** of the real students.")


# ======================================================================== attendance
def attendance_static_form(bundle, page: str, columns: int = 4) -> dict:
    """Plain-language inputs for the static attendance features. Returns {column: value}."""
    cols = st.columns(columns)
    vals = {}
    for i, f in enumerate(bundle.static_cols):
        s = bundle.ui_spec[f]
        with cols[i % columns]:
            lab, hlp, key = G.label(f), G.help_text(f), f"{page}:att:{f}"
            d = bundle.defaults[f]
            if s["type"] == "num" and s.get("int") and s["min"] == 0 and s["max"] == 1:
                vals[f] = st.radio(lab, [0, 1], index=int(round(d)), horizontal=True, key=key, help=hlp,
                                   format_func=lambda v: "Yes" if v else "No")
            elif f == "semester":
                vals[f] = st.slider(lab, 1, 8, int(np.clip(round(d), 1, 8)), key=key, help=hlp)
            elif f == "age":
                vals[f] = st.slider(lab, 17, 40, int(np.clip(round(d), 17, 40)), key=key, help=hlp)
            elif f == "commute_km":
                vals[f] = st.slider(lab + " (km)", 0.0, 60.0, float(np.clip(d, 0, 60)), 0.5, key=key, help=hlp)
            elif f == "prev_sem_attendance":
                vals[f] = st.slider(lab + " (%)", 30.0, 100.0, float(np.clip(d, 30, 100)), 1.0, key=key, help=hlp)
            elif s["type"] == "cat":
                vals[f] = st.selectbox(lab, s["options"], key=key, help=hlp)
            else:
                vals[f] = st.number_input(lab, float(s["min"]), float(s["max"]), float(d), key=key, help=hlp)
    return vals


def _scenario(name: str, bundle) -> pd.DataFrame:
    T = bundle.seq_len
    idx = [f"Week {w + 1}" for w in range(T)]
    ch = bundle.channels
    t = np.arange(T)
    if name == "Steady attender":
        a, asg, lg = 92 - 0.3 * t, 0.9 - 0.0 * t, 8 + 0 * t
    elif name == "Fading attender":
        a, asg, lg = 90 - 5 * t, np.clip(0.85 - 0.08 * t, 0.1, 1), np.clip(9 - t, 1, None)
    elif name == "Sudden dip":
        a = np.where(t < 4, 91 - 0.2 * t, 58 - 1.5 * (t - 4))
        asg, lg = np.where(t < 4, 0.85, 0.35), np.where(t < 4, 8, 2)
    else:
        return pd.DataFrame(np.round(bundle.seq_means, 2), columns=ch, index=idx)
    data = {ch[0]: a, ch[1]: asg, ch[2]: lg}
    return pd.DataFrame({c: np.round(np.asarray(data[c], float), 2) for c in ch}, index=idx)


def weekly_editor(bundle, page: str) -> pd.DataFrame:
    """Editable week-by-week table + one-click example patterns."""
    ss, ver = f"{page}:weekly_df", f"{page}:weekly_ver"
    if ss not in st.session_state:
        st.session_state[ss], st.session_state[ver] = _scenario("Typical", bundle), 0
    st.caption("Start from an example pattern, then edit any cell:")
    cols = st.columns(4)
    for c, name in zip(cols, ["Typical", "Steady attender", "Fading attender", "Sudden dip"]):
        if c.button(name, key=f"{page}:sc:{name}", width="stretch"):
            st.session_state[ss] = _scenario(name, bundle)
            st.session_state[ver] += 1
    cfg = {bundle.channels[0]: st.column_config.NumberColumn(G.label(bundle.channels[0]), min_value=0, max_value=100, format="%.0f"),
           bundle.channels[1]: st.column_config.NumberColumn(G.label(bundle.channels[1]) + " (0-1)", min_value=0.0, max_value=1.0, format="%.2f"),
           bundle.channels[2]: st.column_config.NumberColumn(G.label(bundle.channels[2]), min_value=0, max_value=60, format="%.0f")}
    return st.data_editor(st.session_state[ss], key=f"{page}:weekly_{st.session_state[ver]}", column_config=cfg,
                          num_rows="fixed", width="stretch")


@st.cache_resource(show_spinner=False)
def get_sentiment():
    from .sentiment import load_sentiment_predictor
    return load_sentiment_predictor()


# ============================================================ explanation widgets
def flow(items) -> str:
    """Boxes joined by arrows. items = [(kind, title, text)] with kind in data | model | out."""
    parts = []
    for i, (kind, title, text) in enumerate(items):
        if i:
            parts.append("<div class='arrow'>&#10132;</div>")
        parts.append(f"<div class='box {kind}'><b>{html.escape(title)}</b>{html.escape(text)}</div>")
    return "<div class='flow'>" + "".join(parts) + "</div>"


def module_strip(m: dict):
    """Question / data / inputs / model / output of one module as a labelled flow."""
    real = "real data" if m["real"] else "SIMULATED data"
    st.markdown(flow([("data", f"Trained on ({real})", m["data"]), ("data", "What goes in", m["inputs"]),
                      ("model", "The model", m["model"]), ("out", "What comes out", m["output"])]),
                unsafe_allow_html=True)


def training_curve(hist: dict, title: str = "How the network learned") -> go.Figure:
    ep = list(range(1, len(hist["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=hist["train_loss"], name="Error on students it studies (training)",
                             line=dict(color=PAL["primary"], width=2)))
    fig.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name="Error on students it has not seen (checking)",
                             line=dict(color=PAL["warn"], width=2)))
    b = int(hist.get("best_epoch", 0)) + 1
    fig.add_vline(x=b, line_dash="dot", line_color=PAL["muted"], annotation_text="best point - training stopped here",
                  annotation_position="top")
    fig.update_layout(title=title, xaxis_title="round of training (epoch)", yaxis_title="error (lower is better)",
                      legend=dict(orientation="h", y=-0.28))
    return style(fig, 340)


def highlight_words(words, effects) -> str:
    """Words coloured by how much they supported (green) or opposed (red) the prediction."""
    mx = max([abs(float(e)) for e in effects] + [1e-9])
    out = []
    for w, e in zip(words, effects):
        a = min(1.0, abs(float(e)) / mx) * 0.75
        rgb = "22,163,74" if e > 0 else "220,38,38"
        style_ = f"background:rgba({rgb},{a:.2f});" if abs(e) > 0.01 * mx else ""
        out.append(f"<span style='{style_}padding:1px 3px;border-radius:4px'>{html.escape(w)}</span>")
    return "<div style='line-height:2;font-size:1.05rem'>" + " ".join(out) + "</div>"
