import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from smartedu import config as C  # noqa: E402
from smartedu import engine as E  # noqa: E402
from smartedu import glossary as G  # noqa: E402
from smartedu import nn as NN  # noqa: E402
from smartedu import ui  # noqa: E402

PAGE = "inside"
ui.page_setup("Inside the network", "🧠",
              "Open the black box: see the real network that makes the predictions, follow one student through it, "
              "and watch a tiny network learn.",
              chips=["Real model, real weights", "Interactive", "The code explained"])
ui.require("dropout_sem1", "performance_sem1")

MODELS = {"Dropout warning (Model B)": ("dropout", E.DROP_KEY, C.DROPOUT_STAGES),
          "Subject forecast (Model A)": ("performance", E.PERF_KEY, C.PERFORMANCE_STAGES)}


def friendly_input(name: str) -> str:
    """'num__Age at enrollment' -> 'Age when joining'; 'cat__Course_9500' -> 'Course: Nursing'."""
    n = name.split("__", 1)[-1]
    if name.startswith("cat__") and "_" in n:
        feat, code = n.rsplit("_", 1)
        return f"{G.label(feat)}: {G.format_value(feat, code)}"
    return G.label(n)


def pick_model(where: str):
    c = st.columns([1.3, 1.3, 2])
    which = c[0].radio("Which model?", list(MODELS), key=f"{PAGE}_{where}_m")
    _, keys, stages = MODELS[which]
    stage = c[1].radio("How much do we know?", stages, format_func=lambda s: C.STAGES[s]["label"], key=f"{PAGE}_{where}_s")
    return which, E.load_bundle(keys[stage]), stage


tabs = st.tabs(["🏗️ The real network", "🚶 Follow one student", "🎮 Playground: watch a network learn",
                "💻 The code in plain words"])

# ======================================================================== 1. the real network
with tabs[0]:
    which, bundle, stage = pick_model("arch")
    a = E.architecture(bundle)
    sizes = a["sizes"]
    names = ["Input"] + [f"Hidden layer {i}" for i in range(1, len(sizes) - 1)] + ["Output"]
    sub = [f"{sizes[0]} numbers"] + [f"{s} calculators" for s in sizes[1:-1]] + \
          [("3 chances" if a["task"] == "classification" else "1 number")]

    SHOW = 10
    xs, ys, lines_x, lines_y, labels = [], [], [], [], []
    pos = []
    for li, s in enumerate(sizes):
        k = min(s, SHOW)
        yy = np.linspace(-1, 1, k) if k > 1 else np.array([0.0])
        pos.append([(li, float(y)) for y in yy])
        for y in yy:
            xs.append(li)
            ys.append(float(y))
    for li in range(len(sizes) - 1):
        for (x0, y0) in pos[li]:
            for (x1, y1) in pos[li + 1]:
                lines_x += [x0, x1, None]
                lines_y += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lines_x, y=lines_y, mode="lines", line=dict(color="rgba(20,33,61,.10)", width=1),
                             hoverinfo="skip", showlegend=False))
    colors = [ui.PAL["accent"]] + [ui.PAL["primary"]] * (len(sizes) - 2) + [ui.PAL["dark"]]
    for li in range(len(sizes)):
        pts = pos[li]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
                                 marker=dict(size=20, color=colors[li], line=dict(color="#fff", width=2)),
                                 hoverinfo="skip", showlegend=False))
        more = sizes[li] - len(pts)
        txt = f"<b>{names[li]}</b><br>{sub[li]}" + (f"<br><i>(only {len(pts)} drawn)</i>" if more > 0 else "")
        fig.add_annotation(x=li, y=-1.55, text=txt, showarrow=False, font=dict(size=13))
    fig.update_layout(xaxis=dict(visible=False, range=[-.5, len(sizes) - .5]),
                      yaxis=dict(visible=False, range=[-2.1, 1.3]), title=f"{which.split(' (')[0]}: the real layers of the network (big layers are drawn with 10 dots)")
    st.plotly_chart(ui.style(fig, 430), key=f"{PAGE}_arch")
    k = st.columns(4)
    k[0].markdown(ui.kpi("Layers of calculators", f"{len(sizes) - 1}", "hidden layers + the output layer"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Numbers it learned", f"{a['params']:,}", "weights and biases - the 'memory' of the network"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Inputs per student", f"{sizes[0]}", "after turning the record into numbers"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Switched off while learning", f"{a['dropout']:.0%}", "random 'Dropout' stops it memorising"), unsafe_allow_html=True)

    st.markdown("#### What happens inside every hidden layer")
    c = st.columns(4)
    c[0].markdown(ui.card("1. Weighted sum", "Each calculator multiplies every incoming number by a weight (how much it cares) and adds them up.", "Linear"), unsafe_allow_html=True)
    c[1].markdown(ui.card("2. Re-centre", "The sums are shifted and rescaled so no calculator shouts louder than the rest. Makes learning stable.", "BatchNorm"), unsafe_allow_html=True)
    c[2].markdown(ui.card("3. On / off switch", "Negative results become zero, positive ones pass through. This simple bend is what lets the network learn curved patterns.", "ReLU"), unsafe_allow_html=True)
    c[3].markdown(ui.card("4. Random blackout", "While learning, 30% of calculators are muted at random so the network cannot just memorise students.", "Dropout"), unsafe_allow_html=True)

    hist = getattr(bundle.model, "history_", None)
    if hist and hist.get("train_loss"):
        st.plotly_chart(ui.training_curve(hist), key=f"{PAGE}_curve")
        ui.callout("<b>Reading the curve:</b> the teal line is the error on students the network studies; the orange line is the error on "
                   "students it has not seen. Training stops when the orange line stops improving - going further would only memorise.")

# ======================================================================== 2. follow one student
with tabs[1]:
    which, bundle, stage = pick_model("trace")
    who = st.radio("Pick a student", list(E.PRESETS), horizontal=True, key=f"{PAGE}_who")
    prof = {**E.default_profile(), **E.PRESETS[who]}
    t = E.trace_mlp(bundle, prof)

    st.markdown("#### Step 1 - The record becomes numbers")
    names = [friendly_input(n) for n in t["names"]]
    x = np.asarray(t["inputs"], float)
    order = np.argsort(-np.abs(x))[:12][::-1]
    fig = go.Figure(go.Bar(x=[x[i] for i in order], y=[names[i] for i in order], orientation="h",
                           marker_color=[ui.PAL["primary"] if x[i] > 0 else ui.PAL["warn"] for i in order]))
    fig.update_layout(title="The 12 most unusual facts about this student", xaxis_title="above (+) or below (-) the typical student")
    st.plotly_chart(ui.style(fig, 400), key=f"{PAGE}_inputs")
    st.caption("Numbers are put on a common scale (0 = typical student). Yes/No facts and course names become 0 or 1.")

    st.markdown("#### Step 2 - The hidden layers react")
    cols = st.columns(len(t["blocks"]))
    for i, (col, h) in enumerate(zip(cols, t["blocks"]), start=1):
        n = len(h)
        w = 16 if n >= 64 else 8
        pad = (-n) % w
        z = np.pad(h, (0, pad), constant_values=np.nan).reshape(-1, w)
        hm = go.Figure(go.Heatmap(z=z, colorscale=[[0, "#F4F6FA"], [1, ui.PAL["primary"]]], showscale=False, xgap=2, ygap=2))
        hm.update_layout(title=f"Layer {i}: {int((h > 0).sum())} of {n} 'on'", yaxis=dict(visible=False, autorange="reversed"),
                         xaxis=dict(visible=False))
        col.plotly_chart(ui.style(hm, 230), key=f"{PAGE}_h{i}")
    st.caption("Each square is one calculator; darker = reacting more strongly to this particular student. "
               "A different student lights up a different pattern.")

    st.markdown("#### Step 3 - The answer")
    if bundle.task == "classification":
        p = np.asarray(t["result"], float)
        st.plotly_chart(ui.prob_bar(["Drops out", "Still studying", "Graduates"], p, "Chances that add up to 100%"), key=f"{PAGE}_out")
        st.markdown(f"The output layer turns its three raw scores into chances with a *softmax*: **{p[0]:.0%} drop out**, "
                    f"{p[1]:.0%} still studying, {p[2]:.0%} graduate.")
    else:
        reg = float(prof.get(C.PERFORMANCE_REGISTERED, 6))
        v = float(np.clip(np.asarray(t["result"]).ravel()[0], 0, reg))
        st.markdown(ui.kpi("Subjects expected to clear", f"{v:.1f} of {reg:.0f}", "the single output number", "good"),
                    unsafe_allow_html=True)
    ui.callout("Nothing here is learned for this student - the numbers only <b>flow forward</b>. All the learning happened earlier, on "
               "thousands of past students, and is stored in the weights.")

# ======================================================================== 3. playground
with tabs[2]:
    st.markdown("### Watch a tiny network learn")
    st.markdown("Blue and orange dots are two groups of points. The network must learn to tell them apart by drawing a boundary. "
                "Change the settings - the network re-trains instantly.")
    c = st.columns([1, 1, 1, 1])
    data_kind = c[0].selectbox("Pattern", ["Two moons", "Circles", "Two blobs (easy)"], key=f"{PAGE}_d")
    hidden = c[1].slider("Calculators in the hidden layer", 1, 32, 8, key=f"{PAGE}_hid",
                         help="More calculators = more flexible boundary.")
    lr = c[2].select_slider("Learning speed", [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], value=1.0, key=f"{PAGE}_lr",
                            help="How big a step to take after each round. Too small: slow. Too big: unstable.")
    epochs = c[3].slider("Rounds of training", 10, 1500, 800, 10, key=f"{PAGE}_ep")
    act_name = st.radio("On/off switch (activation)", ["tanh", "ReLU"], horizontal=True, key=f"{PAGE}_act")

    from sklearn.datasets import make_blobs, make_circles, make_moons

    @st.cache_data(show_spinner=False)
    def toy(kind: str):
        if kind == "Two moons":
            return make_moons(300, noise=0.2, random_state=1)
        if kind == "Circles":
            return make_circles(300, noise=0.1, factor=0.45, random_state=1)
        return make_blobs(300, centers=[(-1.3, -1), (1.3, 1)], cluster_std=0.8, random_state=1)

    def train_tiny(X, y, h, lr, epochs, act):
        rng = np.random.default_rng(0)
        W1, b1 = rng.normal(0, 1, (2, h)) * 0.8, np.zeros(h)
        W2, b2 = rng.normal(0, 1, (h, 1)) * 0.8, np.zeros(1)
        f = (np.tanh, lambda a: 1 - a ** 2) if act == "tanh" else (lambda z: np.maximum(z, 0), lambda a: (a > 0).astype(float))
        Y = y[:, None].astype(float)
        losses = []
        for _ in range(epochs):
            a1 = f[0](X @ W1 + b1)
            p = 1 / (1 + np.exp(-(a1 @ W2 + b2)))
            losses.append(float(-np.mean(Y * np.log(p + 1e-9) + (1 - Y) * np.log(1 - p + 1e-9))))
            d2 = (p - Y) / len(X)
            gW2, gb2 = a1.T @ d2, d2.sum(0)
            d1 = (d2 @ W2.T) * f[1](a1)
            gW1, gb1 = X.T @ d1, d1.sum(0)
            W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2   # noqa: E702

        def predict(Z):
            return (1 / (1 + np.exp(-(f[0](Z @ W1 + b1) @ W2 + b2)))).ravel()
        return predict, losses

    X, y = toy(data_kind)
    predict, losses = train_tiny(X, y, hidden, float(lr), int(epochs), act_name)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 70), np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 70))
    zz = predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    acc = float(((predict(X) > .5) == y).mean())

    g1, g2 = st.columns([1.3, 1])
    fig = go.Figure(go.Contour(x=gx[0], y=gy[:, 0], z=zz, showscale=False, contours=dict(showlines=False), opacity=.55,
                               colorscale=[[0, "#F4B183"], [.5, "#FFFFFF"], [1, "#7FB3D5"]]))
    for cls, col in ((0, "#D97706"), (1, "#0E7C86")):
        fig.add_trace(go.Scatter(x=X[y == cls, 0], y=X[y == cls, 1], mode="markers", showlegend=False,
                                 marker=dict(color=col, size=7, line=dict(color="#fff", width=1))))
    fig.update_layout(title=f"What the network believes after {epochs} rounds - {acc:.0%} of dots on the right side",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    g1.plotly_chart(ui.style(fig, 420), key=f"{PAGE}_toy")
    lf = go.Figure(go.Scatter(y=losses, mode="lines", line=dict(color=ui.PAL["primary"], width=2)))
    lf.update_layout(title="Error while learning (lower is better)", xaxis_title="round of training", yaxis_title="error")
    g2.plotly_chart(ui.style(lf, 420), key=f"{PAGE}_toyloss")
    ui.callout("<b>Experiments to try on <i>Two moons</i>:</b> (1) Set the hidden layer to <b>1</b> calculator: the boundary stays almost a straight "
               "line and only about 85% of dots are right. (2) Set it to <b>8</b> (speed 1.0, 800 rounds): the boundary bends around the moons "
               "and about 97% are right. (3) Set speed to <b>0.003</b> with <b>100</b> rounds: the steps are so small it has barely learned "
               "(about 68%). The real models work the same way - just with 48 inputs instead of 2.")

# ======================================================================== 4. the code
with tabs[3]:
    st.markdown("### The code, in plain words")
    st.markdown("Everything is **PyTorch**. A network is a stack of layers; *training* repeats five steps thousands of times.")
    st.code('''for each round (epoch):
    for each small batch of students:
        prediction = network(batch)                 # 1. forward pass: numbers flow through the layers
        error      = loss_function(prediction, truth)   # 2. how wrong was it?
        error.backward()                            # 3. work out which weights caused the error
        optimizer.step()                            # 4. nudge every weight a little to reduce it
    check the error on students it has NOT seen     # 5. stop when this stops improving (early stopping)''', language="python")
    st.markdown("**The real network class** (file `smartedu/nn.py`) - it is only this:")
    st.code(inspect.getsource(NN.MLP), language="python")
    st.markdown(
        "- `nn.Linear(d, h)` = the *weighted sum* of every input for `h` calculators.\n"
        "- `nn.BatchNorm1d(h)` = re-centre; `nn.ReLU()` = on/off switch; `nn.Dropout(p)` = random blackout.\n"
        "- The loop `for h in hidden` stacks these blocks - e.g. `hidden=(128, 64, 32)` gives three hidden layers.\n"
        "- The last `nn.Linear(d, out_dim)` is the output layer: 3 numbers for dropout (then softmax → chances), 1 number for the subject forecast.")
    with st.expander("Where does the BiLSTM (attendance and feedback) differ?"):
        st.markdown(
            "A BiLSTM reads a **sequence** - weeks of attendance, or the words of a comment - one step at a time, keeping a running "
            "memory. *Bi* means it also reads backwards, so each step knows what came before **and** after. "
            "For text, an `nn.Embedding` first turns each word into a short list of learned numbers.")
        from smartedu.sentiment import TextNet
        st.code(inspect.getsource(TextNet), language="python")
    st.markdown("#### Which file does what")
    import pandas as pd
    st.dataframe(pd.DataFrame([
        ("smartedu/data_loaders.py", "Reads the CSV files and returns clean tables"),
        ("smartedu/glossary.py", "Plain-language names and explanations for every field"),
        ("smartedu/tabular.py", "Turns a student record into numbers (encode, scale) and bundles it with the model"),
        ("smartedu/nn.py", "The MLP and the training loop (steps 1-5 above)"),
        ("smartedu/performance.py, dropout.py", "Train models A and B and compare them with simpler models"),
        ("smartedu/attendance.py, sentiment.py", "Train the BiLSTM models C and D"),
        ("smartedu/engine.py", "Makes predictions for the dashboard (no screen code)"),
        ("smartedu/ui.py, app/", "Everything you see on screen"),
        ("train_all.py", "Runs every training in one command"),
    ], columns=["File", "What it does"]), hide_index=True, width="stretch")
