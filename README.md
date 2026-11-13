# Smart Education & Student Intelligence System

An early-warning assistant for **college and university advisers**. It uses four neural-network models and one dashboard to answer: *how many subjects will this student clear, will they drop out, will their attendance fall below 75%, and how do students feel about their courses?* Every number comes with plain-language reasons.

**New here? Read `docs/PROJECT_GUIDE.md` first** (20 minutes: which data trains which model, how the deep-learning code works, every short form explained).

| Model | Question | Trained on | Deep model | Compared with |
|---|---|---|---|---|
| A. Subject forecast | How many of semester-2's registered subjects will the student clear? | Real university records (4,424 students) | MLP | no-model guess, Ridge, Random Forest, XGBoost |
| B. Dropout warning | Drop out / still studying / graduate? | The same real records (4,424 students) | MLP + class weights | Logistic Regression, Random Forest, XGBoost |
| C. Attendance watch | Will final attendance be < 75%, judging by the first 8 weeks? | **Simulated** students (rules in code) | BiLSTM + static features | Logistic Regression, RF, XGBoost, plain MLP |
| D. Feedback mood | Negative / neutral / positive comment, and which words decided it | Real Coursera reviews (12,000 balanced sample from 107,018 raw reviews) | Embedding + BiLSTM (DistilBERT optional on Colab) | TF-IDF + Logistic Regression |

Models A and B are trained at different moments (*before any exam results*, *after semester 1*, and for B *after semester 2*) because a warning only helps if it comes early.

## 1. Setup (once)

Python 3.11 or newer (built and tested on 3.13). Runs on a normal laptop CPU.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python download_data.py          # downloads the 2 real datasets into data/raw/ (about 20 MB)
```

## 2. Train (about 5 minutes on a laptop CPU)

```bash
python train_all.py --source real
```

Trains all four models, saves them in `models/`, and writes `reports/results_summary.md` (the tables for your report) and figures. Options:

```bash
python train_all.py --only dropout sentiment     # a subset
python train_all.py --tune                       # Optuna search for the MLPs (slower)
python train_all.py --imbalance smote            # SMOTE instead of class weights (dropout)
python train_all.py --summary-only               # just rebuild results_summary.md
```

Always re-run this once on your own machine: saved models can depend on library versions.

## 3. Run the dashboard

```bash
streamlit run app/Home.py
```

| Page | What it is for |
|---|---|
| **Home** | Example students, what the system does, who it is for |
| **How It Works** | The big picture, which data trains which model, what every field means, what we left out, honest limits |
| **Student 360** | One student, all four models, one risk card with reasons, what-ifs and next steps |
| **Subject Forecast / Dropout Warning / Attendance Watch / Feedback Mood** | One model each: edit the student's record, see the answer and why |
| **Class Overview** | Score a whole class (the held-out real students, or upload your own CSV), compare groups, reality check |
| **Inside the Network** | The real network drawn with its real size, follow one student through it, watch a tiny network learn, the code in plain words |
| **Accuracy** | Every deep model next to the simpler models, measured on students it never saw |

## 4. Optional: DistilBERT for the comments model

DistilBERT needs internet and a GPU, so run `colab_distilbert.ipynb` in Google Colab (free T4, about 10 minutes), unzip the result into this folder, then `python train_all.py --summary-only`. The dashboard switches to DistilBERT automatically when `models/sentiment_distilbert/` exists. Offline check of the code: `python -m smartedu.sentiment_distilbert --smoke`.

## 5. Deploy on Streamlit Community Cloud

1. Push this repository to GitHub with the included `requirements.txt` and model files.
2. In Streamlit Community Cloud, select the repository and `main` branch.
3. Set the app entrypoint to `app/Home.py`.
4. In Advanced settings, select Python 3.13 to match the environment used to create the included model bundles.
5. Deploy and inspect the build logs before sharing the app.

The deployment requirements use CPU-only PyTorch. The cloud environment is Linux, so it is not affected by Windows Smart App Control. This is a separate deployment route; it does not repair the local Windows PyTorch installation.

## 6. Test

```bash
python tests/test_app_smoke.py        # opens every dashboard page headlessly and fails on any error
```

## Project layout

```
app/                Home.py + pages/       the dashboard
smartedu/           config, data loaders, clean-up pipeline, networks (nn.py), the four modules,
                    glossary.py (plain-English field names), about.py (plain description of each model)
train_all.py        trains everything       download_data.py   fetches the real datasets
data/raw/           datasets                models/            trained models
reports/            metrics + figures       docs/              PROJECT_GUIDE, PROJECT_REPORT, VIVA_GUIDE
colab_distilbert.ipynb
```

## Be honest about the data (important for the viva)

* University records and Coursera reviews are **real public data**. The included `feedback.csv` is a balanced 12,000-review sample derived from `coursera_reviews_full.csv` (107,018 raw rows). The university records come from one Portuguese polytechnic; to use the system at another college, retrain it on that college's records (same code, different CSV).
* Attendance is **simulated** (rules written in `smartedu/synthetic.py`): its high scores show the method works, not how accurate it would be in a real college.
* The overall risk score is an adjustable weighted average, not a separately trained model.
* On this kind of table data, simpler models score about the same as the neural networks. The Accuracy page shows it, and the report discusses it.
