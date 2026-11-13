# Smart Education & Student Intelligence System
### Deep Learning Project Report (draft with real results)

**Student:** <your name> &nbsp;|&nbsp; **Enrolment no.:** <number> &nbsp;|&nbsp; **Programme:** MCA, Invertis University, Bareilly
**Guide:** <name> &nbsp;|&nbsp; **Session:** 2026-27

> Fill the `< >` fields and add your institute's cover, certificate and declaration pages. Numbers come from `reports/results_summary.md`, produced by `python train_all.py --source real` (re-verified: a fresh run reproduces them). For a plain-language explanation of everything below, read `docs/PROJECT_GUIDE.md`.

---

## Abstract

Colleges collect data on students but rarely turn it into early, explainable warnings. This project builds a **Smart Education & Student Intelligence System**, an early-warning assistant for college advisers, with four neural-network models: (A) a forecast of how many registered subjects a student will clear, (B) a dropout / still-studying / graduate prediction, (C) an attendance-risk prediction from the first eight weeks, and (D) sentiment analysis of student comments. Models A and B are trained at different moments (before any exam results, after semester 1, and for B after semester 2), because an early warning is only useful if it comes early. Each model is compared with classical baselines on the same split. All models are served through one Streamlit dashboard that explains every prediction (SHAP, word occlusion) and combines the four signals into a transparent risk card. On real university records, dropout prediction reaches ROC-AUC 0.87 after semester 1 and 0.88 after semester 2; the subject forecast has a typical error of about 1 subject after semester 1; three-class comment sentiment reaches macro-F1 0.70. The results also show where deep learning does *not* help: on table-like data, simpler models score about the same.

**Keywords:** educational data mining, learning analytics, early warning, deep learning, dropout prediction, sentiment analysis, explainable AI.

## 1. Introduction

### 1.1 Problem statement
Advisers usually notice struggling students late: after failed exams, long absences or a withdrawal. The goal is a **decision-support tool for colleges** that estimates, at a chosen point in time, (i) how many subjects a student will clear, (ii) whether they may drop out, (iii) whether attendance is heading below the required level, and (iv) how students feel about their courses, and explains *why*.

### 1.2 Objectives
1. Build one shared pipeline (clean-up, encoding, scaling, 70/15/15 split) and four separate models.
2. Compare each deep model with classical baselines (and a "no-model" guess) using the same split and metrics.
3. Handle class imbalance and choose metrics that match the cost of errors (recall of the Dropout / At-Risk class).
4. Make predictions explainable and use plain-language field names a non-technical adviser can read.
5. Deliver a dashboard with single-student, whole-class and combined-risk views.

### 1.3 Audience
Colleges and universities (degree programmes with semesters, subjects, fees and attendance), not schools.

## 2. Literature review (Educational Data Mining and Learning Analytics)

Educational Data Mining (EDM) applies machine learning to educational data to improve teaching and retention; Romero and Ventura (2010) survey the field. Realinho et al. (2022) built the higher-education dropout and academic-success dataset used here and showed that early academic performance in the first two semesters is the strongest signal. Kuzilek et al. (2017) released the Open University Learning Analytics Dataset (OULAD), where virtual-learning-environment activity serves as an engagement proxy; the code supports it as an optional real-data source for attendance. For text, LSTMs (Hochreiter and Schmidhuber, 1997) became the classic sentence classifier, and pretrained transformers such as DistilBERT (Sanh et al., 2019) now give the strongest accuracy at modest cost. For tabular data, Grinsztajn et al. (2022) show that tree-based models often still match or beat deep networks on medium-sized tables, which is what this project observes. Explanations use SHAP (Lundberg and Lee, 2017).

## 3. Datasets

| Model | Dataset | Size | Notes |
|---|---|---|---|
| A, B | *Predict Students' Dropout and Academic Success* (Realinho et al., 2022): real students of a Portuguese polytechnic | 4,424 students, 35 columns | Outcome: Graduate 2,209 (50%), Dropout 1,421 (32%), Enrolled 794 (18%). No missing values. Model A uses the 4,244 students who registered for at least one subject in semester 2. |
| C | **Simulated** weekly attendance (this project) | 3,000 students | See 3.1. No public attendance dataset exists. At-Risk rate 32.1%. |
| D | Coursera course reviews, 107k reviews rated 1-5 stars | 12,000 used | Class-balanced sample, 4,000 per class: 1-2 stars = Negative, 3 = Neutral, 4-5 = Positive. |

**Fields used and left out.** Inputs are course, age at joining, class timing, living away from home, admission route and preference rank, qualification before joining, fees up to date, dues, scholarship, and semester-1 / semester-2 subject records (registered, cleared, grades, credited). Deliberately **left out**: gender, marital status, nationality, international and special-needs flags (sensitive; a risk tool should not judge students by them), parents' qualification and occupation (obscure and unfair to hold against a student), and national unemployment, inflation and GDP (identical for everyone who joined in the same year). Check performed: adding the sensitive columns back changes dropout ROC-AUC by only +0.001 (after semester 1) and +0.0006 (after semester 2), so nothing useful is lost. Every field has a plain-English label and help text in the dashboard (`smartedu/glossary.py`).

**Stages.** Models A and B are trained separately for each moment: *before any exam results* (background, money matters, subjects registered), *after semester 1* (adds the semester-1 record), and for B *after semester 2*. Semester-2 results are only used by the "after semester 2" dropout model.

**Data provenance.** The files were obtained from public GitHub copies of the originals (the UCI site was not reachable from the build environment) and verified against the published shape and class counts; `download_data.py` reproduces this. For D the natural rating distribution is 91% positive, so a balanced sample is used and macro-F1 is reported.

### 3.1 Attendance simulation (Model C) - state this clearly in the viva
Rules (documented in `smartedu/synthetic.py`): each student has a latent engagement level; baseline attendance depends on engagement, distance from campus, part-time work and hostel residence; 20% of students "fade" about 1.8 points per week; 25% suffer an unpredictable late-semester shock (about -15 points) *after* the observed window; assignments and portal logins correlate with engagement. Label: At-Risk if mean attendance over all 16 weeks < 75%. The model sees only the **first 8 weeks**. Results on this data validate the *pipeline*, not real-world accuracy. The loader also supports real OULAD data.

## 4. Methodology

### 4.1 Common pipeline
Choose the columns for the stage -> stratified 70/15/15 train/validation/test split (seed 42) -> clean-up fitted on the training split only (median imputation, standardisation, one-hot encoding of categories such as *Course*) -> train -> evaluate once on the test students. Model A also reports 5-fold cross-validation.

### 4.2 Models

| Model | Deep-learning model | Compared with |
|---|---|---|
| A | MLP (64-32, BatchNorm, Dropout 0.3, AdamW, early stopping), one output (subjects cleared) | "Typical pass rate" guess, Ridge, Random Forest, XGBoost |
| B | MLP (128-64-32, BatchNorm, Dropout 0.3, class-weighted loss), 3 outputs through softmax. At the first stage: 48 inputs, 17,155 learned parameters | Logistic Regression, Random Forest, XGBoost |
| C | BiLSTM (hidden 48) over 8 weekly steps (attendance %, assignments, portal logins) fused with a small network over static facts; decision threshold tuned on validation for F2 | Logistic Regression, Random Forest, XGBoost, plain MLP |
| D | Embedding (128) + BiLSTM (64) with mean+max pooling; DistilBERT fine-tuned on Colab (optional) | TF-IDF (1-2 grams) + Logistic Regression |

Imbalance (B, C, D) uses class weights; SMOTE (Chawla et al., 2002) is available with `--imbalance smote` and Optuna (Akiba et al., 2019) tuning with `--tune`.

### 4.3 Metrics
Regression: RMSE and MAE in **subjects**, R², and "within +-1 subject". Classification: accuracy, macro-F1, ROC-AUC and, most importantly, **recall of the Dropout / At-Risk class**, because missing a student who needs help costs more than a false alarm.

### 4.4 Explainability
SHAP (global and per student) and permutation importance for tabular models; word occlusion for text; occlusion of weekly signals for attendance.

### 4.5 Combined risk card
Each model yields a risk in [0,1]: subject forecast (expected backlog share = 1 - predicted cleared / registered), dropout (P(Dropout)), attendance (P(At-Risk)), comments (P(Negative)). The overall score is a **weighted average** (default 0.35 / 0.30 / 0.25 / 0.10, editable in the dashboard), renormalised over active models, with bands Low (<0.33), Medium, High (>=0.66). It is a transparent heuristic, not a trained model, because the real data and the simulated and text data describe different students, so no student has outcomes in all of them.

### 4.6 Architecture
```
Data (CSV / simulated) -> pick columns for the stage -> clean-up -> 4 neural networks -> saved bundles
                                                     -> SHAP / occlusion -> Streamlit dashboard -> risk card
```

## 5. Results (held-out test students)

### 5.1 Model A - subjects cleared in semester 2 (test n = 637; spread of the target SD = 2.95 subjects)

**Before any exam results**

| Model | RMSE (subjects) | MAE | R2 | Within +-1 subject | 5-fold CV RMSE |
|---|---|---|---|---|---|
| Typical pass rate (no model) | 2.285 | 1.896 | 0.401 | 45% | 2.26 +- 0.04 |
| Ridge | 1.902 | 1.456 | 0.585 | 60% | 1.89 +- 0.02 |
| Random Forest | 1.958 | 1.472 | 0.560 | 64% | 1.93 +- 0.04 |
| XGBoost | 1.878 | 1.426 | 0.595 | 63% | 1.85 +- 0.04 |
| MLP (deep learning) | 1.885 | 1.432 | 0.592 | 64% | 1.87 +- 0.04 |

**After semester 1**

| Model | RMSE (subjects) | MAE | R2 | Within +-1 subject | 5-fold CV RMSE |
|---|---|---|---|---|---|
| Typical pass rate (no model) | 2.285 | 1.896 | 0.401 | 45% | 2.26 +- 0.04 |
| Ridge | 1.013 | 0.690 | 0.882 | 88% | 1.11 +- 0.06 |
| Random Forest | 1.023 | 0.645 | 0.880 | 88% | 1.13 +- 0.06 |
| XGBoost | 0.964 | 0.638 | 0.893 | 91% | 1.10 +- 0.06 |
| MLP (deep learning) | 0.964 | 0.652 | 0.893 | 90% | 1.11 +- 0.04 |

*Reading:* Before exams, background and the number of subjects registered already beat the no-model guess (RMSE 1.89 vs 2.29), but only moderately. After semester 1, the semester-1 record makes the forecast good (typical error about 1 subject, 90% of students within one subject). SHAP ranks the number of subjects registered for semester 2 first at the early stage (about 1.0 subject), then scholarship and fees up to date; after semester 1 the **number of semester-1 subjects cleared** dominates (about 1.4 subjects), followed by semester-2 subjects registered and semester-1 grade. The MLP ties the best classical model (XGBoost).

### 5.2 Model B - dropout (test n = 664, class weights)

| Stage | Model | Accuracy | Macro-F1 | Dropout recall | Dropout precision | ROC-AUC |
|---|---|---|---|---|---|---|
| Before any exam results | Logistic Regression | 0.575 | 0.540 | 0.563 | 0.619 | 0.757 |
| | Random Forest | 0.622 | 0.587 | 0.624 | 0.646 | 0.765 |
| | XGBoost | 0.604 | 0.573 | 0.606 | 0.648 | 0.766 |
| | MLP (deep learning) | 0.601 | 0.570 | 0.502 | 0.723 | 0.763 |
| After semester 1 | Logistic Regression | 0.703 | 0.666 | 0.667 | 0.793 | 0.866 |
| | Random Forest | 0.712 | 0.670 | 0.667 | 0.751 | 0.864 |
| | XGBoost | 0.712 | 0.668 | 0.685 | 0.764 | 0.863 |
| | MLP (deep learning) | 0.715 | 0.678 | 0.615 | 0.840 | 0.866 |
| After semester 2 | Logistic Regression | 0.736 | 0.697 | 0.690 | 0.812 | 0.887 |
| | Random Forest | 0.748 | 0.704 | 0.728 | 0.829 | 0.887 |
| | XGBoost | 0.736 | 0.691 | 0.709 | 0.807 | 0.881 |
| | MLP (deep learning) | 0.727 | 0.684 | 0.718 | 0.781 | 0.877 |

*Reading:* The more we know, the better the warning: ROC-AUC rises from about 0.76 (background only) to 0.87 (after semester 1) to 0.88 (after semester 2). All four models are close; the MLP is competitive but not better. Per-class F1 of the MLP after semester 1: Dropout 0.71, Graduate 0.82, **Still studying 0.50**, the hardest class. Top SHAP drivers of P(Dropout): at the early stage *fees up to date*, scholarship, age and course; after semester 1 the **number of semester-1 subjects cleared** (by far), then fees up to date; after semester 2 the semester-2 subjects cleared. The semester-1 stage is the realistic early warning; semester 2 is more accurate but later. Note the MLP's lower Dropout recall at some stages (for example 0.62 after semester 1) is a trade-off for higher precision; recall can be raised by lowering the decision threshold.

### 5.3 Model C - attendance risk (SIMULATED data; test n = 450; threshold tuned on validation)

| Model | Recall (At-Risk) | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.924 | 0.788 | 0.851 | 0.973 | 0.953 |
| Random Forest | 0.910 | 0.815 | 0.860 | 0.971 | 0.948 |
| XGBoost | 0.897 | 0.778 | 0.833 | 0.968 | 0.946 |
| MLP (flattened) | 0.917 | 0.756 | 0.829 | 0.965 | 0.940 |
| BiLSTM + static (deep learning) | 0.938 | 0.791 | 0.858 | 0.973 | 0.951 |

All models are essentially equal: in the simulation, early attendance is strongly predictive of final attendance by construction (attendance % is the top feature). The BiLSTM is the natural architecture for real, irregular weekly data. **Do not present these numbers as real-world performance.**

### 5.4 Model D - comment sentiment (real Coursera reviews, test n = 1,800, balanced)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.707 | 0.708 |
| Embedding + BiLSTM (deep learning) | 0.696 | 0.698 |
| DistilBERT fine-tuned (run `colab_distilbert.ipynb`, then insert here) | <fill in> | <fill in> |

Per-class F1 (BiLSTM): Negative 0.68, Neutral 0.61, Positive 0.81. Neutral (3-star) is intrinsically noisy: star ratings are a rough proxy for sentiment, and 3-star reviews mix praise and complaint. Top TF-IDF words: Negative - *not, no, poor, boring, waste*; Neutral - *but, however, bit, not bad, decent*; Positive - *great, excellent, awesome, amazing, love*. A pretrained transformer is expected to beat both on this task; report the real number from Colab.

## 6. System implementation (dashboard)

Pages: **Home** (example students), **How It Works** (data-to-model map, every field explained, what was left out, limits), **Student 360** (all four models, risk card, what-ifs, next steps), **Subject Forecast**, **Dropout Warning**, **Attendance Watch**, **Feedback Mood** (edit the record, see answer and reasons), **Class Overview** (score a class, compare groups, reality check against what really happened; upload your own CSV), **Inside the Network** (the real network drawn with its real size, follow one student, watch a tiny network learn, the code in plain words) and **Accuracy** (every model compared). A headless smoke test opens every page. Include dashboard screenshots and figures from `reports/figures/` (SHAP bars, confusion matrices, training curves).

## 7. Discussion

1. **Where deep learning helped:** nowhere decisively on table-like data. The MLPs tie the best classical model on subject forecasting and dropout, which agrees with Grinsztajn et al. (2022). It should help most on text (with DistilBERT) and on real, irregular sequences.
2. **Timing matters more than model choice.** Moving from "before exams" to "after semester 1" raises dropout ROC-AUC by 0.10 and cuts subject-forecast error by almost half, far more than swapping models.
3. **Metrics must match the decision.** Accuracy hides the hard "Still studying" class; recall and PR-AUC matter for dropout and attendance.
4. **Leakage and honesty:** clean-up is fitted on training data only; the test students are used only for reporting; a "no-model" baseline shows what the networks add.
5. **Fairness by design:** sensitive columns were removed after checking they add essentially nothing.

## 8. Limitations

* Trained on one Portuguese polytechnic's records; another college needs retraining on its own data (same code, different CSV).
* Model C is simulated; model D uses star ratings as noisy labels and a balanced sample.
* The risk card is a weighted heuristic; its weights are a design choice.
* Predictions are statistical estimates for groups of similar students, never verdicts on a person; use with human review.

## 9. Future scope

Real attendance logs (or OULAD) for model C; retraining on an Indian college's records; DistilBERT or multilingual models for Hindi-English feedback; calibrated probabilities and per-college thresholds; a database-backed deployment (FastAPI + React) with authentication; fairness audits and drift monitoring.

## 10. Conclusion

The project delivers a complete, reproducible, explainable early-warning system for colleges, with honest comparisons between deep and classical models, an explicit notion of *when* a warning is given, and a dashboard an adviser can read without knowing machine learning.

## References

1. Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data*, 7(11), 146.
2. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.
3. Romero, C., & Ventura, S. (2010). Educational data mining: a review of the state of the art. *IEEE Trans. Systems, Man, and Cybernetics, Part C*, 40(6), 601-618.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
5. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
6. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
7. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks*.
8. Chawla, N. V., et al. (2002). SMOTE: synthetic minority over-sampling technique. *JAIR*, 16, 321-357.
9. Akiba, T., et al. (2019). Optuna: a next-generation hyperparameter optimization framework. *KDD*.
10. Coursera course reviews dataset (107k reviews, 1-5 stars), public copy: github.com/sharmaroshan/Coursera-Reviews-Analysis.
