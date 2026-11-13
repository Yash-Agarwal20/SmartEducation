# Understand this project in 20 minutes

Read this before the viva. Everything here matches the code and the dashboard (the dashboard's **How It Works** and **Inside the Network** pages say the same things with live examples).

## 1. What it is, and who it is for

An **early-warning assistant for college advisers**. For a *college or university* student (degree programme, semesters, subjects, fees, attendance) it answers four questions, then combines the answers into one risk card with the reasons:

| | Question | Plain name in the dashboard |
|---|---|---|
| **A** | Of the subjects this student registered for in semester 2, how many will they clear? | Subject forecast |
| **B** | Will this student drop out, still be studying, or graduate? | Dropout warning |
| **C** | Judging by the first 8 weeks, will their attendance end below 75%? | Attendance watch |
| **D** | Is a student's comment positive, neutral or negative, and which words decided it? | Feedback mood |

It is **not** for schools. (An earlier version mixed in a school dataset, which is why the fields felt confusing. That is removed.)

## 2. Which data trains which model (the question you asked)

| Model | Trained on | Rows | Real or made-up? | What the model sees | What it outputs |
|---|---|---|---|---|---|
| **A** Subject forecast | University student records | 4,244 students who registered for semester-2 subjects | **Real** | Course, age, admission route, fees/dues/scholarship, (semester-1 results), subjects registered for semester 2 | Expected subjects cleared, expected backlog |
| **B** Dropout warning | The **same** university records | 4,424 students whose final outcome is known | **Real** | Same kind of record, at 3 moments in time (see below) | 3 chances adding to 100%: drops out / still studying / graduates |
| **C** Attendance watch | Generated students (rules in `smartedu/synthetic.py`) | 3,000 | **Simulated** - no public attendance data exists | 8 weeks of attendance %, assignments submitted, portal logins + distance, part-time job, hostel, scholarship, semester, last semester's attendance | Chance final attendance < 75% |
| **D** Feedback mood | Coursera course reviews | 12,000 (4,000 per mood) from 107,000 | **Real** | Free text | 3 chances: negative / neutral / positive + deciding words |

The real university file is *Predict Students' Dropout and Academic Success* (Realinho et al., 2022): real students of a Portuguese polytechnic. The fields (course, fees, scholarship, semester results) exist at every college, but the learned numbers come from that institute. To use it at your own college you retrain on your college's CSV with the same code.

### The "what do we know about the student?" idea (stages)
A warning is only useful if it comes early, so models A and B are trained separately for each moment:

* **Before any exam results** - only background, money matters, and subjects registered. A rough first screen.
* **After semester 1** - adds the semester-1 record. The realistic early warning: there is still time to help.
* **After semester 2** (dropout only) - most accurate, but latest.

### What we deliberately left out (and why)
Gender, marital status, nationality, international/special-needs flags, parents' qualification/occupation, and national economy figures (unemployment, inflation, GDP). Sensitive attributes should not be used to judge a student, and the economy numbers are identical for everyone who joined in the same year. I checked: adding the sensitive columns back changes dropout ROC-AUC by only +0.001 (semester 1) and +0.0006 (semester 2), so nothing useful is lost.

## 3. One student's journey through the code

```
CSV file -> pick the columns for this stage -> split 70/15/15 -> clean-up -> neural network -> prediction -> reasons
```

1. **Load** (`smartedu/data_loaders.py`): reads `data/raw/dropout.csv`.
2. **Pick columns** (`smartedu/config.py`, `STAGES`): the list of fields allowed at each stage.
3. **Split** (`utils.split_indices`): 70% train, 15% validation, 15% test (stratified, seed 42). The network learns from train, validation decides when to stop, test is used only at the end to report honest scores on students it never saw.
4. **Clean-up** (`tabular.py`, `build_preprocessor`): missing numbers filled with the median, numbers put on one scale (StandardScaler), categories like *Course* turned into 0/1 columns (one-hot). It is fitted on the training part only, so nothing leaks from the test students.
5. **Network** (`nn.py`): see section 4.
6. **Train** (`nn.fit_loop`): see section 4.
7. **Evaluate** (`performance.py`, `dropout.py`, ...): compare the network with simpler models on the test students and write `reports/results_summary.md` and figures.
8. **Save**: the clean-up + network + what the dashboard needs are saved together as one "bundle" in `models/*.joblib`.
9. **Dashboard** (`app/`): loads the bundle, takes the fields a person types, runs steps 4-5 for that one student, and **SHAP** explains which facts pushed the result up or down.

## 4. How the deep-learning code works

### 4.1 The idea
A neural network is layers of small calculators. Each takes numbers, multiplies each by a **weight** (how much it cares), adds them up and passes the result on. At the start the weights are random, so answers are nonsense. **Training** shows it thousands of past students whose outcome we know, and after each small batch it nudges every weight so its answers move closer to the truth.

### 4.2 The MLP (models A and B: table of facts about a student)
`nn.MLP` is a stack of blocks: **Linear -> BatchNorm -> ReLU -> Dropout**.

* *Linear*: the weighted sums (the calculators).
* *BatchNorm*: keeps the numbers between layers well-behaved so training is stable.
* *ReLU*: a switch that lets only positive signals through; this is what lets the network learn curved, non-obvious patterns instead of just straight lines.
* *Dropout (30%)*: randomly switches off 30% of calculators during training so the network cannot simply memorise students.

The dropout model at the "before exams" stage is 48 inputs -> 128 -> 64 -> 32 -> 3 outputs, which is **17,155 learned numbers** (the dashboard's Inside the Network page shows the same count). The three outputs go through *softmax*, which turns them into three chances that add to 100%. Model A has one output (subjects cleared).

### 4.3 The training loop (`nn.fit_loop`, in plain words)
Repeat for each **epoch** (one pass over all training students):
1. Shuffle the students and take a **batch** (64 at a time).
2. **Forward pass**: push the batch through the network to get predictions.
3. **Loss**: measure how wrong it was. Classification uses cross-entropy with **class weights** (so rare outcomes like "dropout" count more); regression uses mean squared error.
4. **Backward pass** (`loss.backward()`): work out, for every weight, which direction reduces the loss (back-propagation).
5. **Optimizer step** (AdamW): nudge every weight a little in that direction.
6. After the epoch, measure the loss on the **validation** students. If it has not improved for `patience` epochs, **stop early** and restore the best weights (this is what prevents overfitting).

### 4.3b Why the dropout class weights matter
Only about 32% of students drop out, 18% are "still studying", 50% graduate. Without weights a network can score well by almost never predicting "dropout", which defeats the purpose. Weighting makes missing a dropout cost more.

### 4.4 The BiLSTM (model C: attendance, week after week)
An **LSTM** reads a sequence in order and keeps a memory, so "84, 80, 75, 70" is understood as a *falling trend*, not four unrelated numbers. **Bi**directional means it also reads the weeks backwards. Each week has 3 signals (attendance %, assignments, portal logins). The BiLSTM summary of the 8 weeks (mean and last step) is joined with a small network over the static facts (distance, job, hostel...) and a final head gives the chance of "At-Risk" (`nn.SeqRiskNet`). The decision threshold is tuned on the validation set to favour recall (catching at-risk students).

### 4.5 Text (model D: comments)
Words are first turned into number-lists (an **embedding**: similar words get similar lists, learned during training), then a BiLSTM reads the sentence in order, so "not good" is understood differently from "good" (`sentiment.TextNet`). The dashboard highlights deciding words by removing each word in turn and watching how the prediction changes (**occlusion**).

### 4.6 Why the "baselines"?
Every module is compared against simpler models (Logistic/Ridge Regression, Random Forest, XGBoost, TF-IDF, and for model A even a "no model" guess). That tells us honestly whether deep learning is worth it. **It is not always**: on table-like data the simpler models score about the same (see Accuracy page). Where deep networks should shine is text and sequences.

## 5. Where things are in the code

| File | What it does |
|---|---|
| `train_all.py` | Trains everything and writes the results tables |
| `smartedu/config.py` | Column lists per stage, thresholds, weights of the risk card |
| `smartedu/data_loaders.py` | Reads the CSVs |
| `smartedu/synthetic.py` | The simulated attendance generator (rules documented inside) |
| `smartedu/tabular.py` | Clean-up pipeline, saved "bundle", SHAP and permutation importance |
| `smartedu/nn.py` | The neural networks and the training loop |
| `smartedu/performance.py`, `dropout.py`, `attendance.py`, `sentiment.py` | One module each: data -> train -> compare -> save |
| `smartedu/glossary.py` | Plain-English names and help text for every dashboard field |
| `smartedu/about.py` | The plain-language description of each model (used everywhere, so docs cannot disagree) |
| `smartedu/risk.py` | The combined risk score |
| `smartedu/ui.py`, `app/` | The dashboard |

## 6. Short forms decoded

* **MLP** - multi-layer perceptron, the standard neural network for tables.
* **BiLSTM** - bidirectional long short-term memory network, for sequences.
* **SHAP** - a method that says how much each fact pushed one prediction up or down.
* **RMSE / MAE** - typical error in the same units as the target (here: subjects). Lower is better. **R²** - share of the differences between students that the model captures (1 = perfect, 0 = no better than guessing the average).
* **Recall** - of the students who really drop out, how many did we catch? **Precision** - of the students we flagged, how many really did?
* **ROC-AUC** - how well the model ranks risky students above safe ones (0.5 = coin flip, 1 = perfect).
* **Macro-F1** - an average of per-class scores that treats every class equally, useful when classes are unbalanced.
* **CV (cross-validation)** - repeating the test on 5 different splits to show the result is not luck.
* **TF-IDF** - a classic way to turn text into numbers by word counts; our text baseline.
* **Epoch / batch / loss** - one pass over the data / a small group of students / the number training tries to shrink.

## 7. See it yourself (5 minutes)

1. `streamlit run app/Home.py` -> **How It Works** tab "Which data trains which model".
2. **Inside the Network** -> "Follow one student" and "Playground: watch a network learn".
3. **Dropout Warning** -> switch the stage from "Before any exam results" to "After semester 2" and watch the chances sharpen.
4. **Accuracy** -> see every deep model next to the simpler ones.

## 8. Honest limits (say these in the viva before anyone asks)

* Trained on one institute's records; another college needs retraining on its own data.
* Attendance is simulated: it shows the method works, not how accurate it would be at your college.
* The overall risk score is a weighted average you can adjust, not a separately trained model.
* Predictions describe patterns in past students: an estimate for similar students, never a verdict on a person.
