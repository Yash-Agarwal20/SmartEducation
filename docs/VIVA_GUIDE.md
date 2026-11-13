# Viva guide - Smart Education & Student Intelligence System

Short, honest answers. Numbers are from the real run (`reports/results_summary.md`). If you retrain, re-check them. Read `docs/PROJECT_GUIDE.md` first so the answers make sense to you, not just sound right.

## 30-second pitch
"It is an early-warning assistant for college advisers. Four neural networks answer four questions: how many subjects a student will clear, whether they will drop out, whether their attendance will fall below 75%, and how students feel about their courses. Each is compared with simpler models, every prediction comes with plain-language reasons, and a risk card combines the four. The key design idea is *when* the warning is given: before any exam, after semester 1, or after semester 2."

## Questions you will almost certainly get

**1. Is this for schools or colleges? Which data did you use?**
Colleges and universities. The subject forecast and dropout models use a real dataset of 4,424 students of a Portuguese polytechnic (Realinho et al., 2022): course, age, admission route, fees, scholarship, and semester results. The comments model uses 12,000 real Coursera reviews. Attendance is simulated.

**2. Why is the data from Portugal? Is that grounded for an Indian college?**
No public Indian student-record dataset of this kind exists. The fields (course, fees, scholarship, semester results) exist at every college, so the system works on any college's CSV, but the learned numbers come from that polytechnic. For a real deployment you retrain on your own college's records with the same code. I state this in the dashboard and the report.

**3. Which data trains which model?**
A and B both learn from the same university file (A: the 4,244 students who registered for semester-2 subjects; B: all 4,424). C learns from 3,000 simulated students. D learns from 12,000 real reviews. (Table in `PROJECT_GUIDE.md` section 2.)

**4. What is a "stage" and why does it matter?**
It is how much we know about the student: before any exam results, after semester 1, or (dropout only) after semester 2. A warning is only useful if it comes early, so each model is trained separately per stage. The results show why: dropout ROC-AUC rises from 0.76 to 0.87 to 0.88, and the subject-forecast error falls from 1.89 to 0.96 subjects.

**5. Why deep learning when simpler models score about the same?**
Honestly: on table-like data they do, and my results show it (for example XGBoost 0.964 vs MLP 0.964 subjects after semester 1; dropout AUC 0.863 vs 0.866). I compared against baselines precisely to find out instead of assuming. Deep learning is justified where the data has structure simple models cannot use: sequences (weekly attendance) and text (comments, especially with a pretrained model like DistilBERT).

**6. Which parts are real and which are simulated?**
Real: university records and Coursera reviews. Simulated: attendance, with rules written in `synthetic.py` (engagement, commute, part-time job, a fading group, a late shock). Its high scores show the method works, not real-world accuracy. The dashboard says so on the attendance page.

**7. Why did you leave out gender, nationality and parents' details?**
A risk tool should not judge students by sensitive attributes. I also tested it: adding them back changes dropout ROC-AUC by only +0.001 (after semester 1) and +0.0006 (after semester 2), so nothing useful is lost. National unemployment/inflation/GDP were left out because they are identical for everyone who joined in the same year.

**8. Why recall for dropout?**
Missing a student who is about to leave costs more than a false alarm that leads to a friendly conversation. The dashboard reports recall next to precision. The hardest class is "still studying" (F1 about 0.50), which is typical for this dataset.

**9. Explain your neural network.**
The MLP is blocks of Linear -> BatchNorm -> ReLU -> Dropout. Linear layers are weighted sums, ReLU lets it learn curved patterns, BatchNorm stabilises training, Dropout (30%) stops it memorising. The dropout model is 48 inputs -> 128 -> 64 -> 32 -> 3 outputs (17,155 learned numbers) with softmax giving three chances that add to 100%.

**10. How does training work?**
Shuffle the students, take a batch of 64, run it through the network, measure the error (cross-entropy with class weights, or mean squared error for subjects), back-propagate to see how each weight should change, and let the AdamW optimizer nudge the weights. After each epoch check the validation students; if they stop improving, stop early and restore the best weights. (Section 4 of `PROJECT_GUIDE.md`; the dashboard's Inside the Network page shows it live.)

**11. Why a BiLSTM for attendance and text?**
Order matters. Attendance over weeks is a sequence, so "84, 80, 75, 70" is a falling trend; text is a sequence of words, so "not good" differs from "good". An LSTM keeps a memory across steps; bidirectional also reads backwards.

**12. How do you avoid data leakage and overfitting?**
The split is made first (70/15/15, stratified); clean-up is fitted on training students only; the validation set drives early stopping and the attendance threshold; the test students are used only to report. Dropout layers and weight decay reduce overfitting. For model A I also report 5-fold cross-validation.

**13. How do you handle class imbalance?**
Class-weighted loss for B, C and D (only about 32% of students drop out and 18% are "still studying"). SMOTE is implemented as an option (`--imbalance smote`).

**14. What is SHAP?**
It says how much each fact pushed one prediction up or down. For dropout, the number of semester-1 subjects cleared is by far the strongest, then fees up to date. The dashboard shows these in words ("Sem 1 subjects cleared = 6 lowers the dropout chance by about 14 points").

**15. How is the risk card computed? Is it a model?**
No. It is a weighted average of the four risks (default dropout 0.35, subjects 0.30, attendance 0.25, comments 0.10; editable), renormalised over the active models. The datasets describe different students, so a single trained fusion model is impossible with this data. The weighting is transparent and adjustable.

**16. Your BiLSTM only matches TF-IDF on comments. Is that a failure?**
No. 12,000 short reviews is small for a network trained from scratch, and TF-IDF + logistic regression is a strong baseline. That is why pretrained transformers exist; the Colab notebook fine-tunes DistilBERT on the same split for a fair comparison.

**17. How accurate is it, in one sentence each?**
Subjects: after semester 1, typical error about 1 subject, 90% within one. Dropout: ROC-AUC 0.87 after semester 1. Attendance (simulated): ROC-AUC 0.97. Comments: macro-F1 0.70 (3 classes, balanced).

**18. How would you deploy it at a real college?**
Retrain on that college's records (same code, different CSV), put it behind a service (FastAPI) with authentication, schedule retraining, monitor drift and fairness, and keep a human adviser in the loop.

**19. What would you improve?**
Real attendance logs or OULAD, Indian college records, DistilBERT/multilingual comments (Hindi-English), calibrated probabilities and per-college thresholds.

## Traps to avoid
* Do not call the attendance results real. Say simulated.
* Do not claim deep learning "won" on tables. It tied.
* Do not say it "predicts who will drop out". Say it estimates risk, as a group-level estimate.
* Do not say it is trained on your college. It is trained on a Portuguese polytechnic; retraining on yours is the intended use.
* If asked for DistilBERT numbers, run the Colab notebook first and fill the report table.

## Demo script (4 minutes)
1. **Home:** pick the four example students; show the overall risk and the reasons.
2. **How It Works:** tab "Which data trains which model", then "What every field means".
3. **Dropout Warning:** switch the stage from *Before any exam results* to *After semester 2* and watch the chances sharpen; change *Tuition fees paid up to date*.
4. **Inside the Network:** the real network, then "Follow one student".
5. **Student 360:** turn on attendance and a negative comment and watch the risk card and next steps change.
6. **Class Overview:** the sample class, then the reality check (predicted vs actual dropout).
7. **Accuracy:** show the deep models next to the simpler ones and say plainly where they tie.
