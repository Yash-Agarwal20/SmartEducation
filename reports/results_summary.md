# Results summary

**Data provenance**

- performance: university student records (real)
- dropout: university student records (real)
- sentiment: Coursera course reviews (real)
- attendance: simulated (documented simulation unless OULAD was used)

## A. Subjects cleared in semester 2 - Before any exam results  (test n=637, spread of the target SD=2.95)

| Model | RMSE (subjects) | MAE | R2 | Within +-1 subject | CV RMSE |
|---|---|---|---|---|---|
| Typical pass rate (no model) | 2.285 | 1.896 | 0.401 | 0.45 | 2.26 +- 0.04 |
| Ridge Regression | 1.902 | 1.456 | 0.585 | 0.60 | 1.89 +- 0.02 |
| Random Forest | 1.958 | 1.472 | 0.560 | 0.64 | 1.93 +- 0.04 |
| XGBoost | 1.871 | 1.411 | 0.598 | 0.65 | 1.86 +- 0.04 |
| MLP (deep learning) | 1.885 | 1.432 | 0.592 | 0.64 | 1.88 +- 0.04 |

## A. Subjects cleared in semester 2 - After semester 1  (test n=637, spread of the target SD=2.95)

| Model | RMSE (subjects) | MAE | R2 | Within +-1 subject | CV RMSE |
|---|---|---|---|---|---|
| Typical pass rate (no model) | 2.285 | 1.896 | 0.401 | 0.45 | 2.26 +- 0.04 |
| Ridge Regression | 1.013 | 0.690 | 0.882 | 0.88 | 1.11 +- 0.06 |
| Random Forest | 1.023 | 0.645 | 0.880 | 0.88 | 1.13 +- 0.06 |
| XGBoost | 0.970 | 0.645 | 0.892 | 0.90 | 1.10 +- 0.06 |
| MLP (deep learning) | 0.962 | 0.650 | 0.894 | 0.90 | 1.11 +- 0.04 |

## B. Dropout - Before any exam results (test n=664, imbalance=weights)

| Model | Accuracy | Macro-F1 | Dropout recall | Dropout precision | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.575 | 0.540 | 0.563 | 0.619 | 0.757 |
| Random Forest | 0.622 | 0.587 | 0.624 | 0.646 | 0.765 |
| XGBoost | 0.608 | 0.580 | 0.606 | 0.668 | 0.769 |
| MLP (deep learning) | 0.601 | 0.570 | 0.502 | 0.723 | 0.763 |

## B. Dropout - After semester 1 (test n=664, imbalance=weights)

| Model | Accuracy | Macro-F1 | Dropout recall | Dropout precision | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.703 | 0.666 | 0.667 | 0.793 | 0.866 |
| Random Forest | 0.712 | 0.670 | 0.667 | 0.751 | 0.864 |
| XGBoost | 0.711 | 0.669 | 0.671 | 0.781 | 0.863 |
| MLP (deep learning) | 0.715 | 0.678 | 0.615 | 0.840 | 0.866 |

## B. Dropout - After semester 2 (test n=664, imbalance=weights)

| Model | Accuracy | Macro-F1 | Dropout recall | Dropout precision | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.736 | 0.697 | 0.690 | 0.812 | 0.887 |
| Random Forest | 0.748 | 0.704 | 0.728 | 0.829 | 0.887 |
| XGBoost | 0.733 | 0.689 | 0.718 | 0.805 | 0.881 |
| MLP (deep learning) | 0.733 | 0.691 | 0.737 | 0.797 | 0.877 |

## C. Attendance risk (simulated; label: final attendance < 75%; test n=450)

| Model | Recall (At-Risk) | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.924 | 0.788 | 0.851 | 0.973 | 0.953 |
| Random Forest | 0.910 | 0.815 | 0.860 | 0.971 | 0.948 |
| XGBoost | 0.897 | 0.751 | 0.818 | 0.967 | 0.946 |
| MLP (flattened) | 0.917 | 0.756 | 0.829 | 0.965 | 0.940 |
| BiLSTM + static (deep learning) | 0.938 | 0.791 | 0.858 | 0.973 | 0.951 |

## D. Sentiment (test n=1800)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.707 | 0.708 |
| Embedding + BiLSTM (deep learning) | 0.703 | 0.705 |
