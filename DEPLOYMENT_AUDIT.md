# Deployment audit notes

## Included in this archive
- Streamlit entrypoint: `app/Home.py`
- Nine dashboard pages under `app/pages/`
- `smartedu/` source package
- Six tabular/attendance model bundles plus the BiLSTM sentiment bundle
- Raw data, processed feedback sample, metrics, plots, and project/viva documentation
- `.streamlit/config.toml`

## Verified in the audit container
- 86 archive entries; approximately 26.6 MB uncompressed
- All 30 Python files compile successfully
- `dropout.csv` contains 4,424 rows and 35 columns
- `feedback.csv` contains 12,000 rows; `coursera_reviews_full.csv` contains 107,018 rows
- App loader reads `data/raw/feedback.csv`; the full CSV is the source from which the sample is made
- No `.env`, `.venv`, or `.git` directory was present in the uploaded archive

## Important compatibility note
The saved model bundles were created with scikit-learn 1.9.1 and the environment snapshot records NumPy 2.5.4, pandas 3.1.0, PyTorch 2.15.0, XGBoost 3.4.1, and joblib 1.6.0. Keep the deployment environment aligned with these versions because the model files are serialized Python objects.

Some model bundles import `pyarrow` while being deserialized. It is therefore explicitly included in the deployment requirements.

## Not yet verified
- Full Streamlit smoke test: Streamlit is not installed in the audit container.
- Full model-bundle loading against the exact pinned dependency set: the audit container has older scikit-learn and does not have pyarrow, so testing there would produce misleading compatibility warnings/errors.
- Cloud deployment: not yet deployed.

## Windows issue
The original Windows Smart App Control block on `torch_python.dll` is not fixed by these deployment files. A Linux deployment can bypass that Windows-specific restriction for the hosted app, but local Windows execution remains a separate issue.
