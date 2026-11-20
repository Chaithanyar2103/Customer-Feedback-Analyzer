# Customer Feedback Analyzer

A CPU-friendly baseline ML project that classifies retail/customer feedback into **Product Issue**, **Staff Praise**, **Delivery Issue**, or **General Feedback**. It includes a Streamlit UI for single text and batch CSV classification, a synthetic labeled dataset, automated tests, an evaluation script, saved metrics, and GitHub Actions CI.


## Reproduce locally (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python evaluate.py
streamlit run app.py
```

Streamlit prints a local URL, usually http://localhost:8501.

## Batch CSV format
Upload a CSV with a column named `feedback`. The app adds `category` and `confidence` and allows you to download the result.

## Dataset
`data/sample_feedback.csv` contains 96 synthetic labeled examples (24 per class). Synthetic examples are intentionally simple and should not be treated as representative of real customers. Replace/add permissioned real feedback and label it before deployment.


## Method
- TF-IDF unigram/bigram text features
- Logistic Regression classifier with balanced class weights
- Stratified train/test evaluation
- Baseline comparison: majority-class classifier


## Evaluation results
Run `python evaluate.py` to regenerate `results/evaluation.txt`. Metrics are reported from a held-out split of the bundled synthetic data. Because synthetic data is small and patterned, results are demonstration-only and likely overestimate real-world performance.


## Tests
`pytest -q` verifies model training, category predictions, confidence bounds, blank input validation, and untrained-model behavior.


## Project milestones
1. Dataset design and label definition
2. Baseline majority-class evaluation
3. TF-IDF feature pipeline
4. Logistic Regression classifier and prediction interface
5. Streamlit single and batch analysis UI
6. Automated tests, evaluation report, and CI workflow

## Limitations
The model can misclassify nuanced, mixed, sarcastic, or out-of-domain comments. Confidence is not calibrated. Use human review for business decisions.
