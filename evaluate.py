from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from analyzer import FeedbackAnalyzer

ROOT = Path(__file__).parent
DATA = pd.read_csv(ROOT / "data" / "sample_feedback.csv")
X_train, X_test, y_train, y_test = train_test_split(DATA.feedback, DATA.category, test_size=0.25, random_state=42, stratify=DATA.category)
majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_pred = majority.predict(X_test)
model = FeedbackAnalyzer().train(X_train, y_train)
pred = model.pipeline.predict(X_test)
report = ["Customer Feedback Analyzer — Holdout Evaluation", f"Rows: {len(DATA)} | Train: {len(X_train)} | Test: {len(X_test)}", "", f"Majority baseline accuracy: {accuracy_score(y_test, base_pred):.3f}", f"TF-IDF + Logistic Regression accuracy: {accuracy_score(y_test, pred):.3f}", "", "Classification report:", classification_report(y_test, pred, zero_division=0), "Confusion matrix (label order):", str(model.pipeline.classes_.tolist()), str(confusion_matrix(y_test, pred, labels=model.pipeline.classes_)), "", "Caution: synthetic data; metrics are not evidence of real-world performance."]
out = "\n".join(report)
print(out)
(ROOT / "results" / "evaluation.txt").write_text(out, encoding="utf-8")
