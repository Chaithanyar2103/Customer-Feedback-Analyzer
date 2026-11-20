"""Customer feedback classification baseline using TF-IDF and Logistic Regression."""
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class FeedbackAnalyzer:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ])
        self.trained = False

    def train(self, texts, labels):
        texts = [str(t) for t in texts]
        labels = [str(y) for y in labels]
        if len(texts) < 4 or len(set(labels)) < 2:
            raise ValueError("Training requires at least four examples and two categories.")
        self.pipeline.fit(texts, labels)
        self.trained = True
        return self

    def train_from_csv(self, path):
        df = pd.read_csv(path)
        if not {"feedback", "category"}.issubset(df.columns):
            raise ValueError("CSV must contain feedback and category columns.")
        df = df.dropna(subset=["feedback", "category"])
        return self.train(df.feedback, df.category)

    def predict(self, text):
        if not self.trained:
            raise RuntimeError("Train the model before predicting.")
        text = str(text).strip()
        if not text:
            raise ValueError("Feedback cannot be blank.")
        probs = self.pipeline.predict_proba([text])[0]
        i = probs.argmax()
        return str(self.pipeline.classes_[i]), float(probs[i])
