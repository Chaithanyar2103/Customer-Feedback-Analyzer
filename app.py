import pandas as pd
import streamlit as st
from analyzer import FeedbackAnalyzer
from pathlib import Path


st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="💬", layout="wide")
st.title("Customer Feedback Analyzer")
st.write("Classify customer feedback, inspect category trends, and export batch results.")
DATA_PATH = Path(__file__).parent / "data" / "sample_feedback.csv"

@st.cache_resource
def get_model():
    model = FeedbackAnalyzer().train_from_csv(DATA_PATH)
    return model

model = get_model()
labels = ["Product Issue", "Staff Praise", "Delivery Issue", "General Feedback"]
mode = st.sidebar.radio("Choose analysis", ["Single feedback", "Batch CSV"])
if mode == "Single feedback":
    user_text = st.text_area("Feedback text", placeholder="The delivery arrived two days late...")
    if st.button("Classify feedback", type="primary", disabled=not user_text.strip()):
        category, score = model.predict(user_text)
        st.subheader(category)
        st.metric("Model confidence", f"{score:.1%}")
        st.caption("Confidence is the model's probability estimate, not a calibrated certainty.")
else:
    uploaded = st.file_uploader("Upload a CSV containing a 'feedback' column", type=["csv"])
    if uploaded:
        incoming = pd.read_csv(uploaded)
        if "feedback" not in incoming.columns:
            st.error("CSV needs a column named 'feedback'.")
        elif st.button("Analyze CSV", type="primary"):
            texts = incoming.feedback.fillna("").astype(str)
            preds = [model.predict(t) if t.strip() else ("Blank Feedback", 0.0) for t in texts]
            incoming["category"] = [p[0] for p in preds]
            incoming["confidence"] = [p[1] for p in preds]
            st.dataframe(incoming, use_container_width=True)
            st.bar_chart(incoming.category.value_counts())
            st.download_button("Download classified CSV", incoming.to_csv(index=False), "classified_feedback.csv", "text/csv")
st.divider()
st.subheader("Dataset overview")
df = pd.read_csv(DATA_PATH)
left, right = st.columns(2)
left.metric("Labeled examples", len(df))
left.metric("Categories", df.category.nunique())
right.bar_chart(df.category.value_counts())
st.dataframe(df.sample(min(8, len(df)), random_state=7), use_container_width=True)
st.info("The bundled dataset is synthetic and for demonstration only. Validate with real, consented, representative data before making decisions.")
