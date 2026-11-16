import streamlit as st
from sklearn.metrics import roc_auc_score

from utils import TASK1, TASK2, load_csv, load_json, load_model

st.set_page_config(page_title="British Airways Customer Analytics", page_icon="✈️", layout="wide")


@st.cache_data
def booking_test_auc() -> float:
    """ROC AUC of the deployed booking model on the 9,857 held-out test customers."""
    test = load_csv(TASK2 / "data" / "processed" / "test_set.csv")
    model = load_model(TASK2 / "models" / "booking_model.joblib")
    scores = model.predict_proba(test.drop(columns="booking_complete"))[:, 1]
    return roc_auc_score(test["booking_complete"], scores)


sentiment = load_json(TASK1 / "models" / "sentiment_model_info.json")
hybrid = sentiment["test_results"]["Hybrid (TF-IDF + VADER score)"]
vader = sentiment["test_results"]["VADER (word dictionary)"]

st.title("British Airways Customer Analytics")
st.markdown(
    "Two data science projects based on the **Forage British Airways job simulation**, "
    "rebuilt end to end with honest data handling, tested models and clear limitations."
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Task 1 · Reviews & lounges")
    st.markdown(
        "- **4,015** Skytrax reviews of British Airways (2014–2026), cleaned without inventing values\n"
        "- A sentiment model trained on passengers' own ratings, not on another tool's labels\n"
        "- A lounge demand lookup table for Heathrow Terminal 3 from 10,000 scheduled flights"
    )
    m1, m2 = st.columns(2)
    m1.metric("Sentiment macro F1 (held-out reviews)", f"{hybrid['Macro F1']:.3f}",
              f"{hybrid['Macro F1'] - vader['Macro F1']:+.3f} vs VADER alone")
    m2.metric("Reviews analysed", "4,015")

with col2:
    st.subheader("Task 2 · Booking prediction")
    st.markdown(
        "- **49,281** booking records (duplicates removed)\n"
        "- Gradient boosting model with leak-free preparation and calibrated probabilities\n"
        "- Campaign value analysis showing when targeting pays off"
    )
    m3, m4 = st.columns(2)
    m3.metric("ROC AUC (unseen customers)", f"{booking_test_auc():.3f}")
    m4.metric("Bookers reached in top 20%", "48%", "2.4× random")

st.divider()
st.markdown(
    "**Use the pages in the sidebar** to explore the review insights, try the sentiment model on your own text, "
    "plan lounge demand for a schedule, and test booking predictions and campaign scenarios."
)
st.caption(
    "Data: Skytrax (airlinequality.com) reviews scraped February 2026; schedule and booking datasets provided by "
    "Forage. Figures are for portfolio purposes and are not official British Airways data."
)