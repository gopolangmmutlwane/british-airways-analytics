import streamlit as st
from sklearn.metrics import roc_auc_score

from utils import TASK1, TASK2, load_csv, load_json, load_model

st.set_page_config(page_title="British Airways Customer Analytics", page_icon=":material/flight:", layout="wide")


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
    m1.metric("Sentiment F1", f"{hybrid['Macro F1']:.3f}",
              help="Macro F1 of the hybrid sentiment model on 471 held-out reviews.")
    m2.metric("Reviews", "4,015", help="Skytrax reviews after cleaning.")
    st.caption(f"Sentiment F1 is measured on 471 held-out reviews, {hybrid['Macro F1'] - vader['Macro F1']:.3f} "
               f"higher than VADER alone ({vader['Macro F1']:.3f}).")

with col2:
    st.subheader("Task 2 · Booking prediction")
    st.markdown(
        "- **49,281** booking records (duplicates removed)\n"
        "- Gradient boosting model with leak-free preparation and calibrated probabilities\n"
        "- Campaign value analysis showing when targeting pays off"
    )
    m3, m4 = st.columns(2)
    m3.metric("ROC AUC", f"{booking_test_auc():.3f}",
              help="Measured on 9,857 customers the model never saw during training.")
    m4.metric("Top 20% reach", "48%",
              help="Share of all bookers found among the 20% of customers with the highest scores.")
    st.caption("ROC AUC is measured on 9,857 unseen customers. The top 20% of scores contains "
               "48% of all bookers, 2.4 times better than random targeting.")

st.divider()
st.markdown(
    "**Use the pages in the sidebar** to explore the review insights, try the sentiment model on your own text, "
    "plan lounge demand for a schedule, and test booking predictions and campaign scenarios."
)
st.caption(
    "Data: Skytrax (airlinequality.com) reviews scraped February 2026; schedule and booking datasets provided by "
    "Forage. Figures are for portfolio purposes and are not official British Airways data. "
    "Code: [github.com/gopolangmmutlwane/british-airways-analytics]"
    "(https://github.com/gopolangmmutlwane/british-airways-analytics)"
)