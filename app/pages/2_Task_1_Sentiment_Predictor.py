import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import TASK1, load_csv, load_json, load_model

st.set_page_config(page_title="Sentiment Predictor", page_icon="✈️", layout="wide")
st.title("Task 1 · Sentiment Predictor")
st.caption(
    "Type a short review title and the model estimates whether the passenger had a positive or negative trip. "
    "It was trained on 2,354 Skytrax review titles, labelled using the passengers' own star ratings, and "
    "combines what it learned with VADER's general sentiment score."
)

model = load_model(TASK1 / "models" / "sentiment_model.joblib")
info = load_json(TASK1 / "models" / "sentiment_model_info.json")
vader = SentimentIntensityAnalyzer()

EXAMPLES = ["the crew were wonderful", "The crew was crap", "seat was cramped but staff were lovely",
            "no complaints at all", "delayed for 6 hours with no explanation"]


@st.cache_data
def real_reviews() -> pd.DataFrame:
    df = load_csv(TASK1 / "data" / "processed" / "reviews_clean.csv")
    df = df[df["has_informative_title"] & (df["n_ratings"] >= 3)]
    return df[(df["avg_rating"] <= 2.5) | (df["avg_rating"] >= 3.5)][["review_title", "avg_rating", "seat_type"]]


def model_input(text: str) -> pd.DataFrame:
    """The hybrid model needs the title and its VADER score."""
    return pd.DataFrame({"title": [text], "vader": [vader.polarity_scores(text)["compound"]]})


def contributions(text: str) -> pd.Series:
    """How much each word, phrase and the VADER score pushed the prediction (letter patterns not shown)."""
    features = model.named_steps["features"]
    words = features.named_transformers_["text"].transformer_list[0][1]
    text_weight = features.transformer_weights["text"]
    coef = model.named_steps["classifier"].coef_[0]
    names = words.get_feature_names_out()

    values = words.transform([text]).toarray()[0] * text_weight
    present = np.nonzero(values)[0]
    pushes = pd.Series(values[present] * coef[present], index=names[present])
    pushes["VADER score"] = vader.polarity_scores(text)["compound"] * features.transformer_weights["vader"] * coef[-1]
    return pushes.sort_values()


st.session_state.setdefault("review_text", EXAMPLES[0])


def use_example(text):
    st.session_state.review_text = text


def use_random_review():
    row = real_reviews().sample(1).iloc[0]
    st.session_state.review_text = row["review_title"]


# ---------------- Input ----------------
st.markdown("**Try an example**")
cols = st.columns(len(EXAMPLES) + 1)
for col, example in zip(cols, EXAMPLES):
    col.button(example, on_click=use_example, args=(example,), width="stretch")
cols[-1].button("🎲 Random real review", on_click=use_random_review, width="stretch")

text = st.text_input("Review title", key="review_text", max_chars=150)

if not text.strip():
    st.info("Type a review title above to get a prediction.")
    st.stop()

# ---------------- Prediction ----------------
prob = float(model.predict_proba(model_input(text))[0, 1])
compound = vader.polarity_scores(text)["compound"]

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Our hybrid model")
    label = "Positive 😊" if prob >= 0.5 else "Negative 😞"
    st.metric("Prediction", label, f"{prob:.0%} chance positive", delta_color="off", delta_arrow="off")
    st.progress(prob)
with right:
    st.subheader("VADER alone (word dictionary)")
    vader_label = "Positive 😊" if compound >= 0.05 else ("Negative 😞" if compound <= -0.05 else "Neutral 😐")
    st.metric("Prediction", vader_label, f"compound score {compound:+.2f}", delta_color="off", delta_arrow="off")
    st.progress((compound + 1) / 2)

# If the text is a real review title from the data, show what the passenger actually rated
match = real_reviews()["review_title"].str.lower() == text.strip().lower()
if match.any():
    rating = real_reviews().loc[match, "avg_rating"].iloc[0]
    truth = "positive" if rating >= 3.5 else "negative"
    st.success(f"This is a real review. The passenger's average star rating was **{rating:.1f}**, "
               f"so its true label is **{truth}**. The final model was trained on all labelled reviews, so it "
               f"has seen this one before. The honest accuracy figures are in the box at the bottom.")

# ---------------- Explanation ----------------
st.subheader("Why did the model decide this?")
contrib = contributions(text)
chart = contrib.rename("push").reset_index().rename(columns={"index": "word, phrase or VADER"})
chart["direction"] = np.where(chart["push"] > 0, "towards positive", "towards negative")
fig = px.bar(chart, x="push", y="word, phrase or VADER", orientation="h", color="direction",
             color_discrete_map={"towards positive": "#2166ac", "towards negative": "#b2182b"},
             labels={"push": "Push on the prediction", "word, phrase or VADER": ""}, height=120 + 35 * len(chart))
fig.update_layout(legend_title_text="")
st.plotly_chart(fig, width="stretch")
st.caption("The model combines the words it learned from airline reviews with VADER's general sentiment score. "
           "It also uses letter patterns, which are harder to display, so the bars do not add up exactly to the "
           "prediction. Words that never appeared in the training reviews have no bar of their own, but VADER "
           "may still know them.")

# ---------------- Model card ----------------
with st.expander("How good is this model, and what are its limits?"):
    results = pd.DataFrame(info["test_results"]).T
    st.dataframe(results.style.format("{:.3f}"), width="stretch")
    st.markdown(
        f"- Cross-validated macro F1 **{info['cv_macro_f1']:.3f}** (original text-only model "
        f"{info['cv_macro_f1_original_model']:.3f}). On the 471-review test set the two models scored about the "
        "same, so expect roughly 0.82–0.85 on new reviews.\n"
        "- **Why a hybrid?** The text-only model learned that titles mentioning the crew are usually positive and "
        "ignored words it had never seen, so it scored *\"The crew was crap\"* as 87% positive. Adding VADER's "
        "score fixed this and similar cases.\n"
        "- It reads **short titles**, not full reviews, because full review text could not be collected.\n"
        "- Slang unknown to both VADER and the training data, such as *\"rubbish\"*, can still be misread.\n"
        "- Some titles disagree with the passenger's overall rating, so no model can be right on every review."
    )