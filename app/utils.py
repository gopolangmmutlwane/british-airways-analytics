"""Shared helpers for the Streamlit app: file locations and cached loading."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent          # the project folder
TASK1 = ROOT / "task1_reviews_sentiment"
TASK2 = ROOT / "task2_booking_prediction"

BLUE = "#4a6fa5"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)