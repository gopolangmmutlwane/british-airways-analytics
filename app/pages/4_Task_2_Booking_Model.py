import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from utils import BLUE, TASK2, load_csv, load_json, load_model

st.set_page_config(page_title="Booking Model", page_icon="✈️", layout="wide")
st.title("Task 2 · Booking Prediction")
st.caption(
    "A gradient boosting model trained on 39,424 booking records estimates each customer's chance of completing "
    "a booking. Its scores are calibrated, so a 30% score means about 3 in 10 such customers book."
)

model = load_model(TASK2 / "models" / "booking_model.joblib")
info = load_json(TASK2 / "models" / "booking_model_info.json")
THRESHOLD = info["threshold"]
DAYS = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}


@st.cache_data
def reference_data():
    df = load_csv(TASK2 / "data" / "processed" / "bookings_clean.csv")
    countries = df["booking_origin"].value_counts().index.tolist()
    routes = df.groupby("route")["flight_duration"].median().sort_index()
    base_rate = df["booking_complete"].mean()
    return countries, routes, base_rate


@st.cache_data
def scored_test_set():
    test = load_csv(TASK2 / "data" / "processed" / "test_set.csv")
    y = test["booking_complete"].to_numpy()
    scores = model.predict_proba(test.drop(columns="booking_complete"))[:, 1]
    return y, scores


def money(x: float) -> str:
    """£1,234 or −£1,234."""
    return f"−£{abs(x):,.0f}" if x < 0 else f"£{x:,.0f}"


countries, routes, base_rate = reference_data()
y_test, scores = scored_test_set()

tab_predict, tab_campaign, tab_quality = st.tabs(
    ["Predict a customer", "Campaign simulator", "How good is the model?"]
)

# ======================================================================== Predict
with tab_predict:
    st.markdown("Describe a customer's search and the model estimates how likely they are to complete the booking.")
    left, right = st.columns(2, gap="large")
    with left:
        country = st.selectbox("Booking from (country)", countries, index=countries.index("Malaysia"),
                               help="Listed from most to fewest customers in the data.")
        route = st.selectbox("Route (origin + destination airport codes)", routes.index.tolist(),
                             index=routes.index.tolist().index("KULPER") if "KULPER" in routes.index else 0)
        channel = st.radio("Sales channel", ["Internet", "Mobile"], horizontal=True)
        trip = st.radio("Trip type", ["RoundTrip", "OneWay", "CircleTrip"], horizontal=True)
        passengers = st.slider("Passengers", 1, 9, 1)
    with right:
        lead = st.slider("Days between booking and flight", 0, 365, 30)
        stay = st.slider("Nights at destination", 0, 60, 6)
        hour = st.slider("Departure hour", 0, 23, 9)
        day = st.select_slider("Departure day", list(DAYS), value="Wed")
        st.markdown("**Extras chosen**")
        e1, e2, e3 = st.columns(3)
        baggage = e1.checkbox("Baggage", value=True)
        seat = e2.checkbox("Preferred seat")
        meals = e3.checkbox("Meals")

    customer = pd.DataFrame([{
        "num_passengers": passengers, "sales_channel": channel, "trip_type": trip,
        "purchase_lead": lead, "length_of_stay": stay, "flight_hour": hour, "flight_day": DAYS[day],
        "booking_origin": country, "wants_extra_baggage": int(baggage), "wants_preferred_seat": int(seat),
        "wants_in_flight_meals": int(meals), "flight_duration": float(routes[route]),
        "origin": route[:3], "destination": route[3:],
    }])
    chance = float(model.predict_proba(customer)[0, 1])

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Chance of booking", f"{chance:.0%}", f"{(chance - base_rate) * 100:+.0f} points vs average customer")
    m2.metric("Compared with the average", f"{chance / base_rate:.1f}×",
              f"average customer {base_rate:.0%}", delta_color="off", delta_arrow="off")
    m3.metric("Flagged by the model?", "Yes" if chance >= THRESHOLD else "No",
              f"flag threshold {THRESHOLD:.0%}", delta_color="off", delta_arrow="off")
    st.caption(f"Flight duration for {route} is taken from the data ({routes[route]:.1f} hours). "
               "Try changing the country first. It is by far the strongest factor.")

# ======================================================================== Campaign
with tab_campaign:
    st.markdown(
        "Should a campaign contact a customer? It depends on **what a contact costs** and **what reaching a "
        "booker is worth**. Because the scores are calibrated, the best rule is simply "
        "**contact anyone whose chance of booking is above cost ÷ value**."
    )
    c1, c2, c3 = st.columns(3)
    revenue = c1.number_input("Average booking value (£)", 50, 5000, 450, step=50)
    uplift = c2.slider("Extra chance a contacted booker books (%)", 1, 50, 15,
                       help="How much the campaign raises a likely booker's chance. Illustrative; "
                            "a real value would come from a controlled test.") / 100
    cost = c3.number_input("Cost per contact (£)", 0.0, 200.0, 2.50, step=0.50,
                           help="About £2.50 for an email, £20 or more for a voucher.")

    value = revenue * uplift
    best_rule = cost / value if value > 0 else 1.0
    thresholds = np.round(np.arange(0.0, 0.81, 0.01), 2)
    contacted = np.array([(scores >= t).sum() for t in thresholds])
    reached = np.array([((scores >= t) & (y_test == 1)).sum() for t in thresholds])
    net = reached * value - contacted * cost
    best_i = int(np.argmax(net))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Value of reaching a booker", f"£{value:,.2f}")
    k2.metric("Contact anyone above", f"{best_rule:.1%}", "chance of booking", delta_color="off", delta_arrow="off")
    k3.metric("Best net value (test set)", money(net[best_i]),
              f"contacting {contacted[best_i] / len(scores):.0%} of customers", delta_color="off", delta_arrow="off")
    k4.metric("If everyone is contacted", money(net[0]))

    fig = go.Figure()
    fig.add_scatter(x=thresholds, y=net, mode="lines", line_color=BLUE, name="net value")
    fig.add_vline(x=thresholds[best_i], line_dash="dash", line_color="#08306b",
                  annotation_text=f"best: {thresholds[best_i]:.2f}")
    fig.add_hline(y=0, line_color="grey", line_width=1)
    fig.update_layout(title=f"Campaign net value on {len(scores):,} unseen customers",
                      xaxis_title="Contact if chance of booking is at least", yaxis_title="Net value (£)",
                      yaxis_tickprefix="£")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Net value = bookers reached × value − customers contacted × cost. Not contacting someone costs nothing. "
        "Try £2.50 and then £20 per contact. Cheap emails are best sent to almost everyone, while expensive "
        "offers need the model to stay profitable."
    )
    st.warning(
        "The model predicts **who will book**, not **who will be persuaded** by a campaign. Someone who would "
        "have booked anyway gains nothing from a voucher, so the uplift figure must be measured with a "
        "controlled test before relying on these numbers."
    )

# ======================================================================== Quality
with tab_quality:
    order = np.argsort(-scores)
    share_contacted = np.arange(1, len(scores) + 1) / len(scores)
    share_reached = np.cumsum(y_test[order]) / y_test.sum()

    q1, q2, q3 = st.columns(3)
    q1.metric("ROC AUC (test)", f"{roc_auc_score(y_test, scores):.3f}", "0.5 = guessing", delta_color="off",
              delta_arrow="off")
    q2.metric("PR AUC (test)", f"{average_precision_score(y_test, scores):.3f}",
              f"baseline {y_test.mean():.3f}", delta_color="off", delta_arrow="off")
    top20 = share_reached[int(0.2 * len(scores)) - 1]
    q3.metric("Bookers in top 20% of scores", f"{top20:.0%}", f"{top20 / 0.2:.1f}× random",
              delta_color="off", delta_arrow="off")

    left, right = st.columns(2)
    with left:
        step = max(1, len(scores) // 400)
        gains = pd.DataFrame({"contacted": share_contacted[::step] * 100, "reached": share_reached[::step] * 100})
        fig = px.line(gains, x="contacted", y="reached", color_discrete_sequence=[BLUE],
                      labels={"contacted": "% of customers contacted (highest scores first)",
                              "reached": "% of all bookers reached"},
                      title="Cumulative gains")
        fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="grey"),
                        name="random targeting")
        st.plotly_chart(fig, width="stretch")
    with right:
        bins = pd.qcut(scores, 10, duplicates="drop")
        calib = pd.DataFrame({"predicted": scores, "actual": y_test}).groupby(bins, observed=True).mean() * 100
        fig = px.line(calib, x="predicted", y="actual", markers=True, color_discrete_sequence=[BLUE],
                      labels={"predicted": "Predicted chance (%)", "actual": "Actual booking rate (%)"},
                      title="Calibration: do the scores mean what they say?")
        fig.add_scatter(x=[0, 50], y=[0, 50], mode="lines", line=dict(dash="dash", color="grey"),
                        name="perfect")
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        "**What drives the predictions** (permutation importance, Notebook 03)\n"
        "1. **Booking country** dominates. Conversion ranges from 5% (Australia, the largest market) to 35% (Malaysia).\n"
        "2. **Origin airport** and **length of stay** (short trips book more often), then **destination airport**.\n"
        "3. **Extra baggage** is the most useful customer choice.\n"
        "4. **Days between booking and flight** matters little, contrary to the original version of this project.\n\n"
        "All results are on 9,857 customers the model never saw during training. Duplicates were removed before "
        "splitting, and the threshold was chosen on a separate validation set."
    )