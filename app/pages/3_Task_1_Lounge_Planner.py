import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import TASK1, load_csv

st.set_page_config(page_title="Lounge Planner", page_icon="✈️", layout="wide")
st.title("Task 1 · Lounge Planner, Heathrow Terminal 3")
st.caption(
    "Estimates how many departing passengers will be eligible for each lounge tier, using the lookup table "
    "built from 10,000 flights in the 2025 summer schedule (Forage dataset)."
)

TIERS = {"tier1": "Tier 1", "tier2": "Tier 2", "tier3": "Tier 3"}
SLOTS = ["Morning", "Lunchtime", "Afternoon", "Evening"]
REGIONS = ["Europe", "North America", "Asia", "Middle East"]
Z_90 = 1.2816  # 90th percentile of the normal distribution


@st.cache_data
def load_schedule() -> pd.DataFrame:
    df = pd.read_excel(TASK1 / "data" / "raw" / "ba_summer_schedule_2025.xlsx")
    df.columns = df.columns.str.lower()
    return df.rename(columns={f"{t}_eligible_pax": t for t in TIERS})


schedule = load_schedule()
lookup = load_csv(TASK1 / "data" / "processed" / "lounge_lookup_table.csv").set_index("arrival_region")
# Spread of eligible passengers per flight, used for the "plan for" figure
spread = schedule.groupby("arrival_region")[list(TIERS)].std()


def estimate(departures: pd.DataFrame) -> pd.DataFrame:
    """departures: rows = time slots, columns = regions, values = number of flights.
    Returns expected eligible passengers per slot and tier, plus a 90% planning figure."""
    rows = []
    for slot in departures.index:
        n = departures.loc[slot]
        for tier, name in TIERS.items():
            mean = sum(n[r] * lookup.loc[r, f"{tier}_avg"] for r in departures.columns)
            sd = np.sqrt(sum(n[r] * spread.loc[r, tier] ** 2 for r in departures.columns))
            rows.append({"time slot": slot, "tier": name, "expected": mean, "plan for (90%)": mean + Z_90 * sd})
    return pd.DataFrame(rows)


mode = st.radio("Choose a schedule", ["Build your own day", "Replay a real day from 2025"], horizontal=True)

# ---------------------------------------------------------------- Build your own
if mode == "Build your own day":
    typical = (
        schedule.groupby(["time_of_day", "arrival_region"]).size().unstack(fill_value=0)
        / schedule["flight_date"].nunique()
    ).reindex(index=SLOTS, columns=REGIONS).round().astype(int)

    st.markdown("**Departures in each time slot.** Edit the numbers to plan a different day. "
                "The starting values are a typical day in the 2025 summer schedule.")
    departures = st.data_editor(
        typical.rename_axis("time slot"),
        column_config={r: st.column_config.NumberColumn(r, min_value=0, max_value=60, step=1) for r in REGIONS},
        width="stretch",
    ).fillna(0)

    total_flights = int(departures.to_numpy().sum())
    if total_flights == 0:
        st.info("Add at least one departure to see an estimate.")
        st.stop()

    result = estimate(departures)
    # Whole-day figure: estimate the day as one block, so busy and quiet slots can also balance out
    day = estimate(departures.sum().to_frame("whole day").T).set_index("tier")

    c = st.columns(4)
    c[0].metric("Departures", f"{total_flights}")
    for col, name in zip(c[1:], TIERS.values()):
        col.metric(f"{name} eligible, whole day", f"{day.loc[name, 'expected']:.0f}",
                   f"plan for {day.loc[name, 'plan for (90%)']:.0f}", delta_color="off", delta_arrow="off")

    fig = px.bar(result, x="time slot", y="expected", color="tier", barmode="group",
                 error_y=result["plan for (90%)"] - result["expected"], error_y_minus=[0] * len(result),
                 category_orders={"time slot": SLOTS},
                 color_discrete_sequence=["#08306b", "#2171b5", "#9ecae1"],
                 labels={"expected": "Eligible passengers", "time slot": ""},
                 title="Expected lounge-eligible passengers by departure time")
    st.plotly_chart(fig, width="stretch")
    st.caption("Bars show the expected number. The thin line above each bar reaches the **plan-for** figure, "
               "a level that total demand should stay below on about 9 days out of 10.")

    table = result.pivot(index="time slot", columns="tier", values=["expected", "plan for (90%)"]).reindex(SLOTS)
    st.dataframe(table.round(0).astype(int), width="stretch")

# ---------------------------------------------------------------- Replay a real day
else:
    dates = schedule["flight_date"].dt.date
    chosen = st.date_input("Day", value=dates.min(), min_value=dates.min(), max_value=dates.max())
    day_flights = schedule[dates == chosen]
    if day_flights.empty:
        st.info("No flights in the schedule on this day. Try another date.")
        st.stop()

    departures = (
        day_flights.groupby(["time_of_day", "arrival_region"]).size().unstack(fill_value=0)
        .reindex(index=SLOTS, columns=REGIONS, fill_value=0)
    )
    result = estimate(departures)
    actual = (
        day_flights.groupby("time_of_day")[list(TIERS)].sum().reindex(SLOTS, fill_value=0)
        .rename(columns=TIERS).stack().rename("actual").reset_index()
        .rename(columns={"time_of_day": "time slot", "level_1": "tier"})
    )
    actual.columns = ["time slot", "tier", "actual"]
    compare = result.merge(actual, on=["time slot", "tier"])

    tier = st.radio("Tier", list(TIERS.values()), index=2, horizontal=True)
    sub = compare[compare["tier"] == tier].melt(id_vars="time slot", value_vars=["expected", "actual"],
                                                var_name="figure", value_name="passengers")
    fig = px.bar(sub, x="time slot", y="passengers", color="figure", barmode="group",
                 category_orders={"time slot": SLOTS, "figure": ["expected", "actual"]},
                 color_discrete_map={"expected": "#9ecae1", "actual": "#08306b"},
                 labels={"time slot": "", "figure": ""},
                 title=f"{tier}: lookup-table estimate vs what the schedule recorded on {chosen:%d %B %Y}")
    st.plotly_chart(fig, width="stretch")

    day_total = compare.groupby("tier")[["expected", "actual"]].sum()
    day_total["plan for (90%)"] = estimate(departures.sum().to_frame("whole day").T).set_index("tier")["plan for (90%)"]
    day_total = day_total.round(0).astype(int)
    day_total["within plan-for?"] = np.where(day_total["actual"] <= day_total["plan for (90%)"], "yes", "no")
    st.markdown(f"**{len(day_flights)} departures that day.** Whole-day totals")
    st.dataframe(day_total, width="stretch")
    st.caption("This day was part of the data used to build the lookup table, so this is a sense check of how "
               "much single days vary, not an independent test. The independent test is in Notebook 04.")

with st.expander("How the estimate works and its limits"):
    st.markdown(
        "- Each departure is expected to bring the **average** number of eligible passengers for its region "
        "(about 0.6 Tier 1, 8 Tier 2 and 30 Tier 3 per flight). Averages are added up for each time slot.\n"
        "- The **plan-for** figure adds a safety margin based on how much individual flights vary. Because busy and "
        "quiet flights partly cancel out, the margin grows more slowly than the number of flights.\n"
        "- In this dataset, eligible passengers per flight do **not** depend on aircraft size, so the estimate uses "
        "headcounts rather than a percentage of seats. Notebook 04 showed a percentage-based table would "
        "overestimate A380 flights by about 22 Tier 3 passengers each.\n"
        "- **Eligible is not the same as attending.** Real lounge occupancy also depends on how many eligible "
        "passengers visit and how long they stay.\n"
        "- The schedule is a training dataset, so the table should be rebuilt from real data before use."
    )
    st.dataframe(lookup.reset_index(), hide_index=True, width="stretch")