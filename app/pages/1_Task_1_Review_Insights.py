import pandas as pd
import plotly.express as px
import streamlit as st

from utils import BLUE, TASK1, load_csv

st.set_page_config(page_title="Review Insights", page_icon=":material/flight:", layout="wide")

RATINGS = {
    "seat_comfort": "Seat Comfort",
    "cabin_staff_service": "Cabin Staff Service",
    "food_beverages": "Food & Beverages",
    "inflight_entertainment": "Inflight Entertainment",
    "ground_service": "Ground Service",
    "value_for_money": "Value For Money",
}
CABINS = ["Economy Class", "Premium Economy", "Business Class", "First Class"]
TRAVELLERS = ["Business", "Couple Leisure", "Family Leisure", "Solo Leisure"]


def aircraft_family(name) -> str:
    """Group the many free-text aircraft names into a few families."""
    if pd.isna(name):
        return "Not recorded"
    s = str(name).upper()
    rules = [
        (("A380",), "Airbus A380"), (("A350",), "Airbus A350"),
        (("787", "788", "789", "B78", "DREAMLINER"), "Boeing 787"),
        (("777",), "Boeing 777"), (("747", "744"), "Boeing 747"), (("767",), "Boeing 767"),
        (("A318", "A319", "A32", "A321"), "Airbus A320 family"),
        (("E-1", "E1", "EMB"), "Embraer"),
    ]
    for keys, family in rules:
        if any(k in s for k in keys):
            return family
    return "Other"


@st.cache_data
def load_reviews() -> pd.DataFrame:
    df = load_csv(TASK1 / "data" / "processed" / "reviews_clean.csv").copy()
    df["aircraft_family"] = df["aircraft"].map(aircraft_family)
    return df


df = load_reviews()
FAMILIES = sorted(f for f in df["aircraft_family"].unique() if f != "Not recorded")

# ---------------- Filters ----------------
DEFAULTS = {"f_cabins": [], "f_travellers": [], "f_years": (2014, 2026), "f_undated": True,
            "f_aircraft": [], "f_min_ratings": 1}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_filters():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


with st.sidebar:
    st.header("Filters")
    st.caption("Click to select. Leaving a group empty means *all*.")

    st.pills("Cabin", CABINS, selection_mode="multi", key="f_cabins")
    st.pills("Type of traveller", TRAVELLERS, selection_mode="multi", key="f_travellers")

    st.slider("Year flown", 2014, 2026, key="f_years")
    narrowing = bool(st.session_state.f_travellers) or tuple(st.session_state.f_years) != DEFAULTS["f_years"]
    st.checkbox("Include older reviews with no date or traveller type", key="f_undated", disabled=narrowing,
                help="About 780 older reviews have no date or traveller type. They are left out automatically "
                     "when you filter by traveller type or year, because we cannot tell whether they match.")

    st.multiselect("Aircraft", FAMILIES, key="f_aircraft", placeholder="All aircraft",
                   help="Aircraft is missing for about half of the reviews. Choosing any aircraft "
                        "shows only reviews where it was recorded.")
    st.slider("Minimum star ratings given (of 6)", 1, 6, key="f_min_ratings",
              help="Higher values keep only reviews with more complete ratings, so averages are more reliable.")

    st.button("Reset filters", on_click=reset_filters, width="stretch")

cabins = st.session_state.f_cabins or CABINS
travellers = st.session_state.f_travellers or TRAVELLERS
year_from, year_to = st.session_state.f_years

# Two reviews have no cabin recorded. They are included unless a cabin filter is chosen.
no_cabin_filter = not st.session_state.f_cabins
cabin_ok = df["seat_type"].isin(cabins) | (df["seat_type"].isna() & no_cabin_filter)
view = df[cabin_ok & (df["n_ratings"] >= st.session_state.f_min_ratings)]

# Older reviews lack traveller type and/or date. They are kept or dropped as a group via the checkbox.
has_details = view["type_of_traveller"].notna() & view["year_flown"].notna()
matches = view["type_of_traveller"].isin(travellers) & view["year_flown"].between(year_from, year_to)
include_older = st.session_state.f_undated and not narrowing
view = view[(has_details & matches) | (~has_details & include_older)]

if st.session_state.f_aircraft:
    view = view[view["aircraft_family"].isin(st.session_state.f_aircraft)]

# ---------------- Page ----------------
st.title("Task 1 · Review Insights")
st.caption("Skytrax reviews of British Airways (flights 2014–2026). Ratings run from 1 to 5 stars.")

active = []
if st.session_state.f_cabins:
    active.append(", ".join(st.session_state.f_cabins))
if st.session_state.f_travellers:
    active.append(", ".join(st.session_state.f_travellers))
if (year_from, year_to) != DEFAULTS["f_years"]:
    active.append(f"flown {year_from}–{year_to}")
if st.session_state.f_aircraft:
    active.append(", ".join(st.session_state.f_aircraft))
if st.session_state.f_min_ratings > 1:
    active.append(f"at least {st.session_state.f_min_ratings} ratings")
st.markdown("**Showing** " + ("all reviews" if not active else " · ".join(active)))

if len(view) < 20:
    st.warning(f"Only {len(view)} reviews match these filters, which is too few for reliable averages. "
               "Try widening the filters.")
    st.stop()

# Headline numbers
area_means = view[list(RATINGS)].mean()
best, worst = area_means.idxmax(), area_means.idxmin()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(view):,}")
c2.metric("Average rating", f"{view['avg_rating'].mean():.2f} / 5",
          f"{view['avg_rating'].mean() - df['avg_rating'].mean():+.2f} vs all reviews")
c3.metric("Strongest area", f"{area_means[best]:.2f} / 5")
c3.caption(RATINGS[best])
c4.metric("Weakest area", f"{area_means[worst]:.2f} / 5")
c4.caption(RATINGS[worst])

tab_trend, tab_service, tab_cabin, tab_reviews = st.tabs(
    ["Over time", "Service areas", "Cabins", "Read the reviews"]
)

with tab_trend:
    yearly = (
        view[view["year_flown"].between(max(year_from, 2015), min(year_to, 2025))]
        .groupby("year_flown")
        .agg(reviews=("avg_rating", "size"), avg_rating=("avg_rating", "mean"))
        .reset_index()
    )
    if yearly.empty:
        st.info("No dated reviews in this selection.")
    else:
        left, right = st.columns(2)
        with left:
            fig = px.bar(yearly, x="year_flown", y="reviews", title="Reviews per year",
                         labels={"year_flown": "Year flown", "reviews": "Reviews"},
                         color_discrete_sequence=[BLUE])
            st.plotly_chart(fig, width="stretch")
        with right:
            fig = px.line(yearly, x="year_flown", y="avg_rating", markers=True, range_y=[1, 5],
                          title="Average rating per year",
                          labels={"year_flown": "Year flown", "avg_rating": "Average rating"},
                          color_discrete_sequence=[BLUE])
            fig.add_hline(y=3, line_dash="dash", line_color="grey", annotation_text="midpoint")
            st.plotly_chart(fig, width="stretch")
        st.caption("Only 2015–2025 is charted, because 2014 and 2026 contain too few reviews.")

with tab_service:
    service = area_means.rename(index=RATINGS).sort_values().reset_index()
    service.columns = ["area", "average"]
    service["ratings"] = service["area"].map(view[list(RATINGS)].notna().sum().rename(index=RATINGS))
    fig = px.bar(service, x="average", y="area", orientation="h", range_x=[1, 5], text_auto=".2f",
                 hover_data={"ratings": True}, labels={"average": "Average rating", "area": ""},
                 color_discrete_sequence=[BLUE])
    fig.add_vline(x=3, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, width="stretch")
    st.caption("Hover over a bar to see how many passengers rated that area.")

with tab_cabin:
    shown = [c for c in CABINS if c in cabins]
    by_cabin = view.groupby("seat_type")[list(RATINGS)].mean().reindex(shown).dropna(how="all")
    by_cabin.columns = [RATINGS[c] for c in by_cabin.columns]
    fig = px.imshow(by_cabin.round(2), text_auto=True, color_continuous_scale="RdBu", zmin=1, zmax=5,
                    color_continuous_midpoint=3, aspect="auto", labels={"color": "Average rating"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Red is below the 3-star midpoint and blue is above it.")

with tab_reviews:
    sort_by = st.radio("Show", ["Lowest rated first", "Highest rated first", "Most recent first"], horizontal=True)
    readable = view[view["has_informative_title"]]
    if sort_by == "Lowest rated first":
        readable = readable.sort_values("avg_rating")
    elif sort_by == "Highest rated first":
        readable = readable.sort_values("avg_rating", ascending=False)
    else:
        readable = readable.sort_values("date_flown", ascending=False, na_position="last")
    st.dataframe(
        readable[["review_title", "avg_rating", "seat_type", "type_of_traveller", "route", "date_flown"]].head(200),
        column_config={
            "review_title": "Review title", "avg_rating": st.column_config.NumberColumn("Avg rating", format="%.1f / 5"),
            "seat_type": "Cabin", "type_of_traveller": "Traveller", "route": "Route", "date_flown": "Flown",
        },
        hide_index=True, width="stretch",
    )
    st.caption("Up to 200 reviews shown. Reviews with only a generic site title are left out.")

with st.expander("Key findings from the analysis notebook"):
    st.markdown(
        "- Satisfaction stayed **below the midpoint every year from 2015 to 2025**, falling from 2.95 (2015) to 2.34 (2023).\n"
        "- **Cabin crew** are the only area rated above 3. **Value for money** and **inflight entertainment** are weakest.\n"
        "- **Business Class seats are rated below Premium Economy seats.** A small improvement since 2019 "
        "(p ≈ 0.04) weakly supports the new Club Suite having helped.\n"
        "- **Business travellers are the least satisfied group**, and the least satisfied in three of the four cabins.\n\n"
        "Skytrax reviews are self-selected and probably skew negative, so comparisons matter more than absolute levels."
    )