# British Airways Customer Analytics

Two end-to-end data science projects based on the **Forage British Airways Data Science job simulation**, rebuilt with leak-free modelling, honest evaluation and an interactive Streamlit app.

**Live app:** [ba-customer-analytics.streamlit.app](https://ba-customer-analytics.streamlit.app) (if it has been idle, it may take a few seconds to wake up)

| | Task 1 · Reviews & lounges | Task 2 · Booking prediction |
|---|---|---|
| **Question** | What drives passenger satisfaction, and how many passengers will be eligible for each Heathrow lounge tier? | Which customers are likely to complete a booking, and when is targeting them worth it? |
| **Data** | 4,015 Skytrax reviews (2014–2026) and a 10,000-flight summer schedule | 49,281 booking records (after removing 719 duplicates) |
| **Methods** | Cleaning without imputation, priority matrix, Mann-Whitney test, TF-IDF + VADER hybrid sentiment model, lookup table with tested assumptions | sklearn pipelines, 5-fold CV model comparison, validation-set threshold, calibration, permutation importance, campaign value analysis |
| **Headline result** | Sentiment macro F1 0.84 on held-out reviews vs 0.79 for VADER alone (0.82 cross-validated) | ROC AUC 0.78 on unseen customers. The top 20% of scores contains 48% of bookers (2.4× random) |

---

## Key findings

**Task 1 · Passenger reviews**
- Satisfaction stayed below the 3-star midpoint every year from 2015 to 2025, falling from 2.95 (2015) to 2.34 (2023).
- Cabin crew are BA's only area rated above 3. Food & beverages and ground service are the priorities to fix, because they are rated low **and** closely linked to perceived value.
- Business Class seats are rated below Premium Economy seats. A small improvement since 2019 (Mann-Whitney p ≈ 0.04) weakly supports the new Club Suite having helped.
- Business travellers are the least satisfied group overall, and the least satisfied in three of the four cabins.

**Task 1 · Lounge eligibility**
- In this dataset, eligible passengers per flight (≈0.6 / 8 / 30 for Tiers 1 / 2 / 3) do not depend on aircraft size. A percentage-of-seats lookup would overestimate A380 flights by ~22 Tier 3 passengers each, so the lookup uses headcounts per departure, tested on later flights.
- A "plan for" figure adds a statistically sized safety margin. Built from April–August data, it covered actual demand on 92–93% of the 60 later days (September–October).

**Task 2 · Booking prediction**
- **Booking country** is by far the strongest driver (5% conversion from Australia vs 35% from Malaysia).
- Days between booking and flight matters little, contrary to the original version of this project.
- Scores are calibrated (average prediction 15.0% vs actual 15.0%), so the best campaign rule is simply *contact a customer if their chance of booking exceeds cost ÷ value*.
- For a £2.50 email the model adds about 5% over contacting everyone. For a £20 voucher it turns a £97,000 loss into a profit.

---

## What changed from the original version

The first version of this project had problems that this rebuild fixes.

| Original | Rebuilt |
|---|---|
| Sentiment model trained to predict VADER's labels from VADER's own score (circular, "100% accuracy") | Labels from passengers' star ratings, with VADER as a baseline to beat |
| Missing star ratings filled with invented decimal values | Missing values left missing. Averages use only real ratings |
| Encoding fitted on the full dataset; threshold tuned on the test set; duplicates in train and test | Pipelines fitted on training data only, separate validation set, duplicates removed first |
| Lounge table gave every Premium Economy passenger lounge access and used reviewers as a proxy for passengers | Built from the actual schedule data, with assumptions tested |
| Several README claims contradicted the data (e.g. which travellers were least satisfied) | Claims checked against the notebook outputs in a full verification audit |

A stress test of the deployed sentiment model also revealed a **spurious correlation** ("crew" appeared mostly in positive titles, so *"The crew was crap"* scored 87% positive). Adding VADER's score as a feature fixed it. The investigation is documented in Task 1 Notebook 03.

---

## Project structure

```
british-airways-analytics/
├── task1_reviews_sentiment/
│   ├── 01_data_collection.ipynb         data source and first look at quality
│   ├── 02_cleaning_and_eda.ipynb        cleaning and exploratory analysis
│   ├── 03_sentiment_model.ipynb         sentiment model, error analysis, experiments
│   ├── 04_lounge_eligibility.ipynb      lounge lookup table
│   ├── scraper_feb2026.py               the scraper that collected the reviews
│   ├── data/                            raw and processed data
│   └── models/                          saved sentiment model
├── task2_booking_prediction/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_and_modelling.ipynb
│   ├── 03_evaluation_and_business_value.ipynb
│   ├── data/
│   └── models/                          saved booking model
├── app/                                 Streamlit app (Home + 4 pages)
├── requirements.txt                     packages for the app
└── requirements-dev.txt                 extra packages for the notebooks and scraper
```

## Run it locally

Requires **Python 3.14** (built and tested on 3.14.8).

```bash
python -m venv .venv
.venv\Scripts\activate          # on macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app/Home.py
```

Notebooks are numbered and meant to be read in order within each task. Open them from inside their task folder, because they use paths relative to it. Notebooks that produce data or models save them for the next notebook and the app.

## Data sources and limitations

- **Reviews** were scraped from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways/) in February 2026 using `scraper_feb2026.py`. Only review titles were captured, and a later re-scrape with full text was blocked by regional restrictions. Reviews are self-selected, so comparisons are more reliable than absolute levels.
- **The schedule and booking datasets** were provided by Forage for the job simulation. They are training datasets, not official British Airways data, and the booking data appears to cover an Asia-Pacific network. All data files are included for reproducibility only.
- Campaign values use illustrative assumptions. The model predicts *who will book*, not *who will be persuaded*, so real uplift would need a controlled test.

---

**Gopolang Mmutlwane** · [Portfolio](https://gopolang-mmutlwane.onrender.com) · [LinkedIn](https://www.linkedin.com/in/gopolang-mmutlwane-016250437) · [GitHub](https://github.com/gopolangmmutlwane)