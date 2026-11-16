"""
Scraper used in February 2026 to collect British Airways reviews from Skytrax.

This is the code that produced data/raw/skytrax_ba_reviews_2026-02.csv (4,018 reviews).
It is kept for transparency about how the data was collected.

Notes
- It captures review titles, traveller details and star ratings, but NOT the full review text.
- Missing fields were written as placeholder text ("No Aircraft", "No rating"). Notebook 02
  converts these to proper missing values.
- In November 2026 the Skytrax review pages returned HTTP 404 from South African connections,
  so the scrape could not be repeated (see Notebook 01).
- The only change from the original is the output file name, so a new run can never overwrite
  the archived February 2026 data.

Run from the task1_reviews_sentiment folder:
    python scraper_feb2026.py
"""
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.airlinequality.com/airline-reviews/british-airways"
OUTPUT_FILE = "data/raw/skytrax_reviews_new_scrape.csv"


def extract_detail(review, label):
    """Extracts review details based on label inside the table."""
    table = review.find("table")
    if table:
        label_td = table.find("td", string=label)
        return label_td.find_next_sibling("td").text.strip() if label_td else f"No {label}"
    return f"No {label}"


def extract_star_rating(review, label):
    """Extracts star ratings from the table."""
    table = review.find("table")
    if table:
        label_td = table.find("td", string=label)
        if label_td:
            stars_td = label_td.find_next_sibling("td", class_="review-rating-stars stars")
            if stars_td:
                stars = stars_td.find_all("span", class_="star fill")
                return len(stars)  # Number of filled stars
    return "No rating"


def scrape():
    reviews_data = []
    page_number = 1
    while True:
        url = BASE_URL if page_number == 1 else f"{BASE_URL}/page/{page_number}/"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page_number} (HTTP {response.status_code}). Stopping...")
            break

        print(f"Fetching page {page_number}...")
        soup = BeautifulSoup(response.text, "html.parser")
        reviews = soup.find_all("article", class_=["comp_media-review-rated", "list-item"])
        if not reviews:
            print("No more reviews found. Stopping...")
            break

        for review in reviews:
            title = review.find("h2", class_="text_header")
            reviews_data.append([
                title.text.strip() if title else "No Review Title",
                extract_detail(review, "Aircraft"),
                extract_detail(review, "Type Of Traveller"),
                extract_detail(review, "Seat Type"),
                extract_detail(review, "Route"),
                extract_detail(review, "Date Flown"),
                extract_star_rating(review, "Seat Comfort"),
                extract_star_rating(review, "Cabin Staff Service"),
                extract_star_rating(review, "Food & Beverages"),
                extract_star_rating(review, "Inflight Entertainment"),
                extract_star_rating(review, "Ground Service"),
                extract_star_rating(review, "Value For Money"),
            ])
        page_number += 1

    return pd.DataFrame(reviews_data, columns=[
        "Review Title", "Aircraft", "Type of Traveller", "Seat Type", "Route", "Date Flown",
        "Seat Comfort", "Cabin Staff Service", "Food Beverages", "Inflight Entertainment",
        "Ground Service", "Value For Money",
    ])


if __name__ == "__main__":
    df = scrape()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df):,} reviews to {OUTPUT_FILE}")