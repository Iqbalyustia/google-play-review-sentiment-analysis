import pandas as pd
from google_play_scraper import reviews, Sort


def scrape_reviews(app_id, count=1000, country="id", lang="id"):
    reviews_data, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        count=count,
        sort=Sort.MOST_RELEVANT,
        filter_score_with=None
    )

    return pd.DataFrame(reviews_data)