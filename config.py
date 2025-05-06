import os
from dotenv import load_dotenv

load_dotenv()

# Environment/config constants
CSV_FILENAME = "scraped_data.csv"
CONTENT_DIR = "scraped_content"
NEUSCRAPER_URL = os.getenv("NEUSCRAPER_URL", "http://127.0.0.1:1688/predict/")
