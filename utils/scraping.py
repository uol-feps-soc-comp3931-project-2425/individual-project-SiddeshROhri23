# utils/scraping.py
import time
import random
import requests
import streamlit as st
import trafilatura
from config import NEUSCRAPER_URL
from utils.file_io import save_content_to_file

def neuscraper_extract(url):
    st.info(f"Extracting content from {url} using NeuScraper...")
    payload = {"url": url}
    try:
        response = requests.post(NEUSCRAPER_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            st.write("Raw JSON response:", data)
            pages = data.get("Pages", [])
            if pages:
                content = "\n\n".join([page.get("text", "") for page in pages if page.get("text")])
            else:
                content = data.get("Text", "")
            if content and len(content.strip()) > 0:
                st.success("Content extracted successfully.")
                return content
            else:
                st.warning("NeuScraper returned no text. Falling back to basic extraction.")
                return fetch_and_clean(url)
        else:
            st.error(f"NeuScraper API error: {response.status_code} - {response.text}")
            return fetch_and_clean(url)
    except Exception as e:
        st.error(f"Error calling NeuScraper API: {e}")
        return fetch_and_clean(url)

def fetch_and_clean(url):
    st.info(f"Fetching content from: {url}")
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/14.1.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    try:
        time.sleep(random.uniform(1, 3))
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            st.success("Content fetched successfully!")
            html_content = response.text
            clean_text = trafilatura.extract(html_content)
            final = clean_text if clean_text else html_content
            save_content_to_file(url, final)
            return final
        else:
            st.error(f"Error: Received status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Exception while fetching content: {e}")
        return None

_scraped_cache = {}

def scrape_link_content(url):
    if url in _scraped_cache:
        st.info(f"Using cached content for {url}")
        return _scraped_cache[url]
    else:
        st.info(f"Scraping content for {url}")
        content = neuscraper_extract(url)
        if content:
            _scraped_cache[url] = content
        return content
