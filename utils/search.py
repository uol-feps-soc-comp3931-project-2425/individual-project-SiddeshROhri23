# utils/search.py
import streamlit as st
from duckduckgo_search import DDGS

def search_query_links(query):
    st.info(f"Searching for relevant links using the query: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        links = []
        for result in results:
            result_url = result.get("href") or result.get("url")
            if result_url and result_url not in links:
                links.append(result_url)
        st.write("Found links:", links)
        return links
    except Exception as e:
        st.error(f"Error searching for links: {e}")
        return []
