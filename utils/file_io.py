# utils/file_io.py
import os
import hashlib
from urllib.parse import urlparse
import streamlit as st
from config import CONTENT_DIR

def save_content_to_file(url, content):
    try:
        hash_digest = hashlib.md5(url.encode('utf-8')).hexdigest()
        domain = urlparse(url).netloc
        filename = f"{domain}_{hash_digest}.txt"
        filepath = os.path.join(CONTENT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Content extracted from {url}:\n\n")
            f.write(content)
        st.success(f"Content saved to {filepath}")
        return filepath
    except Exception as e:
        st.error(f"Error saving content to file: {e}")
        return None
