# app.py
import streamlit as st
import os
from config import CONTENT_DIR
from utils.parsing import process_query_search

# Create directory for storing scraped content files if it doesn't exist
if not os.path.exists(CONTENT_DIR):
    os.makedirs(CONTENT_DIR)

# Initialize session state variables for caching scraped content and conversation history
if "scraped_link_contents" not in st.session_state:
    st.session_state["scraped_link_contents"] = {}  # Cache: {url: content}
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("Query Search with Optional Website Constraint")
st.markdown(
    """
    This application processes your natural-language query with deep semantic analysis.
    
    If you provide a website link, the system will first consider content only from that website.
    If that content does not sufficiently address your query, the system will then perform an internet search
    (restricted to that website) and combine results from the additional sources.
    """
)

# --- Query Section ---
user_query = st.text_input(
    "Enter your query:", 
    "Could you give me the reviews of Blue Lock in a well formatted table which should include the name of the reviewer, the review by the reviewer"
)
website_link = st.text_input(
    "Optional: Provide a website link (if you want to restrict the search):", 
    "https://myanimelist.net/"
)

if st.button("Search Query"):
    if user_query:
        st.info(f"Processing your query: {user_query}")
        with st.spinner("Searching for relevant information..."):
            final_response = process_query_search(
                user_query, 
                website_link if website_link.strip() != "" else None
            )
        st.success("Query processing complete.")
        st.session_state["chat_history"].append({"user": user_query, "assistant": final_response})
    else:
        st.warning("Please enter a query.")

# --- Conversation History ---
if st.session_state["chat_history"]:
    st.markdown("### Conversation History")
    for chat in st.session_state["chat_history"]:
        st.write("**User:** " + chat["user"])
        st.write("**Assistant:** " + chat["assistant"])

# --- View Scraped Content Files ---
with st.expander("View Scraped Content Files"):
    if os.path.exists(CONTENT_DIR) and os.listdir(CONTENT_DIR):
        files = [f for f in os.listdir(CONTENT_DIR) if f.endswith('.txt')]
        selected_file = st.selectbox("Select a file to view", files)
        if selected_file:
            try:
                with open(os.path.join(CONTENT_DIR, selected_file), 'r', encoding='utf-8') as f:
                    content = f.read()
                st.text(content)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        st.info("No scraped content files found yet.")
