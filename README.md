# Query-Driven Web Scraper & Evaluator

A Streamlit-based pipeline that takes a natural-language query, performs deep semantic analysis, discovers relevant web pages via DuckDuckGo, scrapes and cleans their content (with NeuScraper + fallback), verifies relevance and content type with an LLM, semantically matches to user intent, then extracts exactly the requested information.

Additionally, a full evaluation harness (under `tests/`) computes precision/recall/F1/latency for each stage against ground-truth JSON fixtures.

---

## Features

- **Semantic Query Analysis** (`utils/semantic_analysis.py`)  
  • Breaks down user queries into search terms, expected output format, filters, and clarifications using an LLM.  
  • Provides both initial intent and deep multi-dimensional analysis.

- **Search** (`utils/search.py`)  
  • DuckDuckGo integration to retrieve top N URLs for a query.  
  • Supports optional domain-restricted searches.

- **Scraping** (`utils/scraping.py`)  
  • Primary extraction via NeuScraper API.  
  • Fallback HTML fetch + `trafilatura` cleanup.  
  • Caching and file-output of scraped text.

- **Relevance Checking** (`utils/relevance_check.py`)  
  • LLM-based semantic relevance scoring.  
  • Filters out irrelevant pages early.

- **Content-Type Verification & Semantic Match** (`utils/content_match.py`)  
  • Determines if content matches expected type (review, tutorial, news, comparison, etc.).  
  • Verifies semantic match to the user’s intent (e.g. actual product comparisons, real reviews).

- **Parsing & Field Extraction** (`utils/parsing.py`)  
  • Rule-based + LLM-enhanced parsing to extract reviewer names and review text.  
  • Generalized functions to explain parsing and generate structured responses.

- **Streamlit UI** (`app.py`)  
  • Interactive front end to enter queries and optional website constraint.  
  • Displays each processing stage, caching, conversation history, and scraped content.

- **Evaluation Harness** (`tests/`)  
  • `tests/relevance_classification.py`  
  • `tests/content_type_matching.py`  
  • `tests/semantic_matching.py`  
  • `tests/field_extraction.py`  
  • `tests/evaluator.py`  
  • Ground-truth fixtures under `tests/data/` for end-to-end metric reporting.

---

## 📁 Repository Structure

![image](https://github.com/user-attachments/assets/b75b1b2e-77fd-42a7-92de-960de37c62c1)


---

---

## ⚙️ Prerequisites

- **Python 3.8+**  
- **Git**  
- **Streamlit CLI**  
- **NeuScraper** running at `NEUSCRAPER_URL` (default `http://127.0.0.1:1688/predict/`)

---

## 🛠 Installation

```bash
# Clone the repo
git clone git@github.com:uol-feps-soc-comp3931-project-2425/individual-project-SiddeshROhri23.git
cd individual-project-SiddeshROhri23

# Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
echo "NEUSCRAPER_URL=http://127.0.0.1:1688/predict/" > .env
```
## Running the App
```bash
streamlit run app.py
```
## From project root:
```bash
python tests/relevance_classification.py
python tests/content_type_matching.py
python tests/semantic_matching.py
python tests/field_extraction.py
```

## Or run everything via:
```bash
python tests/evaluator.py
```

Built with ❤️ by Siddesh R Ohri.
