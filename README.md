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

## ⚙️ Prerequisites

- **Python 3.8+**  
- **Git**  
- **Streamlit CLI**  
- **NeuScraper** service  
  ```bash
  git clone https://github.com/OpenMatch/NeuScraper.git
  cd NeuScraper
  # follow NeuScraper’s README to install dependencies and start the service
---

Once the Neuscraper is installed, change these files in the app/ folder
## app.py
```python
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import re

app = FastAPI()

class InputData(BaseModel):
    url: HttpUrl
    search_terms: Optional[List[str]] = None

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def extract_relevant_content(full_text: str, search_terms: List[str]) -> str:
    paras = [p.strip() for p in full_text.split('\n\n') if p.strip()]
    matched = [
        p for p in paras
        if any(term.lower() in p.lower() for term in search_terms)
    ]
    return '\n\n'.join(matched) if matched else full_text

@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        response = requests.get(
            input_data.url,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10
        )
        if response.status_code != 200:
            raise HTTPException(400, f"Error fetching URL: {response.status_code}")
        raw_html = response.content

        soup = BeautifulSoup(raw_html, 'html.parser')
        for tag in ['script','style','noscript','iframe','svg']:
            for el in soup.find_all(tag):
                el.decompose()

        redundant = ['ad-','ads-','cookie-','popup-','newsletter-','subscribe-','login-','signup-']
        for pat in redundant:
            for el in soup.find_all(attrs={"class": re.compile(pat, re.I)}):
                el.decompose()
            for el in soup.find_all(attrs={"id": re.compile(pat, re.I)}):
                el.decompose()

        text = soup.body.get_text(separator='\n') if soup.body else soup.get_text(separator='\n')
        text = clean_text(text)

        lines = text.split('\n')
        skip = [
            r'^Sign (in|up|out)$', r'^Log (in|out)$', r'^Search$',
            r'^Menu$', r'^Home$', r'^Close$', r'^Accept (all|cookies)$',
            r'^Submit$', r'^Share$'
        ]
        filtered = [ln for ln in lines if not any(re.match(p, ln.strip(), re.I) for p in skip)]
        extracted_text = '\n'.join(filtered)

        if input_data.search_terms:
            extracted_text = extract_relevant_content(extracted_text, input_data.search_terms)

        save_dir = "extracted_texts"
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        path = os.path.join(save_dir, f"extracted_{ts}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

        return {"Text": extracted_text, "saved_file": path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing content: {e}")
```
## builder.py
```python
from tokenization import TokenizerProcessor
from api import CommonCrawlApi
import warnings
import json
from bs4 import BeautifulSoup
import chardet
import pandas as pd

CSV_COLUMN_NAMES = ['Url', 'TextNodeId', 'Text']
JSON_COLUMN_NAMES = ['TokenId', 'NodeIds', 'Url']

class FeatureExtractorApplierProcessor:
    def __init__(self):
        self.comment = 'This is the constant comment for all rows returned'
        self.chunk_size = 384
        self.max_token_length = 50

    def _chunk_nodes(self, node_texts, node_seq, node_url):
        chunks = []
        for i in range(0, len(node_texts), self.chunk_size):
            chunk = (node_texts[i:i+self.chunk_size],
                     node_seq[i:i+self.chunk_size],
                     node_url[i:i+self.chunk_size])
            chunks.append(chunk)
        return chunks

    def add_node_id(self, html_str):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
            soup = BeautifulSoup(html_str, 'html.parser')
        all_soup_nodes = soup.find_all()
        stack = [all_soup_nodes[0]]
        node_index = 0
        while stack:
            node = stack.pop()
            if "data-dcnode-id" in node.attrs:
                continue
            node.attrs["data-dcnode-id"] = node_index
            node_index += 1
            for child in node.children:
                if node.name == "span" and isinstance(child, str):
                    continue
                if isinstance(child, str):
                    new_node = soup.new_tag("span")
                    new_node.string = child
                    new_node.attrs["instrument_node"] = None
                    child.replace_with(new_node)
                    stack.append(new_node)
                else:
                    stack.append(child)
        return soup

    def Apply(self, url, api):
        tokenizer = TokenizerProcessor(self.max_token_length)
        node_sequence, node_texts_tokens, node_url = [], [], []
        for node_id, node in api.all_nodes.items():
            if node.is_textnode or node.html_node.name in ["ol", "dl", "table"]:
                text = node.html_node.text.strip('\r\n\t\xa0 ')
                node_sequence.append(node_id)
                node_texts_tokens.append(tokenizer.tokenize_sequence(text))
                node_url.append(url)
        for chunk in self._chunk_nodes(node_texts_tokens, node_sequence, node_url):
            yield json.dumps({'TokenId': chunk[0], 'NodeIds': chunk[1], 'Url': chunk[2]}, separators=(',', ':'))

def build(url, raw_html):
    generator = FeatureExtractorApplierProcessor()
    text_nodes_data = []
    json_data = []
    try:
        html_content = raw_html.decode('utf-8')
    except UnicodeDecodeError:
        guess = chardet.detect(raw_html)['encoding']
        if not guess or guess == 'UTF-8': return
        try:
            html_content = raw_html.decode(guess)
        except:
            return
    html_soup = generator.add_node_id(html_content)
    api = CommonCrawlApi(html_soup=html_soup)
    json_data.extend(generator.Apply(url, api))
    for node in api.all_nodes.values():
        if node.is_textnode or node.html_node.name in ["ol", "dl", "table"]:
            text = node.html_node.text.strip('\r\n\t\xa0 ')
            text_nodes_data.append([url, node.nodeid, text])
    text_nodes_df = pd.DataFrame(text_nodes_data, columns=CSV_COLUMN_NAMES)
    return text_nodes_df, json_data, html_soup
```
## model.py
```python
import math
import torch
import torch.nn as nn
from pipeline_evaluator import *
from transformers import BertConfig, XLMRobertaConfig, XLMRobertaModel
from transformers.models.bert.modeling_bert import BertEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        current_dim = input_dim
        layers = []
        for hdim in hidden_dim:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            current_dim = hdim
        layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)

class ContentExtractionTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_version = config.model_version
        self.sigmoid = nn.Sigmoid()
        self.max_sequence_len = config.max_sequence_len
        self.text_in_emb_dim = config.text_in_emb_dim
        self.text_emb_dim = config.text_emb_dim
        self.hidden = MLP(self.text_emb_dim, config.num_classes, [])
        self.max_token_len = config.max_token_len
        self.enable_positional_encoding = not config.disable_positional_encoding
        if self.enable_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model=self.text_emb_dim, max_len=config.max_sequence_len)
        self.textlinear = nn.Linear(config.text_in_emb_dim, config.text_emb_dim)
        configuration = BertConfig(
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            hidden_size=self.text_emb_dim,
            intermediate_size=1024
        )
        self.encoder = BertEncoder(configuration)
        roberta_cfg = XLMRobertaConfig.from_pretrained(
            "xlm-roberta-base",
            num_attention_heads=12,
            num_hidden_layers=config.text_encoder_num_hidden_layer
        )
        self.text_roberta = XLMRobertaModel(roberta_cfg)

    def forward(self, x):
        token_ids, token_masks = x
        token_ids = token_ids.view(-1, self.max_token_len)
        token_masks = token_masks.view(-1, self.max_token_len)
        text_output = self.text_roberta(input_ids=token_ids, attention_mask=token_masks)
        all_text_emb = text_output.pooler_output.reshape(-1, self.max_sequence_len, self.text_in_emb_dim)
        text_x = self.textlinear(all_text_emb)
        features = [text_x]
        text_visual_x = torch.cat(features, 2)
        if self.enable_positional_encoding:
            text_visual_x = text_visual_x.permute(1, 0, 2)
            text_visual_x = self.pos_encoder(text_visual_x)
            text_visual_x = text_visual_x.permute(1, 0, 2)
        if 'bert' in self.model_version:
            emb_output = self.encoder(text_visual_x, head_mask=[None, None, None])[0]
        else:
            emb_output = text_visual_x
        x_hidden = self.hidden(emb_output)
        return self.sigmoid(x_hidden)
```
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
