import json
import streamlit as st
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="mistral")

def advanced_relevance_check(query: str, content: str, url: str, semantic_analysis: dict) -> tuple[bool, str]:
    st.info(f"Advanced Relevance Check for URL: {url}")
    prompt = f"""
ADVANCED SEMANTIC SIMILARITY VERIFICATION

IMPORTANT: You MUST use EXACTLY the following query for your relevance check, with NO modifications:
"{query}"

Key Search Terms: {', '.join(semantic_analysis.get('search_terms', [query]))}
URL: {url}

CONTENT SNIPPET:
{content[:10000]}

Based solely on the semantic meaning of the query above, evaluate if the content 
SEMANTICALLY addresses the user's information need. DO NOT consider superficial aspects like 
output formatting (tables, lists, etc.) or exact literal matching.

In your evaluation, consider:
1. Whether the content contains the TYPE of information being requested
2. The comprehensiveness of the content relative to the query
3. The reliability and quality of the information found

Return ONLY a valid JSON object with the following keys:
  - "relevance_score": a number between 0 and 100 (0 = completely irrelevant, 100 = perfectly relevant)
  - "is_relevant": a boolean indicating if the content is relevant to the query
  - "relevance_check_query": MUST be exactly "{query}" with no changes whatsoever
  - "justification": a detailed explanation of your reasoning
  - "matching_sections": a list of objects, each with "excerpt" and "context" keys
  - "information_coverage": an assessment of how much of the requested information is present

CRITICAL: The "relevance_check_query" field in your JSON MUST be EXACTLY: "{query}"
Do not include any additional text or formatting outside the JSON object.
"""
    try:
        resp = model.invoke(prompt)
        st.expander("Relevance Check Full Reasoning", expanded=False).write(resp)
        start = resp.find("{")
        end = resp.rfind("}") + 1
        if start != -1 and end != -1:
            js = resp[start:end]
            data = json.loads(js)
            if data.get("relevance_check_query") != query:
                st.warning("Model modified the query; overriding back.")
                data["relevance_check_query"] = query
            st.expander("Relevance Analysis Results", expanded=False).json(data)
            return data.get("is_relevant", False), json.dumps(data, indent=2)
        else:
            st.warning("No JSON object found in response.")
            return False, resp
    except Exception as e:
        st.error(f"Advanced Relevance Check Error: {e}")
        return False, str(e)
