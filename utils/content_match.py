import json
import streamlit as st
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="mistral")

def semantic_content_match(query: str, content: str, url: str, semantic_analysis: dict) -> tuple[bool, str]:
    st.info(f"Semantic Content Match for URL: {url}")
    prompt = f"""
SEMANTIC CONTENT MATCH VERIFICATION

User query: "{query}"
URL: {url}

CONTENT SNIPPET:
{content[:8000]}

TASK:
Determine if this content SEMANTICALLY matches what the user is asking for in their query.
This goes beyond just checking the content type - verify if this content actually contains
what the user wants.

Return ONLY a valid JSON object with the following keys:
  - "is_semantic_match": a boolean
  - "semantic_match_score": a number 0–100
  - "what_user_wants": interpretation of the user's need
  - "what_content_provides": what this content actually provides
  - "justification": detailed explanation with examples
  - "missing_elements": key elements absent but required

Do not include any extra text outside the JSON.
"""
    try:
        resp = model.invoke(prompt)
        st.expander("Match Full Reasoning", expanded=False).write(resp)
        start = resp.find("{")
        end = resp.rfind("}") + 1
        if start != -1 and end != -1:
            js = resp[start:end]
            data = json.loads(js)
            st.expander("Semantic Match Analysis", expanded=False).json(data)
            return data.get("is_semantic_match", False), json.dumps(data, indent=2)
        else:
            st.warning("No JSON object found in semantic match response.")
            return False, resp
    except Exception as e:
        st.error(f"Semantic Content Match Error: {e}")
        return False, str(e)

def verify_content_type(query: str, content: str, url: str, semantic_analysis: dict) -> tuple[bool, str]:
    st.info(f"Content Type Verification for URL: {url}")
    # Determine expected type from semantic_analysis or infer from query
    expected = None
    ic = semantic_analysis.get("intent_classification", {})
    expected = ic.get("content_type") or ic.get("expected_content")
    filters = semantic_analysis.get("filters", {})
    if isinstance(filters, list):
        for f in filters:
            if isinstance(f, dict) and f.get("content_type"):
                expected = f["content_type"]
    elif isinstance(filters, dict) and filters.get("content_type"):
        expected = filters["content_type"]

    if not expected:
        indicators = {
            "review": ["review","rating","opinion","critique"],
            "news": ["news","latest","breaking"],
            "tutorial": ["tutorial","how to","step by step"],
            "comparison": ["compare","vs","versus"],
            "specification": ["specs","features","technical details"]
        }
        ql = query.lower()
        for ct, inds in indicators.items():
            if any(i in ql for i in inds):
                expected = ct
                break

    if not expected:
        expected = "information"

    prompt = f"""
CONTENT TYPE VERIFICATION

Query: "{query}"
Expected Content Type: {expected}
URL: {url}

CONTENT SNIPPET:
{content[:8000]}

TASK:
Determine if this content matches the expected type "{expected}".

Return ONLY a JSON object with:
  - "matches_content_type": boolean
  - "content_type_identified": actual type found
  - "confidence_score": 0–100
  - "explanation": reasoning
  - "key_indicators": list of supporting phrases

No extra text outside JSON.
"""
    try:
        resp = model.invoke(prompt)
        st.expander("Type Verification Details", expanded=False).write(resp)
        start = resp.find("{")
        end = resp.rfind("}") + 1
        if start != -1 and end != -1:
            js = resp[start:end]
            data = json.loads(js)
            st.expander("Content Type Analysis", expanded=False).json(data)
            return data.get("matches_content_type", False), json.dumps(data, indent=2)
        else:
            st.warning("No JSON object found in content type response.")
            return False, resp
    except Exception as e:
        st.error(f"Content Type Verification Error: {e}")
        return False, str(e)
