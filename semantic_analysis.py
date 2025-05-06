import re
import json
import streamlit as st
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="mistral")

def understand_user_intent(query: str) -> dict:
    st.info("Deeply understanding user query with semantic analysis...")
    prompt = f"""
You are an expert at analyzing and understanding the semantics behind user queries.

Analyze the following query in depth and provide a detailed, step-by-step breakdown of your reasoning. In your explanation include:
1. The main topics and keywords identified.
2. Any ambiguities or parts that might require clarification.
3. The intended output format (e.g., table, list, summary).
4. Any filters or special criteria mentioned.

After your detailed explanation, provide a JSON object with exactly these keys:
- "search_terms": a list of key search terms,
- "format": the desired output format (or null if not specified),
- "filters": a list of filters or criteria (or null),
- "clarifications": any clarifications or questions regarding the query (or null).

Do not include any additional text except for your detailed reasoning followed by the JSON object.

Query: {query}
    """
    try:
        full = model.invoke(prompt)
        st.text_area("Semantic Analysis", full, height=300)
        m = re.search(r'\{.*\}', full, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            st.json(data)
            return data
        else:
            st.warning("Could not parse JSON; falling back to simple terms.")
            terms = [line.strip() for line in full.splitlines() if line.strip()]
            return {"search_terms": terms, "format": None, "filters": None, "clarifications": None}
    except Exception as e:
        st.error(f"Error during semantic analysis: {e}")
        return {"search_terms":[query], "format":None, "filters":None, "clarifications":None}

def advanced_semantic_analysis(query: str) -> dict:
    st.info("Performing Advanced Semantic Analysis...")
    prompt = f"""
You are an EXPERT semantic analyst tasked with performing a COMPREHENSIVE multi-dimensional analysis of a user query.

QUERY: {query}

PERFORM A DETAILED ANALYSIS WITH THE FOLLOWING DIMENSIONS:

1. LINGUISTIC DECOMPOSITION:
   - Break down the query into fundamental linguistic components.
   - Identify parts of speech.
   - Extract key semantic roles (subject, verb, object).
   - Detect implied or hidden semantic intentions.

2. SEMANTIC INTENT CLASSIFICATION:
   - Categorize the query's primary intent (e.g., informational, comparative, analytical).
   - Identify secondary intents or sub-goals.
   - Determine the expected response format or structure.
   - Identify the expected content type (e.g., review, news, tutorial, comparison).

3. CONCEPTUAL MAPPING:
   - List all potential conceptual domains related to the query.
   - Identify potential knowledge domains.
   - Map abstract and concrete concepts mentioned.
   - Highlight potential ambiguities or multiple interpretations.

4. INFORMATION EXTRACTION STRATEGY:
   - Develop a detailed strategy for extracting relevant information.
   - Specify key search terms and alternative phrasings.
   - Outline potential data sources and search approaches.
   - Predict potential challenges in information retrieval.

5. GRANULARITY AND SPECIFICITY ANALYSIS:
   - Assess the level of detail requested.
   - Identify explicit and implicit specificity requirements.
   - Determine precision of information needed.

6. CONTEXTUAL INFERENCE:
   - Infer potential contextual background.
   - Identify any unstated assumptions.
   - Suggest potential clarifying questions.

7. DIFFERENTIATION AND OUTPUT SPECIFICATION:
   - Identify parts of the query that need to be differentiated (e.g., distinguishing between content extraction and formatting instructions).
   - Determine the specific type of output expected (e.g., table, list, summary, or detailed explanation).

8. CONTENT TYPE DETECTION:
   - Determine what type of content the user is looking for (e.g., review, news article, tutorial, etc.)
   - Look for keywords that indicate content type preferences (e.g., "reviews", "guide", "news")
   - Identify if the user has specific content source preferences

REQUIRED OUTPUT FORMAT:
Provide a comprehensive JSON object with exhaustive details covering all analysis dimensions. Include the following keys at a minimum:
  - "search_terms": a list of key search terms,
  - "expected_output": the expected output type (e.g., table, list, summary, or detailed explanation),
  - "content_type": the type of content the user is looking for (e.g., review, news, tutorial),
  - "filters": an object including content type filtering requirements,
  - "differentiation_points": an object detailing which aspects of the query should be treated differently,
  - "linguistic_decomposition": an object,
  - "semantic_intent_classification": an object,
  - "conceptual_mapping": an array,
  - "information_extraction_strategy": an object,
  - "granularity_and_specificity": an object,
  - "contextual_inference": an object

IMPORTANT: Return ONLY a valid JSON object without any extra text or commentary.
    """
    try:
        full = model.invoke(prompt)
        st.expander("Full Semantic Analysis Reasoning", expanded=False).write(full)
        start = full.find("{")
        end = full.rfind("}") + 1
        if start != -1 and end != -1:
            js = full[start:end]
            data = json.loads(js)
            enhanced = {
                "original_query": query,
                "linguistic_components": data.get("linguistic_decomposition", {}),
                "intent_classification": data.get("semantic_intent_classification", {}),
                "conceptual_domains": data.get("conceptual_mapping", []),
                "information_strategy": data.get("information_extraction_strategy", {}),
                "specificity_analysis": data.get("granularity_and_specificity", {}),
                "contextual_insights": data.get("contextual_inference", {}),
                "search_terms": data.get("search_terms", [query]),
                "expected_output": data.get("expected_output", "text"),
                "content_type": data.get("content_type"),
                "filters": data.get("filters", {}),
                "differentiation_points": data.get("differentiation_points", {}),
                "complexity_score": len(full)
            }
            st.expander("Enhanced Semantic Intent Data", expanded=False).json(enhanced)
            return enhanced
        else:
            st.warning("No JSON found in analysis.")
            return {"original_query": query, "search_terms":[query], "fallback_reasoning": full}
    except Exception as e:
        st.error(f"Advanced Semantic Analysis Error: {e}")
        return {"original_query": query, "search_terms":[query], "error": str(e)}
