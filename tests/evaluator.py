import json
import time

from relevance_classification import evaluate_relevance_classification
from content_type_matching    import evaluate_content_type_matching
from semantic_matching        import evaluate_semantic_matching
from field_extraction         import evaluate_field_extraction

if __name__ == "__main__":
    # load inputs
    labeled   = json.load(open("labeled_urls.json", "r", encoding="utf-8"))
    extracted = json.load(open("extracted_content.json", "r", encoding="utf-8"))
    annotated = json.load(open("annotated_extraction.json", "r", encoding="utf-8"))

    if not isinstance(labeled, list):   labeled   = [labeled]
    if not isinstance(extracted, list): extracted = [extracted]
    if not isinstance(annotated, list): annotated = [annotated]

    # attach content into labeled, if needed
    content_map = { item["url"]: item["content"] for item in extracted }
    for entry in labeled:
        entry["content"] = content_map.get(entry["url"], "")

    # run all stages
    rel = evaluate_relevance_classification(labeled)
    ctm = evaluate_content_type_matching(labeled)
    sem = evaluate_semantic_matching(labeled)
    fld = evaluate_field_extraction(annotated)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "relevance_classification": rel,
        "content_type_matching":    ctm,
        "semantic_matching":        sem,
        "field_extraction":         fld
    }
    print(json.dumps(report, indent=2))
