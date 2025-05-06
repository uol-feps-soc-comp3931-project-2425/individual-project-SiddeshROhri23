import time
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# model call from utils
from utils.relevance_check import advanced_relevance_check

# config for this module
MAX_WORKERS = 16

def _extract_text_from_content(content):
    if isinstance(content, dict):
        if "Text" in content:
            return str(content["Text"])
        elif "content" in content:
            return str(content["content"])
        else:
            import json
            return json.dumps(content)
    elif isinstance(content, str):
        return content
    else:
        try:
            return str(content)
        except:
            return ""

def evaluate_relevance_classification(data):
    """
    Returns a dict with precision, recall, f1, accuracy, sample_count,
    and latency_stats (mean, std, 95th percentile).
    """
    results = {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
        "sample_count": len(data),
        "latency_stats": {"mean": 0.0, "std": 0.0, "percentile_95": 0.0}
    }
    if not data:
        return results

    def _task(entry):
        q = entry["query"]
        c = _extract_text_from_content(entry.get("content", ""))
        url = entry["url"]
        true_label = entry.get("is_relevant", False)

        # simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]+\b", q)
        keywords = [
            w for w in words
            if len(w) > 3 and w.lower() not in ["what","when","where","which","that","this","those","these","with","from","about"]
        ]
        ctx = {
            "search_terms": keywords or q.split(),
            "expected_output": "relevant information about " + q,
            "filters": [{"content_type": "any", "recency": "any"}]
        }

        start = time.time()
        try:
            pred, _ = advanced_relevance_check(q, c[:15000], url, ctx)

            # extra density check
            if pred:
                density = sum(c.lower().count(kw.lower()) for kw in keywords) / (len(c.split()) or 1)
                if density < 0.005:
                    pred = False
        except Exception:
            pred = False

        latency = time.time() - start
        return true_label, pred, latency

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        out = list(exe.map(_task, data))

    y_true, y_pred, latencies = zip(*out)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    results.update({
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "accuracy":  float(acc),
        "latency_stats": {
            "mean":          float(np.mean(latencies)),
            "std":           float(np.std(latencies)),
            "percentile_95": float(np.percentile(latencies, 95))
        }
    })
    return results

if __name__ == "__main__":
    import json
    data = json.load(open("labeled_urls.json", "r", encoding="utf-8"))
    # ensure list
    if not isinstance(data, list):
        data = [data]
    metrics = evaluate_relevance_classification(data)
    print(json.dumps(metrics, indent=2))
