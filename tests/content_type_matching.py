import time
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from utils.content_match import verify_content_type

MAX_WORKERS = 16

def evaluate_content_type_matching(data):
    """
    Returns a dict with f1, precision, recall, sample_count, and latency_stats.
    """
    results = {
        "f1": 0.0, "precision": 0.0, "recall": 0.0,
        "sample_count": len(data),
        "latency_stats": {"mean": 0.0, "std": 0.0, "percentile_95": 0.0}
    }
    if not data:
        return results

    def _task(entry):
        q = entry["query"]
        content = str(entry.get("content", ""))[:10000]
        url = entry["url"]
        true_label = entry.get("matches_content_type", False)

        # infer expected type from query
        type_patterns = {
            "review": ["review","rating","opinion","critique"],
            "tutorial": ["tutorial","guide","how to","instructions","steps"],
            "news": ["news","latest","update","announced"],
            "product": ["product","specs","details","features"],
            "comparison": ["comparison","versus","vs","compare","difference"]
        }
        ql = q.lower()
        expected = None
        for ctype, pats in type_patterns.items():
            if any(p in ql for p in pats):
                expected = ctype
                break
        if not expected:
            expected = entry.get("expected_content_type", "information")

        ctx = {
            "intent_classification": {"content_type": expected, "expected_content": expected},
            "filters": [{"content_type": expected}]
        }

        start = time.time()
        try:
            pred, _ = verify_content_type(q, content, url, ctx)
        except Exception:
            pred = False
        latency = time.time() - start
        return true_label, pred, latency

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        out = list(exe.map(_task, data))

    y_true, y_pred, lat = zip(*out)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    results.update({
        "f1":        float(f1),
        "precision": float(p),
        "recall":    float(r),
        "latency_stats": {
            "mean":          float(np.mean(lat)),
            "std":           float(np.std(lat)),
            "percentile_95": float(np.percentile(lat, 95))
        }
    })
    return results

if __name__ == "__main__":
    import json
    data = json.load(open("labeled_urls.json", "r", encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]
    metrics = evaluate_content_type_matching(data)
    print(json.dumps(metrics, indent=2))
