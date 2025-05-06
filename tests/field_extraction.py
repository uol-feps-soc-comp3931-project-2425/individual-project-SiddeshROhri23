import time
import re
import numpy as np
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

from utils.parsing import extract_fields_from_content

def evaluate_field_extraction(data):
    """
    Returns a dict with reviewer_extraction_f1, review_extraction_f1,
    field_level_f1, sample_count, and latency_stats.
    """
    results = {
        "field_level_f1":        0.0,
        "reviewer_extraction_f1":0.0,
        "review_extraction_f1":  0.0,
        "sample_count":          len(data),
        "latency_stats":         {"mean":0.0,"std":0.0,"percentile_95":0.0}
    }
    if not data:
        return results

    rouge = Rouge()
    sentence_model = SentenceTransformer("all-mpnet-base-v2")

    true_rev, pred_rev = [], []
    true_txt, pred_txt = [], []
    latencies = []

    for entry in data:
        url    = entry["url"]
        query  = entry["query"]
        gt_rev = entry["reviewer_true"]
        gt_txt = entry["review_true"]

        out, lat = extract_fields_from_content(url, entry.get("content", ""), query)
        true_rev.append(gt_rev)
        pred_rev.append(out["reviewer"])
        true_txt.append(gt_txt)
        pred_txt.append(out["review"])
        latencies.append(lat)

    # reviewer F1 via partial/full match
    matches = 0
    for tr, pr in zip(true_rev, pred_rev):
        trl, prl = tr.lower().strip(), pr.lower().strip()
        if trl == prl or trl in prl or prl in trl:
            matches += 1
    rev_f1 = matches / len(true_rev) if true_rev else 0.0

    # review text F1 via ROUGE-L (fallback to semantic sim)
    scores = []
    for tr, pr in zip(true_txt, pred_txt):
        try:
            r = rouge.get_scores(pr, tr)[0]["rouge-l"]["f"]
        except:
            # fallback
            te = sentence_model.encode(tr, convert_to_tensor=True)
            pe = sentence_model.encode(pr, convert_to_tensor=True)
            r = float(util.pytorch_cos_sim(te, pe).item())
        scores.append(r)
    txt_f1 = float(np.mean(scores)) if scores else 0.0

    fld_f1 = 0.4 * rev_f1 + 0.6 * txt_f1

    results.update({
        "reviewer_extraction_f1": rev_f1,
        "review_extraction_f1":   txt_f1,
        "field_level_f1":         fld_f1,
        "latency_stats": {
            "mean":          float(np.mean(latencies)),
            "std":           float(np.std(latencies)),
            "percentile_95": float(np.percentile(latencies, 95))
        }
    })
    return results

if __name__ == "__main__":
    import json
    data = json.load(open("annotated_extraction.json", "r", encoding="utf-8"))
    if not isinstance(data, list):
        data = [data]
    metrics = evaluate_field_extraction(data)
    print(json.dumps(metrics, indent=2))
