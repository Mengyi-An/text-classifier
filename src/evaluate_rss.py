"""
Evaluate the model on hand-labelled real RSS news (labels/rss_sample.csv from src/export_labels.py).

Compares with AG News to measure domain shift:
    accuracy on all labelled articles            vs AG News test acc 94.7%
    coverage / accuracy at the 0.9 threshold     vs 91.2% / 97.3% on the AG News validation set
    how many "other" articles (outside the four classes) were caught as uncertain

Usage:
    python -m src.evaluate_rss
"""

import csv
import numpy as np
from src.data import classes
from src.export_labels import sample_file, preds_file
from src.daily_brief import confidence_threshold


def load_labelled():
    """
    function: join the hand labels with the model's predictions by id
    output:
        [{"id": "1", "title": ..., "label": "World", "pred": "World", "confidence": 0.99}, ...]
        only rows that have a label
    """
    with open(preds_file) as f:
        preds = {row["id"]: row for row in csv.DictReader(f)}

    rows = []
    with open(sample_file) as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            if not label:
                continue
            if label not in classes + ["other"]:
                raise ValueError(f"id {row['id']}: unknown label '{label}', use one of {classes + ['other']}")
            row["label"] = label
            row["pred"] = preds[row["id"]]["pred"]
            row["confidence"] = float(preds[row["id"]]["confidence"])
            rows.append(row)
    return rows


def evaluate(rows):
    """
    function: print accuracy overall, at the threshold, and how "other" articles were handled
    output: printed report
    """
    in_scope = [r for r in rows if r["label"] != "other"]
    other = [r for r in rows if r["label"] == "other"]
    print(f"labelled: {len(rows)}  (in the four classes: {len(in_scope)}, other: {len(other)})")

    correct = np.array([r["pred"] == r["label"] for r in in_scope])
    confidences = np.array([r["confidence"] for r in in_scope])
    keep = confidences >= confidence_threshold

    print(f"\naccuracy, all articles:          {correct.mean():.1%}   (AG News test: 94.7%)")
    print(f"threshold {confidence_threshold}: coverage {keep.mean():.1%}, accuracy {correct[keep].mean():.1%}"
          f"   (AG News val: 91.2%, 97.3%)")

    if other:
        caught = np.mean([r["confidence"] < confidence_threshold for r in other])
        print(f"\n'other' articles sent to uncertain: {caught:.1%}")

    # the mistakes the page actually shows (confident but wrong) are the most useful to read
    shown_wrong = [r for r, c, k in zip(in_scope, correct, keep) if k and not c]
    print(f"\nconfident but wrong ({len(shown_wrong)}):")
    for r in shown_wrong:
        print(f"  [{r['label']} -> {r['pred']} {r['confidence']:.2f}] {r['title']}")


if __name__ == "__main__":
    evaluate(load_labelled())
