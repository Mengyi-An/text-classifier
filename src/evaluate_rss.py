"""
Evaluate the model on hand-labelled real RSS news (labels/rss_sample.csv from src/export_labels.py).

Compares with AG News to measure domain shift:
    accuracy on all labelled articles            vs AG News test acc 94.7%
    coverage / accuracy at the 0.9 threshold     vs 91.2% / 97.3% on the AG News validation set

    Usage:
    python -m src.evaluate_rss
"""

import pandas as pd
from src.export_labels import sample_file, preds_file
from src.daily_brief import confidence_threshold


def load_labelled():
    """
    function: take "id", "label", "title" out from smaple_file then merge it with preds_file according to "id"
    output:
        "id", "pred", "conf", "label", "title"
    """
    labels = pd.read_csv(sample_file)
    preds = pd.read_csv(preds_file)

    id_label = labels[["id", "label", "title"]]
    final_table = preds.merge(id_label, on="id")      # id, pred, confidence, label, title
    return final_table


def evaluate(final_table):
    """
    function: print accuracy overall, at the threshold
    output: printed report
    """
    in_scope = final_table[final_table["label"] != "other"]       # rows in the four classes
    print(f"labelled: {len(final_table)}  (in the four classes: {len(in_scope)}")

    correct = in_scope["label"] == in_scope["pred"] # correct = [True, False, True, True]
    keep = in_scope["confidence"] >= confidence_threshold 
    
    acc_all = correct.mean() # accuracy for correct regardless conf
    coverage = keep.mean()
    acc_kept = correct[keep].mean() # accuracy for correct and conf >= 0.9       

    print(f"\naccuracy, all articles: {acc_all:.1%}   (AG News test: 94.7%)")
    print(f"threshold {confidence_threshold}: coverage {coverage:.1%}, accuracy {acc_kept:.1%}   (AG News val: 91.2%, 97.3%)")

    # confident but wrong: these are shown on the page with a wrong class
    shown_wrong = in_scope[keep & (correct == False)]
    num_wrong = len(shown_wrong) # the number of misclassified news
    print(f"\nconfident but wrong: {num_wrong}")

    wrong_table = shown_wrong[["label", "pred", "confidence", "title"]]
    print(wrong_table.to_string(index=False))


if __name__ == "__main__":
    evaluate(load_labelled())
