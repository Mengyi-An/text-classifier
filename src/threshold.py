"""
Choose the confidence threshold on the validation set.

For each threshold, texts with confidence >= threshold are classified, the rest go to "uncertain".
Prints coverage / accuracy per threshold and plots the curve.

Output: figures/confidence_threshold.png

We choose final threshold as 0.9 according to the confidence_threshold.png (This procedure is determined manually) 

Usage:
    python -m src.threshold
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from src.data import split_dataset, model_name, classes
from src.predict import load_model, predict
from src.plot import figures_path

batch_size = 64
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def predict_dataset(dataset, model, tokenizer):
    """
    function: compute confidence and correct for each batch of text
    output:
        confidences: numpy array, the confidence of each text, e.g. [0.99, 0.87, 0.42, ...]
        correct: numpy array, whether each prediction is right, e.g. [True, True, False, ...]
    """
    confidences = []
    correct = [] # the type of correct is [true, false...]
    for i in range(0, len(dataset), batch_size):
        texts = list(dataset["text"][i: i+batch_size])
        labels = dataset["label"][i: i+batch_size]

        results = predict(texts, model, tokenizer)

        for result, label in zip(results, labels):
            confidences.append(result["confidence"])
            correct.append(classes.index(result["class"]) == label) # label is index

    return np.array(confidences), np.array(correct)


def coverage_accuracy(confidences, correct, threshold):
    """
    function: keep the texts with confidence >= threshold, the rest go to "uncertain"
    output:
        cov: share of texts that are kept, e.g. 0.952
        acc: accuracy on the kept texts only, e.g. 0.981
    """
    keep = confidences >= threshold # # the type of keep is [true, false...]
    cov = keep.mean() # true is 1, false is 0
    acc = correct[keep].mean() # get the position which keep == true

    return cov, acc

def plot_threshold(confidences, correct):
    """
    function: plot coverage vs accuracy, one point per threshold
        a higher threshold keeps fewer texts (lower coverage) but they are more accurate
    output: figures/confidence_threshold.png
    """
    # many thresholds give a smooth curve; 0.25 is the lowest possible max probability for 4 classes
    curve = [coverage_accuracy(confidences, correct, t) for t in np.linspace(0.25, 0.99, 50)]
    coverages = [c for c, a in curve]
    accuracies = [a for c, a in curve]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverages, accuracies, label="DistilBERT (val)")

    # mark the thresholds that are printed in the table, so they can be read from the plot
    for t in thresholds:
        c, a = coverage_accuracy(confidences, correct, t)
        ax.plot(c, a, "o", color="black", markersize=4)
        ax.annotate(f"{t}", (c, a), textcoords="offset points", xytext=(5, -10), fontsize=8)

    ax.set_xlabel("coverage (share of texts classified)")
    ax.set_ylabel("accuracy on classified texts")
    ax.set_title("Confidence threshold on AG News validation")
    ax.legend(loc="lower left")

    os.makedirs(figures_path, exist_ok=True)
    out_path = os.path.join(figures_path, "confidence_threshold.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    _, val_set, _ = split_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model()

    confidences, correct = predict_dataset(val_set, model, tokenizer)
    for t in thresholds:
        cov, acc = coverage_accuracy(confidences, correct, t)
        print(f"threshold {t:.2f}  coverage {cov:.1%}  accuracy {acc:.1%}")
    plot_threshold(confidences, correct)


