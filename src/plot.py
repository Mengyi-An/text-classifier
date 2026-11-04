"""
Plot training curves from data/metrics.json (written by src/train.py).

Draws train vs val loss and train vs val accuracy per epoch,
and saves the figure to figures/training_curves.png.

Usage:
    python -m src.plot
"""

import os
import json
import matplotlib.pyplot as plt
from src.data import data_path

figures_path = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_metrics():
    with open(os.path.join(data_path, "metrics.json")) as f:
        return json.load(f)


def plot_curves(metrics):
    # history is a list of dicts, one per epoch
    history = metrics["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    # left: loss, right: accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_loss, marker="o", label="train")
    ax1.plot(epochs, val_loss, marker="o", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.set_xticks(epochs)
    ax1.legend()

    ax2.plot(epochs, train_acc, marker="o", label="train")
    ax2.plot(epochs, val_acc, marker="o", label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xticks(epochs)
    ax2.legend()

    # show the test accuracy of the best model in the overall title
    fig.suptitle(f"DistilBERT on AG News (test acc {metrics['test_acc']:.4f})")

    os.makedirs(figures_path, exist_ok=True)
    out_path = os.path.join(figures_path, "training_curves.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    metrics = load_metrics()
    plot_curves(metrics)
