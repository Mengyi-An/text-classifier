"""
Export a random sample of today's RSS articles for hand-labelling.

Two files are written to labels/:
    rss_sample.csv        id, title, source, link, label (empty) -> fill in the label column
    rss_sample_preds.csv  id, pred, confidence                   -> the model's answers, kept apart

The predictions are in a separate file so they cannot bias the labelling
(seeing the model's guess makes people agree with it).

Label with one of: World, Sports, Business, Sci/Tech, other
    World     international news, politics of any country (including US), war, diplomacy
    Sports    sport
    Business  companies, economy, markets, personal finance
    Sci/Tech  technology, science, health and medicine
    other     fits none of the four (entertainment, lifestyle, culture); not counted in accuracy

Usage:
    python -m src.export_labels
"""

import os
import csv
import json
import random

docs_path = os.path.join(os.path.dirname(__file__), "..", "docs")
labels_path = os.path.join(os.path.dirname(__file__), "..", "labels")
sample_file = os.path.join(labels_path, "rss_sample.csv")
preds_file = os.path.join(labels_path, "rss_sample_preds.csv")

sample_size = 100
seed = 42


def load_articles():
    """
    function: read docs/brief.json and put all groups (including "uncertain") back into one list
    output:
        [{"title": ..., "source": ..., "link": ..., "class": ..., "confidence": ...}, ...]
    """
    with open(os.path.join(docs_path, "brief.json")) as f:
        brief = json.load(f)
    articles = [a for group in brief["groups"].values() for a in group]
    print(f"brief from {brief['generated_at']}: {len(articles)} articles")
    return articles


def save_sample(articles):
    """
    function: pick sample_size articles at random (fixed seed), number them 1..n, write the two csv files
    output: labels/rss_sample.csv and labels/rss_sample_preds.csv
    """
    sample = random.Random(seed).sample(articles, min(sample_size, len(articles)))
    os.makedirs(labels_path, exist_ok=True)

    with open(sample_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "source", "link", "label"])
        for i, a in enumerate(sample, start=1):
            writer.writerow([i, a["title"], a["source"], a["link"], ""])

    with open(preds_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pred", "confidence"])
        for i, a in enumerate(sample, start=1):
            writer.writerow([i, a["class"], a["confidence"]])

    print(f"saved {len(sample)} articles to {sample_file}")
    print(f"saved predictions to {preds_file}")


if __name__ == "__main__":
    # never overwrite a file that may already contain hand-written labels
    if os.path.exists(sample_file):
        print(f"{sample_file} already exists; delete it first if you really want a new sample")
    else:
        save_sample(load_articles())
