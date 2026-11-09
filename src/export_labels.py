"""
Export a random sample of today's RSS articles for hand-labelling.

Two files are written to labels/:
    rss_sample.csv        id, title, source, link, label (The column of label is empty which need to be labelled by hand) 
    rss_sample_preds.csv  id, pred, confidence                   

Label with one of: World, Sports, Business, Sci/Tech, other
    World     international news, politics, war, diplomacy
    Sports    sport
    Business  companies, economy, markets, personal finance
    Sci/Tech  technology, science, health and medicine
    other     fits none of the four (entertainment, lifestyle, culture); not counted in accuracy

Usage:
    python -m src.export_labels
"""

import os
import json
import csv
import random

docs_path = os.path.join(os.path.dirname(__file__), "..", "docs")
labels_path = os.path.join(os.path.dirname(__file__), "..", "labels")
sample_file = os.path.join(labels_path, "rss_sample.csv")
preds_file = os.path.join(labels_path, "rss_sample_preds.csv")

sample_size = 100
seed = 42


def load_articles():
    """
    function: read docs/brief.json and export 100 articles from brief.json
    output:
        [{"title": ..., "source": ..., "link": ..., "class": ..., "confidence": ...}, ...]
    """
    with open(os.path.join(docs_path, "brief.json")) as f:
        brief = json.load(f) # The type of brief is a dict
    groups = brief["groups"]
    articles = []
    for a in groups.values():
        articles.extend(a)
    generated_at = brief["generated_at"]
    print(f"brief from {generated_at}: {len(articles)} articles")
    return articles


def save_sample(articles):
    """
    function: pick sample_size articles at random (fixed seed), number them 1..n, write the two csv files
    output: labels/rss_sample.csv and labels/rss_sample_preds.csv
    """
    rng = random.Random(seed)
    export_num = min(sample_size, len(articles))   # export 100 samples, or export all articles if len(articles) < 100
    sample = rng.sample(articles, export_num)

    os.makedirs(labels_path, exist_ok=True)

    # sample file
    with open(sample_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "source", "link", "label"]) # column names
        for i, a in enumerate(sample, start=1): # index from 1
            writer.writerow([i, a["title"], a["source"], a["link"], ""])

    # pred file
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
