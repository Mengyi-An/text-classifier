"""
Upload the trained model (data/model.pt) to the Hugging Face Hub, with a model card.

The daily GitHub Actions job downloads the model from there, so the 260MB file never
goes into the git repository. The repo is public, so downloading needs no token.

Log in once before running (paste a Write token when asked):
    hf auth login

Usage:
    python -m src.upload_model
"""

import os
import json
from huggingface_hub import HfApi
from src.data import data_path

repo_id = "callmezoey/agnews-distilbert"
model_path = os.path.join(data_path, "model.pt")
metrics_path = os.path.join(data_path, "metrics.json")


def model_card(metrics):
    """
    function: write the README.md shown on the model's Hub page
    output: markdown text
    """
    return f"""---
language: en
datasets:
- fancyzhx/ag_news
base_model: distilbert-base-uncased
pipeline_tag: text-classification
---

# DistilBERT fine-tuned on AG News

News topic classifier (World / Sports / Business / Sci/Tech): `distilbert-base-uncased`
with a dropout + linear head on the `[CLS]` vector, trained with a hand-written PyTorch loop.

| Metric | Value |
| --- | --- |
| Best val acc | {metrics['best_val_acc']:.2%} |
| Test acc | {metrics['test_acc']:.2%} |
| TF-IDF + LR baseline (test) | 91.72% |

At confidence >= 0.9 (chosen on the validation set), 91.2% of articles are classified at 97.3% accuracy.

`model.pt` is a PyTorch `state_dict` for the `TextClassifier` class in
[github.com/Mengyi-An/text-classifier](https://github.com/Mengyi-An/text-classifier) (`src/model.py`).
It powers a daily RSS news brief built by that repository.
"""


if __name__ == "__main__":
    with open(metrics_path) as f:
        metrics = json.load(f)

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=model_path, path_in_repo="model.pt", repo_id=repo_id)
    api.upload_file(path_or_fileobj=model_card(metrics).encode(), path_in_repo="README.md", repo_id=repo_id)
    print(f"uploaded to https://huggingface.co/{repo_id}")
