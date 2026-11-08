"""
Build the daily news brief: fetch RSS -> classify -> group by class.
Prepare the news for the web page.

Articles with confidence >= confidence_threshold go to their predicted class,
the rest go to "uncertain". Only titles and links are kept (no full text, for copyright).

Output: docs/brief.json (read by docs/index.html, the GitHub Pages site)

Usage:
    python -m src.daily_brief
"""

import os
import json
from datetime import datetime, timezone
from transformers import AutoTokenizer
from src.data import model_name, classes
from src.predict import load_model, predict
from src.fetch_rss import fetch_all

docs_path = os.path.join(os.path.dirname(__file__), "..", "docs")

# chosen on the validation set with `python -m src.threshold`:
# conf >= 0.9 keeps 91.2% of articles at 97.3% accuracy
confidence_threshold = 0.9
batch_size = 64


def classify_articles(articles: list[dict], model, tokenizer):
    """
    function: predict the class and confidence of each article, in batches
    注意: 很多篇文章分成一批一批来预测, 不是切割同一篇文章。
    classify_articles：给每篇文章贴标签，加上 class 和 confidence。文章还是按原来抓取的顺序混在一起。
    """
    for i in range(0, len(articles), batch_size):
        batch = articles[i: i + batch_size] # batch[0]   = {"title": "Surg...", "source": "BBC", "link": ..., "text": ...}
        results = predict([a["text"] for a in batch], model, tokenizer)
        for article, result in zip(batch, results):
            article["class"] = result["class"]
            article["confidence"] = result["confidence"]


def build_brief(articles):
    """
    function: group the articles by class; low-confidence articles go to "uncertain"
    build_brief：按标签分堆。把同一类的文章放在一起，低于 0.9 的单独放一堆，每堆按置信度排好。
    output:
        {"generated_at": ..., "threshold": 0.9, "total": 202,
         "groups": {"World": [...], "Sports": [...], "Business": [...], "Sci/Tech": [...], "uncertain": [...]}}
    """
    groups = {name: [] for name in classes + ["uncertain"]}
    for a in articles:
        group = a["class"] if a["confidence"] >= confidence_threshold else "uncertain"
        groups[group].append({
            "title": a["title"],
            "link": a["link"],
            "source": a["source"],
            "published": a["published"],
            "class": a["class"],  # the model's guess, also shown for uncertain articles
            "confidence": a["confidence"],
        })

    # sort the news according confidence of each text (descending)
    for name in groups:
        groups[name].sort(key=lambda x: x["confidence"], reverse=True)

    brief = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "threshold": confidence_threshold,
        "total": len(articles),
        "groups": groups,
    }

    return brief


def save_brief(brief):
    """
    function: save the classified news to brief.json
    """
    os.makedirs(docs_path, exist_ok=True)
    out_path = os.path.join(docs_path, "brief.json")
    with open(out_path, "w") as f:
        # convert file format to json
        json.dump(brief, f, indent=2, ensure_ascii=False)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    articles = fetch_all()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model()

    classify_articles(articles, model, tokenizer)
    brief = build_brief(articles)
    for name, items in brief["groups"].items(): # .items(): get (key, value)
        print(f"{name}: {len(items)}")
    save_brief(brief)
