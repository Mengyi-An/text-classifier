"""
Fetch news from a few RSS feeds, to be classified by the trained model.

Each article becomes "title + summary", the same format as AG News (headline + short description).
Articles that appear in more than one feed are kept only once.

Usage:
    python -m src.fetch_rss
"""

import re
import html
import feedparser

feeds = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "BBC Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Sport": "https://feeds.bbci.co.uk/sport/rss.xml",
    "The Guardian World": "https://www.theguardian.com/world/rss",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
}


def clean_text(text):
    """
    function: turn an RSS summary into plain text
        some feeds (e.g. The Guardian) put HTML tags, entities like &amp; and invisible
        zero-width characters in the summary
    output:
        "<p>Prices rose &amp; fell</p>" -> "Prices rose & fell"
    """
    text = re.sub(r"<[^>]+>", " ", text)              # drop HTML tags
    text = html.unescape(text)                         # &amp; -> &
    text = re.sub(r"[​-‏⁠]", "", text)  # drop zero-width characters
    return " ".join(text.split())                      # collapse spaces and newlines


def fetch_feed(source, url):
    """
    function: download one feed and keep the fields we need
    output:
        [{"title": ..., "summary": ..., "link": ..., "source": "BBC World", "published": ...}, ...]
    """
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        articles.append({
            "title": clean_text(entry.get("title", "")),
            "summary": clean_text(entry.get("summary", "")),
            "link": entry.get("link", ""),
            "source": source,
            "published": entry.get("published", ""),
        })
    return articles


def fetch_all():
    """
    function: fetch every feed and remove duplicates (same link or same title)
        a feed that fails to download is skipped, so one broken feed does not stop the rest
    output:
        list of articles, each with an extra "text" field (title + summary) for the model
    """
    articles = []
    seen = set()
    for source, url in feeds.items():
        try:
            feed_articles = fetch_feed(source, url)
        except Exception as e:
            print(f"skip {source}: {e}")
            continue

        for article in feed_articles:
            key_link = article["link"]
            key_title = article["title"].lower()
            if not article["title"] or key_link in seen or key_title in seen:
                continue
            seen.add(key_link)
            seen.add(key_title)

            article["text"] = f"{article['title']} {article['summary']}".strip()
            articles.append(article)
        print(f"{source}: {len(feed_articles)} articles")

    print(f"total after removing duplicates: {len(articles)}")
    return articles


if __name__ == "__main__":
    articles = fetch_all()
    for article in articles[:5]:
        print(f"[{article['source']}] {article['text'][:100]}")
