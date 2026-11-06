// Render brief.json (written by src/daily_brief.py) as one card per class plus an "uncertain" card.
// textContent is used instead of innerHTML, so a headline can never inject HTML into the page.

const groupOrder = ["World", "Sports", "Business", "Sci/Tech"];

// CSS class for each category's colour ("Sci/Tech" -> "scitech")
const slug = (name) => name.toLowerCase().replace(/[^a-z]/g, "");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function articleItem(article, showGuess) {
  const li = el("li");
  const link = el("a", "", article.title);
  link.href = article.link;
  link.target = "_blank";
  link.rel = "noopener";
  li.append(link);

  const metaLine = el("div", "meta");
  metaLine.append(el("span", "source", article.source), " · ");
  // round down, so 0.8957 shows as 0.89 and never looks like it reached the 0.9 threshold
  const conf = (Math.floor(article.confidence * 100) / 100).toFixed(2);
  metaLine.append(el("span", "conf", `conf ${conf}`));
  if (showGuess) {
    // for uncertain articles, show what the model would have picked, in that category's colour
    metaLine.append(" · best guess ", el("span", `guess ${slug(article.class)}`, article.class));
  }
  li.append(metaLine);
  return li;
}

function groupCard(name, articles, uncertain) {
  const card = el("section", `card ${slug(name)}`);
  const title = el("h2", "", uncertain ? "Uncertain" : name);
  title.append(el("span", "count", String(articles.length)));
  card.append(title);

  if (uncertain) {
    card.append(el("p", "note", "The model was not confident enough to file these."));
  }
  if (articles.length === 0) {
    card.append(el("p", "empty", "No articles today."));
    return card;
  }

  const list = el("ul");
  for (const article of articles) list.append(articleItem(article, uncertain));
  card.append(list);
  return card;
}

// date line, summary numbers and the coloured bar showing each topic's share
function renderHeader(brief) {
  const [day, time] = brief.generated_at.split(" ");
  const date = new Date(`${day}T00:00:00Z`).toLocaleDateString("en-GB", {
    weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC",
  });
  document.getElementById("date").textContent = `${date} · updated ${time} UTC`;

  const uncertain = brief.groups.uncertain.length;
  const classified = brief.total - uncertain;
  const stats = [
    [brief.total, "articles"],
    [`${Math.round((classified / brief.total) * 100)}%`, "auto-sorted"],
    [uncertain, "uncertain"],
    [`≥ ${brief.threshold}`, "confidence threshold"],
  ];
  const statsBox = document.getElementById("stats");
  for (const [value, label] of stats) {
    const stat = el("div", "stat", label);
    stat.prepend(el("strong", "", String(value)));
    statsBox.append(stat);
  }

  const mix = document.getElementById("mix");
  const legend = document.getElementById("legend");
  for (const name of [...groupOrder, "uncertain"]) {
    const count = brief.groups[name].length;
    const color = `var(--${name === "uncertain" ? "warn" : slug(name)})`;
    const part = el("span");
    part.style.width = `${(count / brief.total) * 100}%`;
    part.style.background = color;
    part.title = `${name}: ${count}`;
    mix.append(part);

    const item = el("span", "", `${name === "uncertain" ? "Uncertain" : name} ${count}`);
    const dot = el("i");
    dot.style.background = color;
    item.prepend(dot);
    legend.append(item);
  }
}

async function main() {
  const status = document.getElementById("status");
  try {
    // no-store: always fetch today's file instead of a cached one
    const response = await fetch("brief.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const brief = await response.json();

    renderHeader(brief);

    const layout = document.getElementById("layout");
    for (const name of groupOrder) layout.append(groupCard(name, brief.groups[name] || [], false));
    layout.append(groupCard("uncertain", brief.groups.uncertain, true));
    status.remove();
  } catch (err) {
    status.textContent = `Could not load brief.json (${err.message}). Run python -m src.daily_brief first.`;
    status.classList.add("error");
  }
}

main();
