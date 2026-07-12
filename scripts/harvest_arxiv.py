#!/usr/bin/env python3
"""Harvest high-recall arXiv candidates from versioned review queries.

The output is candidate metadata, not the included catalog. Abstracts are not
copied into the repository; screening should verify each canonical paper page.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data" / "search_queries.json"
OUTPUT_PATH = ROOT / "data" / "candidates" / "arxiv.csv"
COUNTS_PATH = ROOT / "data" / "candidates" / "arxiv-search-counts.csv"

API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 500
REQUEST_INTERVAL_SECONDS = 3.2
USER_AGENT = (
    "ai-safety-interpretability-circuits/0.1 "
    "(https://github.com/Danny0951/ai-safety-interpretability-circuits)"
)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}

FIELDS = (
    "arxiv_id",
    "title",
    "authors",
    "published",
    "updated",
    "canonical_url",
    "pdf_url",
    "primary_category",
    "categories",
    "matched_queries",
    "harvested_on",
    "screening_decision",
)


def compact(value: str) -> str:
    return " ".join(value.split())


def date_bound(value: str, end_of_day: bool = False) -> str:
    parsed = dt.date.fromisoformat(value)
    suffix = "2359" if end_of_day else "0000"
    return parsed.strftime("%Y%m%d") + suffix


def normalize_arxiv_id(entry_id: str) -> str:
    identifier = entry_id.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", identifier)


def request_feed(params: dict[str, str | int], retries: int = 5) -> ET.Element:
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return ET.fromstring(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * (2**attempt)))
    raise RuntimeError("unreachable")


def entry_to_row(entry: ET.Element, query_id: str, harvested_on: str) -> dict[str, str]:
    arxiv_id = normalize_arxiv_id(entry.findtext("atom:id", default="", namespaces=NS))
    authors = [
        compact(node.findtext("atom:name", default="", namespaces=NS))
        for node in entry.findall("atom:author", NS)
    ]
    categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", NS)]
    primary = entry.find("arxiv:primary_category", NS)
    return {
        "arxiv_id": arxiv_id,
        "title": compact(entry.findtext("atom:title", default="", namespaces=NS)),
        "authors": "; ".join(author for author in authors if author),
        "published": entry.findtext("atom:published", default="", namespaces=NS)[:10],
        "updated": entry.findtext("atom:updated", default="", namespaces=NS)[:10],
        "canonical_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "primary_category": primary.attrib.get("term", "") if primary is not None else "",
        "categories": "|".join(categories),
        "matched_queries": query_id,
        "harvested_on": harvested_on,
        "screening_decision": "pending",
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    start = date_bound(config["earliest_date"])
    end = date_bound(config["cutoff_date"], end_of_day=True)
    cap = int(config["arxiv_max_results_per_query"])
    harvested_on = dt.date.today().isoformat()
    category_filter = "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.CV OR cat:stat.ML)"

    records: dict[str, dict[str, str]] = {}
    counts: list[dict[str, str | int]] = []
    last_request_at = 0.0

    for query_spec in config["arxiv_queries"]:
        query_id = query_spec["id"]
        full_query = (
            f"{category_filter} AND ({query_spec['query']}) "
            f"AND submittedDate:[{start} TO {end}]"
        )
        start_index = 0
        total = None
        retrieved = 0

        while total is None or (start_index < min(total, cap)):
            elapsed = time.monotonic() - last_request_at
            if elapsed < REQUEST_INTERVAL_SECONDS:
                time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
            feed = request_feed(
                {
                    "search_query": full_query,
                    "start": start_index,
                    "max_results": min(PAGE_SIZE, cap - start_index),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
            )
            last_request_at = time.monotonic()
            total = int(feed.findtext("opensearch:totalResults", default="0", namespaces=NS))
            entries = feed.findall("atom:entry", NS)
            if not entries:
                break
            for entry in entries:
                row = entry_to_row(entry, query_id, harvested_on)
                existing = records.get(row["arxiv_id"])
                if existing is None:
                    records[row["arxiv_id"]] = row
                else:
                    matched = set(existing["matched_queries"].split("|"))
                    matched.add(query_id)
                    existing["matched_queries"] = "|".join(sorted(matched))
            retrieved += len(entries)
            start_index += len(entries)

        counts.append(
            {
                "search_date": harvested_on,
                "query_id": query_id,
                "query": full_query,
                "reported_results": total or 0,
                "retrieved_results": retrieved,
                "cap": cap,
            }
        )
        print(f"{query_id}: reported={total or 0}, retrieved={retrieved}", flush=True)

    rows = sorted(
        records.values(),
        key=lambda row: (row["published"], row["title"].casefold()),
        reverse=True,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with COUNTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=counts[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(counts)

    print(f"Wrote {len(rows)} unique candidates to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
