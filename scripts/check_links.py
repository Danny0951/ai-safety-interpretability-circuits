#!/usr/bin/env python3
"""Check unique catalog URLs without downloading linked papers."""

from __future__ import annotations

import argparse
import csv
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog.csv"
URL_FIELDS = ("canonical_url", "pdf_url", "code_url", "data_url")
MARKDOWN_URL = re.compile(r"(?:\]\(|href=[\"'])(https://[^\s)\"']+)")
SOFT_SUCCESS = {401, 403, 405, 429}
USER_AGENT = (
    "Mozilla/5.0 (compatible; literature-link-checker/0.1; "
    "+https://github.com/Danny0951/ai-safety-interpretability-circuits)"
)


def check(url: str, timeout: float) -> tuple[str, int | None, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return url, response.status, "ok"
    except urllib.error.HTTPError as error:
        if error.code in SOFT_SUCCESS:
            return url, error.code, "reachable-but-restricted"
        if error.code not in {400, 501}:
            return url, error.code, "http-error"
    except (urllib.error.URLError, TimeoutError) as error:
        return url, None, f"network-error: {error}"

    fallback = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(fallback, timeout=timeout) as response:
            return url, response.status, "ok"
    except urllib.error.HTTPError as error:
        if error.code in SOFT_SUCCESS:
            return url, error.code, "reachable-but-restricted"
        return url, error.code, "http-error"
    except (urllib.error.URLError, TimeoutError) as error:
        return url, None, f"network-error: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--include-docs", action="store_true")
    args = parser.parse_args()

    with CATALOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    urls = sorted(
        {
            row[field].strip()
            for row in rows
            for field in URL_FIELDS
            if (row.get(field) or "").strip()
        }
    )
    if args.include_docs:
        course_urls = {
            match.group(1).rstrip(".,;")
            for path in (ROOT / "docs").rglob("*.md")
            for match in MARKDOWN_URL.finditer(path.read_text(encoding="utf-8"))
        }
        urls = sorted(set(urls) | course_urls)

    failures: list[tuple[str, int | None, str]] = []
    restricted = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check, url, args.timeout): url for url in urls}
        for future in as_completed(futures):
            url, status, result = future.result()
            if result == "reachable-but-restricted":
                restricted += 1
            elif result != "ok":
                failures.append((url, status, result))

    print(
        f"Checked {len(urls)} unique URLs: "
        f"{len(urls) - restricted - len(failures)} ok, "
        f"{restricted} restricted/rate-limited, {len(failures)} failed."
    )
    for url, status, result in sorted(failures):
        print(f"FAIL {status or '-'} {result}: {url}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
