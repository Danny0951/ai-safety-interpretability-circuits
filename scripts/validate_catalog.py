#!/usr/bin/env python3
"""Validate the systematic-review catalog using only the standard library."""

from __future__ import annotations

import csv
import datetime as dt
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog.csv"
TOPICS_FILE = ROOT / "data" / "topics.txt"

REQUIRED = {
    "id",
    "title",
    "year",
    "authors",
    "kind",
    "publication_status",
    "canonical_url",
    "topics",
    "scope",
    "access",
    "review_stage",
    "priority",
    "relevance",
    "added_on",
    "verified_on",
}

ENUMS = {
    "kind": {
        "paper",
        "article",
        "repository",
        "dataset",
        "model",
        "demo",
        "course",
        "book",
    },
    "publication_status": {
        "peer-reviewed",
        "preprint",
        "research-article",
        "research-note",
        "blog",
        "documentation",
        "tutorial",
    },
    "scope": {"core", "foundation", "safety-application", "peripheral"},
    "access": {"open", "abstract-only", "unknown"},
    "review_stage": {"discovered", "screened", "skimmed", "read", "replicated"},
    "priority": {"essential", "high", "normal", "reference"},
    "artifact_status": {"active", "maintained", "stable", "archived", "reference", "unknown"},
}

URL_FIELDS = ("canonical_url", "pdf_url", "code_url", "data_url")
DATE_FIELDS = ("date", "added_on", "verified_on")


def normalize_url(value: str) -> str:
    parts = urlsplit(value.strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def validate() -> list[str]:
    errors: list[str] = []
    topics = {
        line.strip()
        for line in TOPICS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    with CATALOG.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_fields = REQUIRED - fields
        if missing_fields:
            errors.append(f"catalog header missing fields: {sorted(missing_fields)}")
        rows = list(reader)

    ids: Counter[str] = Counter()
    urls: Counter[str] = Counter()
    current_year = dt.date.today().year

    for line_number, row in enumerate(rows, start=2):
        label = f"line {line_number} ({row.get('id') or 'missing-id'})"
        for field in REQUIRED:
            if not (row.get(field) or "").strip():
                errors.append(f"{label}: missing {field}")

        entry_id = (row.get("id") or "").strip()
        ids[entry_id] += 1
        if entry_id and (
            entry_id != entry_id.lower()
            or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in entry_id)
        ):
            errors.append(f"{label}: id must be lowercase kebab-case")

        try:
            year = int(row.get("year") or "")
            if year < 1980 or year > current_year + 1:
                errors.append(f"{label}: implausible year {year}")
        except ValueError:
            errors.append(f"{label}: year must be an integer")

        for field, allowed in ENUMS.items():
            value = (row.get(field) or "").strip()
            if value and value not in allowed:
                errors.append(f"{label}: invalid {field}={value!r}")

        for field in DATE_FIELDS:
            value = (row.get(field) or "").strip()
            if value:
                try:
                    dt.date.fromisoformat(value)
                except ValueError:
                    errors.append(f"{label}: {field} must be YYYY-MM-DD")

        for field in URL_FIELDS:
            value = (row.get(field) or "").strip()
            if value:
                parsed = urlsplit(value)
                if parsed.scheme != "https" or not parsed.netloc:
                    errors.append(f"{label}: {field} must be a valid https URL")
                if field == "canonical_url":
                    urls[normalize_url(value)] += 1

        entry_topics = [item.strip() for item in (row.get("topics") or "").split("|") if item.strip()]
        unknown_topics = set(entry_topics) - topics
        if unknown_topics:
            errors.append(f"{label}: unknown topics {sorted(unknown_topics)}")
        if len(entry_topics) != len(set(entry_topics)):
            errors.append(f"{label}: duplicate topic tag")

    for entry_id, count in ids.items():
        if entry_id and count > 1:
            errors.append(f"duplicate id {entry_id!r} appears {count} times")
    for url, count in urls.items():
        if url and count > 1:
            errors.append(f"duplicate canonical URL {url!r} appears {count} times")

    print(f"Validated {len(rows)} catalog entries against {len(topics)} topic tags.")
    return errors


if __name__ == "__main__":
    problems = validate()
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        raise SystemExit(1)
