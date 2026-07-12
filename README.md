# AI Safety, Interpretability, and Circuits

A living systematic review, learning path, and reproducible research map for
mechanistic interpretability of modern neural networks, with an emphasis on
transformer circuits and safety-relevant applications.

Snapshot date: **2026-07-12**. The catalog is a work in progress; “collected”
does not mean “endorsed,” “read,” or “replicated.” Review state is tracked for
every source.

## What this repository is for

This project has two linked goals:

1. Build a systematic, auditable map of papers, research articles, code,
   datasets, models, demos, and serious educational resources.
2. Learn the field by progressing from reading to implementation, replication,
   comparison, and finally a small novel safety-oriented circuits project.

The review is deliberately narrower than “all AI safety” or “all explainable
AI.” A source is core when it helps reverse-engineer internal representations
or computations, evaluates whether such explanations are faithful, or uses
those methods to investigate a safety-relevant behavior.

## Start here

- Read the [review protocol](REVIEW_PROTOCOL.md) to understand what is and is
  not included.
- Follow the [learning path](LEARNING_PATH.md) rather than reading the catalog
  chronologically.
- Use the generated [catalog](CATALOG.md) to browse the current collection.
- Import the generated [BibTeX bibliography](BIBLIOGRAPHY.bib) into a reference
  manager; canonical pages remain authoritative for venue-specific metadata.
- Use the [taxonomy](TAXONOMY.md) to understand topic labels.
- Create notes from the [paper-note template](notes/TEMPLATE.md).

## Review states

| State | Meaning |
| --- | --- |
| `discovered` | Found by a search or citation trail; not yet screened in detail. |
| `screened` | Title/abstract or full resource checked against the protocol. |
| `skimmed` | Main claims, method, evidence, and limitations inspected. |
| `read` | Read closely and summarized in a structured note. |
| `replicated` | At least one central result or artifact was independently run. |

These labels prevent a large bibliography from being mistaken for mastery.

## Repository map

```text
.
├── data/
│   ├── catalog.csv          # Included and provisionally included sources
│   ├── candidates/          # Reproducible high-recall search outputs
│   ├── exclusions.csv       # Excluded candidates and explicit reasons
│   ├── search_queries.json  # Versioned scholarly-index queries
│   ├── search_log.csv       # Reproducible search history
│   └── topics.txt           # Controlled topic vocabulary
├── notes/
│   └── TEMPLATE.md          # Structured reading/replication note
├── scripts/
│   ├── render_catalog.py    # Builds CATALOG.md from catalog.csv
│   └── validate_catalog.py  # Checks schema, enums, URLs, tags, and duplicates
├── CATALOG.md               # Generated human-readable bibliography
├── LEARNING_PATH.md
├── REVIEW_PROTOCOL.md
└── TAXONOMY.md
```

## Local checks

Python 3.11+ is sufficient; catalog generation has no third-party dependencies.

```bash
make check
make catalog
make harvest-arxiv
make links
```

## Evidence standards

- Prefer canonical paper pages, official lab articles, and official artifact
  repositories over summaries.
- Record both positive results and critiques or null results.
- Separate localization, causal validation, human interpretability, and useful
  safety impact; success on one does not imply success on the others.
- Treat model-generated feature labels and explanations as hypotheses until
  independently tested.
- Record model, dataset, metric, intervention, compute, and released artifacts
  when a source is read closely.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Additions should include a canonical
source URL, topic tags, a scope judgment, and a falsifiable one-sentence reason
for inclusion.

## License

Code and original repository content are released under the MIT License. Linked
papers, articles, datasets, models, and repositories retain their own licenses.
