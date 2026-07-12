# Contributing

## Add a source

1. Check for duplicate titles, canonical URLs, DOI/arXiv IDs, and earlier
   versions.
2. Verify the entry against a canonical paper page, official lab page, or
   official artifact repository.
3. Add one row to `data/catalog.csv`. Use `|` between controlled topic tags.
4. State relevance as a concrete contribution or evaluation, not praise.
5. Set `review_stage` honestly. New entries normally begin as `discovered` or
   `screened`.
6. Run `make check` and regenerate `CATALOG.md` with `make catalog`.

## Exclude a candidate

Do not delete a screened candidate without a trace. Add it to
`data/exclusions.csv` with a short exclusion code and explanation. Duplicates
should point to the retained catalog ID.

## Write a paper note

Copy `notes/TEMPLATE.md` to `notes/papers/<catalog-id>.md`. Distinguish the
authors’ claims from your assessment, cite page/section/figure locations, and
write at least one plausible alternative explanation or falsification test.

## Corrections

Corrections to metadata, scope, or interpretation are welcome. When changing a
substantive judgment, explain the evidence in the commit or pull request.
