.PHONY: bibliography catalog check harvest-arxiv links

catalog:
	python3 scripts/render_catalog.py
	python3 scripts/render_bibliography.py

bibliography:
	python3 scripts/render_bibliography.py

check:
	python3 scripts/validate_catalog.py
	python3 scripts/render_catalog.py
	python3 scripts/render_bibliography.py
	git diff --exit-code -- CATALOG.md
	git diff --exit-code -- BIBLIOGRAPHY.bib

harvest-arxiv:
	python3 scripts/harvest_arxiv.py

links:
	python3 scripts/check_links.py
