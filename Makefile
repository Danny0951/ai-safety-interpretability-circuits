.PHONY: bibliography catalog check course course-check course-serve harvest-arxiv links links-all

catalog:
	python3 scripts/render_catalog.py
	python3 scripts/render_bibliography.py

bibliography:
	python3 scripts/render_bibliography.py

check:
	python3 scripts/validate_catalog.py
	python3 scripts/validate_course.py
	python3 scripts/render_catalog.py
	python3 scripts/render_bibliography.py
	git diff --exit-code -- CATALOG.md
	git diff --exit-code -- BIBLIOGRAPHY.bib
	mkdocs build --strict

course:
	mkdocs build --strict

course-check:
	python3 scripts/validate_course.py
	mkdocs build --strict

course-serve:
	mkdocs serve

harvest-arxiv:
	python3 scripts/harvest_arxiv.py

links:
	python3 scripts/check_links.py

links-all:
	python3 scripts/check_links.py --include-docs
