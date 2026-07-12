#!/usr/bin/env python3
"""Structural quality checks for the visual course."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

MODULES = [
    "00-orientation",
    "01-transformer-anatomy",
    "02-circuit-algebra",
    "03-causal-experiments",
    "04-canonical-circuits",
    "05-automated-circuit-discovery",
    "06-superposition",
    "07-sparse-autoencoders",
    "08-transcoders-crosscoders",
    "09-attribution-graphs",
    "10-explanations-workspace",
    "11-steering",
    "12-safety-mechanisms",
    "13-auditing",
    "14-research-rigor",
    "15-project-design",
]

LABS = [
    "00-environment",
    "01-activation-cache",
    "02-activation-patching",
    "03-sae-features",
    "04-circuit-tracer",
    "05-jacobian-lens",
    "06-safety-audit",
    "07-explanation-showdown",
    "08-capstone",
]

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
WORD = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
PLACEHOLDER = re.compile(r"\b(?:TODO|TBD|FIXME)\b|PLACEHOLDER\s*:", re.IGNORECASE)


def check_lesson(path: Path, *, minimum_words: int, minimum_diagrams: int, module: bool) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing required lesson: {path.relative_to(ROOT)}"]

    text = path.read_text(encoding="utf-8")
    lower = text.casefold()
    words = len(WORD.findall(re.sub(r"```.*?```", "", text, flags=re.DOTALL)))
    diagrams = text.count("```mermaid")
    kind = "module" if module else "lab"

    if not text.startswith("---\n"):
        errors.append(f"{path.relative_to(ROOT)}: missing YAML front matter")
    if words < minimum_words:
        errors.append(f"{path.relative_to(ROOT)}: only {words} prose words; expected >= {minimum_words}")
    if diagrams < minimum_diagrams:
        errors.append(f"{path.relative_to(ROOT)}: only {diagrams} Mermaid diagrams; expected >= {minimum_diagrams}")
    if PLACEHOLDER.search(text):
        errors.append(f"{path.relative_to(ROOT)}: contains placeholder text")
    if text.count("```") % 2:
        errors.append(f"{path.relative_to(ROOT)}: unbalanced fenced code blocks")

    if module:
        expected = {
            "learning objectives": "learning objectives",
            "failure": "failure-mode section",
            "knowledge check": "knowledge check",
            "exercise": "exercise",
            "<details": "hidden answers",
        }
        for needle, description in expected.items():
            if needle not in lower:
                errors.append(f"{path.relative_to(ROOT)}: missing {description}")
    else:
        lab_requirements = {
            "objective": ("objective", "what you will", "outcome", "goal"),
            "procedure": ("procedure", "workflow", "step "),
            "deliverable": ("deliverable", "output", "submission", "artifact"),
        }
        for description, alternatives in lab_requirements.items():
            present = any(needle in lower for needle in alternatives)
            if description == "procedure" and re.search(r"^##\s+\d+[.·]", text, re.MULTILINE):
                present = True
            if not present:
                errors.append(f"{path.relative_to(ROOT)}: {kind} missing {description} section")

    if "https://" not in text:
        errors.append(f"{path.relative_to(ROOT)}: no primary-source or artifact link")

    print(f"{path.relative_to(ROOT)}: {words} words, {diagrams} diagrams")
    return errors


def check_internal_links() -> list[str]:
    errors: list[str] = []
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().split()[0].strip("<>")
            if target.startswith(("https://", "http://", "mailto:", "#", "javascript:")):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(ROOT)}: broken relative link {target!r} "
                    f"(resolved to {resolved})"
                )
    return errors


def main() -> None:
    errors: list[str] = []
    for name in MODULES:
        errors.extend(
            check_lesson(
                DOCS / "modules" / f"{name}.md",
                minimum_words=700,
                minimum_diagrams=2,
                module=True,
            )
        )
    for name in LABS:
        errors.extend(
            check_lesson(
                DOCS / "labs" / f"{name}.md",
                minimum_words=450,
                minimum_diagrams=1,
                module=False,
            )
        )
    errors.extend(check_internal_links())

    if errors:
        print("", file=sys.stderr)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(MODULES)} modules and {len(LABS)} labs.")


if __name__ == "__main__":
    main()
