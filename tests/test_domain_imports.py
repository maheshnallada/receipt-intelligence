"""Domain must not import infrastructure, interfaces, or third-party stacks."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = (
    "app.application",
    "app.infrastructure",
    "app.interfaces",
    "fastapi",
    "pydantic",
    "redis",
    "fitz",
    "pymupdf",
    "torch",
    "logfire",
    "cachetools",
)


def test_domain_imports_are_stdlib_or_self() -> None:
    root = Path(__file__).resolve().parents[1] / "app" / "domain"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == banned or name.startswith(f"{banned}.") for banned in FORBIDDEN_ROOTS):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []
