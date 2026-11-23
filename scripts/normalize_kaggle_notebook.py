from __future__ import annotations

import ast
import copy
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


class NotebookNormalizationError(RuntimeError):
    pass


def _locked_ruff_version(repo_root: Path) -> str:
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    versions = {package["version"] for package in lock["package"] if package["name"] == "ruff"}
    if len(versions) != 1:
        raise NotebookNormalizationError("uv.lock must pin exactly one Ruff version")
    return versions.pop()


def _text(source: str | list[str]) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        return "".join(source)
    raise NotebookNormalizationError("notebook cell source must be text or a text list")


def _without_cell_sources(notebook: dict[str, Any]) -> dict[str, Any]:
    preserved = copy.deepcopy(notebook)
    for cell in preserved["cells"]:
        cell.pop("source", None)
    return preserved


def normalize_notebook(path: Path) -> bool:
    """Format a notebook with locked Ruff, preserving every non-source field.

    Returns True if the file changed. Formatting occurs on a sibling temporary
    copy. The target is atomically replaced only after JSON, output, metadata,
    execution-count, Markdown-source and code-AST checks pass and the target has
    not changed since it was read.
    """
    path = path.resolve(strict=True)
    if path.suffix != ".ipynb":
        raise NotebookNormalizationError("input must be an .ipynb file")

    repo_root = Path(__file__).resolve().parents[1]
    locked_version = _locked_ruff_version(repo_root)
    installed_version = importlib.metadata.version("ruff")
    if installed_version != locked_version:
        raise NotebookNormalizationError(
            f"Ruff {installed_version} is installed; uv.lock requires {locked_version}. "
            "Run this script through `uv run`."
        )

    original_bytes = path.read_bytes()
    try:
        before = json.loads(original_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookNormalizationError(f"invalid notebook JSON: {exc}") from exc
    if not isinstance(before, dict) or not isinstance(before.get("cells"), list):
        raise NotebookNormalizationError("notebook must contain a cells list")
    for cell in before["cells"]:
        if not isinstance(cell, dict) or cell.get("cell_type") not in {
            "code",
            "markdown",
            "raw",
        }:
            raise NotebookNormalizationError("notebook contains an invalid cell")
        _text(cell.get("source", []))

    mode = stat.S_IMODE(path.stat().st_mode)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}.normalize-",
            suffix=".ipynb",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(original_bytes)
        os.chmod(temp_path, mode)

        subprocess.run(
            [sys.executable, "-m", "ruff", "format", str(temp_path)],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        formatted_bytes = temp_path.read_bytes()
        try:
            after = json.loads(formatted_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NotebookNormalizationError(f"Ruff produced invalid notebook JSON: {exc}") from exc

        if _without_cell_sources(before) != _without_cell_sources(after):
            raise NotebookNormalizationError(
                "Ruff changed notebook metadata, cell structure, execution counts, "
                "outputs, attachments, or another non-source field"
            )

        for old_cell, new_cell in zip(before["cells"], after["cells"], strict=True):
            old_source = _text(old_cell.get("source", []))
            new_source = _text(new_cell.get("source", []))
            if old_source == new_source:
                continue
            if old_cell["cell_type"] != "code":
                raise NotebookNormalizationError("Ruff changed Markdown or raw cell source")
            try:
                old_ast = ast.dump(ast.parse(old_source), include_attributes=False)
                new_ast = ast.dump(ast.parse(new_source), include_attributes=False)
            except SyntaxError as exc:
                raise NotebookNormalizationError(
                    "cannot prove code-cell formatting is semantics-preserving"
                ) from exc
            if old_ast != new_ast:
                raise NotebookNormalizationError("Ruff changed code-cell syntax beyond formatting")

        if formatted_bytes == original_bytes:
            return False
        if path.read_bytes() != original_bytes:
            raise NotebookNormalizationError(
                "notebook changed while normalization ran; refusing to replace it"
            )
        os.replace(temp_path, path)
        temp_path = None
        return True
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise NotebookNormalizationError(
            f"Ruff normalization failed: {detail or exc.returncode}"
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: uv run python scripts/normalize_kaggle_notebook.py NOTEBOOK.ipynb",
            file=sys.stderr,
        )
        return 2
    try:
        changed = normalize_notebook(Path(sys.argv[1]))
    except (OSError, NotebookNormalizationError) as exc:
        print(f"normalization failed: {exc}", file=sys.stderr)
        return 1
    print("normalized notebook" if changed else "notebook already normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
